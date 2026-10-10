//! [`CubeclEngine`]: the public entry point.
//!
//! Collection flows through Polars' post-optimization hook
//! (`LazyFrame::_collect_post_opt`). The engine lowers the optimized IR into
//! its own [`CubePlan`](polars_cubecl_lower::CubePlan); when lowering or
//! execution cannot proceed, the plan is left untouched and Polars runs it on
//! the CPU engine. A computed result is spliced back as `IR::DataFrameScan`,
//! which Polars then simply returns.

use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_cubecl_lower::{LowerError, lower};
use polars_cubecl_runtime::{BackendKind, RuntimeContext};
use polars_error::PolarsError;
use polars_lazy::frame::LazyFrame;
use polars_plan::plans::{AExpr, IR};
use polars_utils::arena::{Arena, Node};

use crate::{EngineConfig, EngineError, exec, report::ExecReport};

/// Third-party execution engine for Polars, backed by CubeCL.
#[derive(Debug)]
pub struct CubeclEngine {
    config: EngineConfig,
    runtime: RuntimeContext,
}

impl CubeclEngine {
    /// Create an engine, selecting the configured backend.
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        let runtime = RuntimeContext::select(config.backend)?;
        Ok(Self { config, runtime })
    }

    /// The backend this engine runs on.
    #[must_use]
    pub const fn backend_kind(&self) -> BackendKind {
        self.runtime.kind()
    }

    /// Execute `lf`, falling back to the Polars CPU engine when the plan is
    /// unsupported, too small, or fails at runtime (unless strict).
    pub fn collect(&self, lf: LazyFrame) -> Result<DataFrame, PolarsError> {
        self.collect_inner(lf, false).map(|(frame, _)| frame)
    }

    /// Execute `lf`, erroring instead of falling back.
    pub fn collect_strict(&self, lf: LazyFrame) -> Result<DataFrame, PolarsError> {
        self.collect_inner(lf, true).map(|(frame, _)| frame)
    }

    /// [`CubeclEngine::collect`] with the execution report attached.
    pub fn collect_with_report(
        &self,
        lf: LazyFrame,
    ) -> Result<(DataFrame, ExecReport), PolarsError> {
        self.collect_inner(lf, false)
    }

    /// [`CubeclEngine::collect_strict`] with the execution report attached.
    pub fn collect_strict_with_report(
        &self,
        lf: LazyFrame,
    ) -> Result<(DataFrame, ExecReport), PolarsError> {
        self.collect_inner(lf, true)
    }

    fn collect_inner(
        &self,
        lf: LazyFrame,
        strict: bool,
    ) -> Result<(DataFrame, ExecReport), PolarsError> {
        let kind = self.runtime.kind();
        let strict = strict || self.config.raise_on_fail;
        let min_rows = self.config.min_rows_for_gpu;
        let runtime = &self.runtime;
        let mut report = ExecReport::fallback_report(kind);
        let frame = lf._collect_post_opt(|root, lp_arena, expr_arena, _opt_duration| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Phase 1: whole-plan fast path (optimal transfers). A
                // successful execution always engages, even transfer-only:
                // the hook, upload, splice, and download all ran.
                match lower(root, lp_arena, expr_arena) {
                    Ok(plan) => {
                        if plan.source_height() < min_rows {
                            return Ok(None);
                        }
                        let (computed, ops) =
                            exec::execute_plan(runtime, &plan).map_err(EngineError::from)?;
                        splice_result(root, lp_arena, computed);
                        Ok(Some(ops))
                    }
                    // Phase 2: hybrid partial substitution. Unsupported tails
                    // stay on the CPU engine; supported heads run on device.
                    // Transfer-only subtrees are skipped, so `None` means
                    // nothing ran.
                    Err(LowerError::Unsupported { .. }) => {
                        let ops = hybrid_node(root, lp_arena, expr_arena, runtime, min_rows);
                        Ok((!ops.is_empty()).then_some(ops))
                    }
                }
            }));
            match outcome {
                Ok(Ok(Some(ops))) => {
                    report = ExecReport::engaged_report(kind, ops);
                    Ok(())
                }
                Ok(Ok(None)) => {
                    if strict {
                        Err(PolarsError::ComputeError("plan outside the engine subset".into()))
                    } else {
                        Ok(())
                    }
                }
                Ok(Err(EngineError::Unsupported(reason))) => {
                    if strict {
                        Err(PolarsError::ComputeError(reason.into()))
                    } else {
                        Ok(())
                    }
                }
                Ok(Err(error)) => {
                    if strict {
                        Err(PolarsError::ComputeError(error.to_string().into()))
                    } else {
                        Ok(())
                    }
                }
                Err(_) => {
                    if strict {
                        Err(PolarsError::ComputeError(
                            "engine panicked during plan execution".into(),
                        ))
                    } else {
                        Ok(())
                    }
                }
            }
        })?;
        Ok((frame, report))
    }
}

/// Splice a computed frame back into the plan at `node`.
fn splice_result(node: Node, lp_arena: &mut Arena<IR>, computed: DataFrame) {
    let schema = computed.schema().clone();
    lp_arena.replace(node, IR::DataFrameScan {
        df: Arc::new(computed),
        schema,
        output_schema: None,
    });
}

/// Whether an op list represents real device work (as opposed to a pure
/// transfer/projection round-trip, which the hybrid driver skips to avoid
/// pointless traffic).
fn has_kernel_work(ops: &[crate::Op]) -> bool {
    ops.iter().any(|op| matches!(op, crate::Op::Filter | crate::Op::Slice | crate::Op::GroupBy))
}

/// Top-down hybrid substitution: execute the largest supported subtrees on
/// the device, leaving the rest for the CPU engine. Returns the ops that
/// ran on the device (possibly empty).
fn hybrid_node(
    node: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    runtime: &RuntimeContext,
    min_rows: usize,
) -> Vec<crate::Op> {
    if let Ok(plan) = polars_cubecl_lower::lower(node, lp_arena, expr_arena) &&
        plan.source_height() >= min_rows &&
        let Ok((computed, ops)) = exec::execute_plan(runtime, &plan) &&
        has_kernel_work(&ops)
    {
        splice_result(node, lp_arena, computed);
        return ops;
    }
    let mut ran = Vec::new();
    for child in polars_cubecl_lower::child_nodes(node, lp_arena) {
        ran.extend(hybrid_node(child, lp_arena, expr_arena, runtime, min_rows));
    }
    // Children may now be materialized scans; retry this level once.
    if ran.is_empty() {
        return ran;
    }
    if let Ok(plan) = polars_cubecl_lower::lower(node, lp_arena, expr_arena) &&
        plan.source_height() >= min_rows &&
        let Ok((computed, ops)) = exec::execute_plan(runtime, &plan) &&
        has_kernel_work(&ops)
    {
        splice_result(node, lp_arena, computed);
        return ops;
    }
    ran
}

/// Map [`LowerError`] into [`EngineError`].
impl From<LowerError> for EngineError {
    fn from(error: LowerError) -> Self {
        match error {
            LowerError::Unsupported { reason } => Self::Unsupported(reason),
        }
    }
}

/// Map executor failures into [`EngineError`].
impl From<exec::ExecError> for EngineError {
    fn from(error: exec::ExecError) -> Self {
        match error {
            exec::ExecError::Buffer(error) => Self::Execution(error.to_string()),
            exec::ExecError::Kernel(error) => Self::Execution(error.to_string()),
            exec::ExecError::BackendUnsupported(reason) => Self::Unsupported(reason),
            exec::ExecError::Launch(reason) |
            exec::ExecError::Inconsistent(reason) |
            exec::ExecError::NotExecutable(reason) => Self::Execution(reason),
        }
    }
}
