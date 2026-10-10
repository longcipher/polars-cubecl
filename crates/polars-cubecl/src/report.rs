//! Execution report: proof that the engine actually ran.

/// A plan operator the engine executed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// A filter stage.
    Filter,
    /// A column projection stage.
    Project,
    /// A slice stage.
    Slice,
    /// A group-by aggregation stage.
    GroupBy,
}

/// Report attached to every engine result.
///
/// Differential tests assert [`ExecReport::engaged`] so a silently
/// falling-back engine can never pass: a test that claims to exercise a
/// kernel must observe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecReport {
    engaged: bool,
    backend: polars_cubecl_runtime::BackendKind,
    ops_ran: Vec<Op>,
}

impl ExecReport {
    /// Whether the engine executed (part of) the plan on the device.
    #[must_use]
    pub const fn engaged(&self) -> bool {
        self.engaged
    }

    /// The backend the engine ran (or would have run) on.
    #[must_use]
    pub const fn backend(&self) -> polars_cubecl_runtime::BackendKind {
        self.backend
    }

    /// Operators the engine executed, in order.
    #[must_use]
    pub fn ops_ran(&self) -> &[Op] {
        &self.ops_ran
    }

    /// Build an engaged report.
    #[must_use]
    pub const fn engaged_report(
        backend: polars_cubecl_runtime::BackendKind,
        ops_ran: Vec<Op>,
    ) -> Self {
        Self { engaged: true, backend, ops_ran }
    }

    /// Build a fallback report: the plan ran on the Polars CPU engine.
    #[must_use]
    pub const fn fallback_report(backend: polars_cubecl_runtime::BackendKind) -> Self {
        Self { engaged: false, backend, ops_ran: Vec::new() }
    }
}
