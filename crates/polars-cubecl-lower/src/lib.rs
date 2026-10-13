//! Lowering from Polars IR to the engine's [`CubePlan`].
//!
//! This is the only crate that reads Polars internals (`IR`, `AExpr`). It
//! walks the optimized plan handed over by `_collect_post_opt` and decides,
//! per node and per expression, whether the engine can execute it. Anything
//! outside the supported subset yields [`LowerError::Unsupported`], which the
//! facade turns into silent CPU fallback (or a strict-mode error in tests).
//!
//! The subset is intentionally narrow and conservative: widening it is a
//! deliberate, tested change, and every widening must survive the
//! differential suite plus mutation testing of the support predicates.

use std::sync::Arc;

use polars_core::{
    chunked_array::ops::sort::options::SortMultipleOptions,
    datatypes::{AnyValue, DataType},
    frame::DataFrame,
    schema::{Schema, SchemaRef},
};
use polars_cubecl_buffer::DType;
#[cfg(feature = "parquet")]
use polars_plan::dsl::ScanSource;
use polars_plan::{
    dsl::{FileScanIR, Operator, ScanSources, SinkTypeIR},
    plans::{
        AExpr, DynLiteralValue, FileInfo, IR, IRAggExpr, IRBooleanFunction, IRFunctionExpr,
        LiteralValue, expr_ir::ExprIR,
    },
};
#[cfg(feature = "parquet")]
use polars_utils::pl_path::PlRefPath;
use polars_utils::{
    arena::{Arena, Node},
    pl_str::PlSmallStr,
};
use thiserror::Error;

/// Error returned when a plan cannot be lowered.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LowerError {
    /// The plan (or part of it) is outside the supported subset.
    #[error("unsupported plan shape: {reason}")]
    Unsupported {
        /// Human-readable reason, naming the node/expression that was rejected.
        reason: String,
    },
}

/// A scalar literal in the engine's subset.
///
/// # Examples
///
/// ```
/// use polars_cubecl_lower::ScalarLit;
///
/// let lit = ScalarLit::I32(42);
/// assert_eq!(lit, ScalarLit::I32(42));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarLit {
    /// 32-bit signed integer literal.
    I32(i32),
    /// 64-bit signed integer literal.
    I64(i64),
    /// 32-bit unsigned integer literal.
    U32(u32),
    /// 64-bit unsigned integer literal.
    U64(u64),
    /// 32-bit float literal.
    F32(f32),
    /// 64-bit float literal.
    F64(f64),
    /// Boolean literal.
    Bool(bool),
}

impl ScalarLit {
    /// The Polars [`DataType`] of this literal.
    #[must_use]
    pub fn to_polars(self) -> DataType {
        match self {
            Self::I32(_) => DataType::Int32,
            Self::I64(_) => DataType::Int64,
            Self::U32(_) => DataType::UInt32,
            Self::U64(_) => DataType::UInt64,
            Self::F32(_) => DataType::Float32,
            Self::F64(_) => DataType::Float64,
            Self::Bool(_) => DataType::Boolean,
        }
    }
}

/// Binary operators for elementwise column-to-column expressions.
///
/// These mirror Polars' `Operator` variants re-mapped onto the engine's
/// type-safe kernel set. Both arithmetic and comparison operators are
/// represented: comparisons appear here too because `select(a < b)` must
/// produce a new boolean column, distinct from the filter predicate path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `left + right`.
    Add,
    /// `left - right`.
    Sub,
    /// `left * right`.
    Mul,
    /// `left / right` (true division).
    Div,
    /// `left // right` (floor division).
    FloorDiv,
    /// `left % right` (modulus; remainder has the sign of the dividend).
    Mod,
    /// `left == right`.
    Eq,
    /// `left != right`.
    NotEq,
    /// `left < right`.
    Lt,
    /// `left <= right`.
    LtEq,
    /// `left > right`.
    Gt,
    /// `left >= right`.
    GtEq,
}

impl BinOp {
    /// Polars [`Operator`] variant that maps to this [`BinOp`], if any.
    #[must_use]
    pub fn from_polars(op: Operator) -> Option<Self> {
        match op {
            Operator::Plus => Some(Self::Add),
            Operator::Minus => Some(Self::Sub),
            Operator::Multiply => Some(Self::Mul),
            Operator::RustDivide | Operator::TrueDivide => Some(Self::Div),
            Operator::FloorDivide => Some(Self::FloorDiv),
            Operator::Modulus => Some(Self::Mod),
            Operator::Eq => Some(Self::Eq),
            Operator::NotEq => Some(Self::NotEq),
            Operator::Lt => Some(Self::Lt),
            Operator::LtEq => Some(Self::LtEq),
            Operator::Gt => Some(Self::Gt),
            Operator::GtEq => Some(Self::GtEq),
            _ => None,
        }
    }
}

/// Unary (single-argument) elementwise operators on a column.
///
/// These are the math functions Polars lowers to a `Function` node — never a
/// binary op — and that run as a single elementwise kernel per column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `|x|` (absolute value).
    Abs,
    /// `⌈x⌉` (ceiling).
    Ceil,
    /// `⌊x⌋` (floor).
    Floor,
    /// `e^x` (exponential).
    Exp,
    /// `√x` (square root).
    Sqrt,
    /// `ln(1 + x)` (log1p).
    Log1p,
    /// `sin(x)`.
    Sin,
    /// `cos(x)`.
    Cos,
    /// `sign(x)` (`-1`, `0`, `+1` by sign, `NaN` stays `NaN`).
    Sign,
    /// `-x` (arithmetic negation).
    Negate,
}

/// An elementwise column expression in the engine's subset.
///
/// Trees of these drive `select(a + b)`, `select(sin(a))`, and similar
/// non-aggregate projections. Every leaf is either a plain column reference
/// or a [`ScalarLit`]; every interior node is a [`BinOp`] or a [`UnaryOp`].
///
/// # Examples
///
/// ```
/// use polars_cubecl_buffer::DType;
/// use polars_cubecl_lower::{BinOp, Expr, UnaryOp};
///
/// // `sin(a + 1)`, where `a` is a plain column.
/// let expr = Expr::Unary(
///     UnaryOp::Sin,
///     Box::new(Expr::BinOp(BinOp::Add, Box::new(Expr::col("a")), Box::new(Expr::lit_i32(1)))),
/// );
/// assert!(matches!(expr, Expr::Unary(..)));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A plain column reference.
    Col(PlSmallStr),
    /// A literal scalar value.
    Lit(ScalarLit),
    /// A binary operation between two sub-expressions.
    BinOp(BinOp, Box<Self>, Box<Self>),
    /// A unary operation on a sub-expression.
    Unary(UnaryOp, Box<Self>),
}

impl Expr {
    /// Build a plain-column reference.
    #[must_use]
    pub fn col(name: &str) -> Self {
        Self::Col(PlSmallStr::from(name))
    }

    /// Build a literal `i32`.
    #[must_use]
    pub fn lit_i32(value: i32) -> Self {
        Self::Lit(ScalarLit::I32(value))
    }

    /// Build a literal `f64`.
    #[must_use]
    pub fn lit_f64(value: f64) -> Self {
        Self::Lit(ScalarLit::F64(value))
    }

    /// Build a literal `bool`.
    #[must_use]
    pub fn lit_bool(value: bool) -> Self {
        Self::Lit(ScalarLit::Bool(value))
    }
}

/// Comparison operators supported by the predicate kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    NotEq,
    /// Less than.
    Lt,
    /// Less than or equal.
    LtEq,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    GtEq,
}

impl CmpOp {
    /// Flip the operator for `literal op column` shapes.
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Eq | Self::NotEq => self,
            Self::Lt => Self::Gt,
            Self::LtEq => Self::GtEq,
            Self::Gt => Self::Lt,
            Self::GtEq => Self::LtEq,
        }
    }

    /// Negate the operator for pushed-down `NOT` (`De Morgan` at the leaves).
    #[must_use]
    pub const fn negate(self) -> Self {
        match self {
            Self::Eq => Self::NotEq,
            Self::NotEq => Self::Eq,
            Self::Lt => Self::GtEq,
            Self::LtEq => Self::Gt,
            Self::Gt => Self::LtEq,
            Self::GtEq => Self::Lt,
        }
    }
}

/// A filter predicate in the engine's subset, in negation normal form.
///
/// Every variant evaluates to a null-excluding mask: a null input never
/// passes. `NOT` is pushed down to the leaves at lowering time (De Morgan),
/// so no variant needs separate validity tracking: for filter purposes,
/// Kleene `AND`/`OR` coincide with bitwise ops over null-excluding masks,
/// and negated leaves (`NotCompare` via flipped operators, `NotColumn` via
/// `!values & validity`) exclude nulls exactly.
///
/// # Examples
///
/// ```
/// use polars_core::prelude::PlSmallStr;
/// use polars_cubecl_buffer::DType;
/// use polars_cubecl_lower::{CmpOp, Predicate, ScalarLit};
///
/// let pred = Predicate::Compare {
///     column: PlSmallStr::from_static("a"),
///     kernel_dtype: DType::I32,
///     op: CmpOp::Gt,
///     literal: ScalarLit::I32(42),
/// };
/// assert_eq!(pred, pred.clone()); // predicates are value-comparable
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// `column op literal`, evaluated in `kernel_dtype`.
    ///
    /// `kernel_dtype` equals the column dtype except for the single allowed
    /// widening: `Int32`/`UInt32` columns compared against `f64` literals run
    /// in `Float64` (the CPU engine widens the integer side to `f64`, which is
    /// exact for 32-bit integers).
    Compare {
        /// Compared column.
        column: PlSmallStr,
        /// Dtype the comparison kernel runs in.
        kernel_dtype: DType,
        /// Comparison operator.
        op: CmpOp,
        /// Literal right-hand side, already validated against `kernel_dtype`.
        literal: ScalarLit,
    },
    /// Bare boolean column used as a mask (nulls do not pass).
    ColumnMask {
        /// Mask column.
        column: PlSmallStr,
    },
    /// Negated boolean column: `!values & validity` (nulls do not pass).
    NotColumn {
        /// Mask column.
        column: PlSmallStr,
    },
    /// Logical conjunction.
    And(Box<Self>, Box<Self>),
    /// Logical disjunction.
    Or(Box<Self>, Box<Self>),
}

/// A group-by aggregation in the engine's subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Aggregation {
    /// Sum; groups without valid values yield the zero identity.
    Sum {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Minimum; null for groups without valid values.
    Min {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Maximum; null for groups without valid values.
    Max {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Mean as `sum / count_valid` in `f64`; null for groups without valid values.
    Mean {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Count of valid (non-null) values.
    Count {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// First value in group order, null when it is null.
    First {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Last value in group order, null when it is null.
    Last {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Product; groups without valid values yield the identity `1`.
    Product {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Standard deviation as `sqrt(sum((x - mean)^2) / (n - ddof))` in `f64`.
    Std {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
        /// Delta degrees of freedom.
        ddof: u8,
    },
    /// Variance as `sum((x - mean)^2) / (n - ddof)` in `f64`.
    Var {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
        /// Delta degrees of freedom.
        ddof: u8,
    },
    /// Median; null for groups without valid values.
    Median {
        /// Aggregated column.
        input: PlSmallStr,
        /// Output column name.
        output: PlSmallStr,
    },
    /// Number of rows in the group.
    Len {
        /// Output column name.
        output: PlSmallStr,
    },
}

impl Aggregation {
    /// Source-column name read by this aggregation, if any.
    #[must_use]
    pub fn input_column(&self) -> Option<&PlSmallStr> {
        match self {
            Self::Sum { input, .. } |
            Self::Min { input, .. } |
            Self::Max { input, .. } |
            Self::Mean { input, .. } |
            Self::Count { input, .. } |
            Self::First { input, .. } |
            Self::Last { input, .. } |
            Self::Product { input, .. } |
            Self::Std { input, .. } |
            Self::Var { input, .. } |
            Self::Median { input, .. } => Some(input),
            Self::Len { .. } => None,
        }
    }
}

/// One pipeline step of a [`CubePlan`].
#[derive(Clone, Debug, PartialEq)]
pub enum PlanOp {
    /// Keep rows satisfying the predicate, preserving order.
    Filter(Predicate),
    /// Produce new columns from elementwise expressions. Each entry pairs an
    /// output name with an [`Expr`] tree; every input column referenced by
    /// the `Expr` trees must exist in the schema preceding this op.
    Project(Vec<(PlSmallStr, Expr)>),
    /// Keep `len` rows starting at `offset` (negative counts from the end).
    Slice {
        /// Start offset; negative counts from the end.
        offset: i64,
        /// Number of rows.
        len: u32,
    },
    /// Sort rows by a single column, optionally keeping a slice of the result.
    ///
    /// The engine sorts by downloading the sort column, computing the
    /// permutation on the host, then gathering all columns on the device.
    /// This keeps the device kernels simple and portable at the cost of a
    /// host/device round-trip for the sort column.
    Sort {
        /// Column to sort by.
        by: PlSmallStr,
        /// Sort direction per key (single entry: applies to `by`).
        descending: bool,
        /// Where nulls land (single entry: applies to `by`).
        nulls_last: bool,
        /// Optional `(offset, len)` window into the sorted output.
        slice: Option<(i64, usize)>,
    },
    /// Group rows by plain-column keys and aggregate.
    GroupBy {
        /// Key columns as `(output_name, input_name)` pairs, in output order.
        keys: Vec<(PlSmallStr, PlSmallStr)>,
        /// Aggregations, in output order.
        aggs: Vec<Aggregation>,
    },
}

/// The engine's plan: a source plus a linear pipeline of supported ops.
#[derive(Clone, Debug)]
pub struct CubePlan {
    source: PlanSource,
    ops: Vec<PlanOp>,
}

/// Plan inputs the engine can ingest.
#[derive(Clone, Debug)]
pub enum PlanSource {
    /// In-memory frame plus the columns to read (projection already applied).
    DataFrame {
        /// Input frame.
        frame: Arc<DataFrame>,
        /// Columns to read, in order.
        columns: Vec<PlSmallStr>,
    },
    /// Parquet file plus the columns to read (projection already applied).
    Parquet {
        /// File path.
        path: String,
        /// Columns to read, in order.
        columns: Vec<PlSmallStr>,
        /// Row-count estimate for the small-frame CPU policy (`usize::MAX`
        /// when unknown).
        height: usize,
    },
}

impl CubePlan {
    /// Plan source.
    #[must_use]
    pub const fn source(&self) -> &PlanSource {
        &self.source
    }

    /// Pipeline steps in execution order.
    #[must_use]
    pub fn ops(&self) -> &[PlanOp] {
        &self.ops
    }

    /// Row count of the plan source (used for the small-frame CPU policy).
    #[must_use]
    pub fn source_height(&self) -> usize {
        match &self.source {
            PlanSource::DataFrame { frame, .. } => frame.height(),
            PlanSource::Parquet { height, .. } => *height,
        }
    }

    /// Source-column names that must be uploaded for this plan to run.
    ///
    /// By default [`PlanSource`] lists *every* column of the source frame,
    /// but many plans touch only a fraction of them. For a plan like
    /// `select([col("b").add(col("b"))])` the default uploads four columns
    /// when only `"b"` reads from the source. This method walks the op
    /// pipeline tracking the live column set and marks only source columns
    /// that survive to an op reading them — skipping uploads (and the
    /// host→device DMA) for the rest.
    ///
    /// `Sort` and `Slice` are "wide" ops: they read every resident column,
    /// so they pull in whatever survived the previous step. `Filter` is
    /// narrow: it only reads the predicate columns. `GroupBy` is narrow
    /// (keys + agg inputs). `Project` replaces the live set with what the
    /// expressions reference.
    #[must_use]
    pub fn needed_source_columns(&self) -> Vec<PlSmallStr> {
        let mut needed = std::collections::HashSet::new();
        // `live` tracks the columns resident after each op. We seed it
        // from the source so wide ops (Sort/Slice) can mark everything.
        let source_columns: std::collections::HashSet<PlSmallStr> = match &self.source {
            PlanSource::DataFrame { columns, .. } | PlanSource::Parquet { columns, .. } => {
                columns.iter().cloned().collect()
            }
        };
        let mut live = source_columns.clone();
        for op in &self.ops {
            match op {
                PlanOp::Filter(predicate) => {
                    // Filter is narrow: only the predicate columns are
                    // read. The output keeps every live column, so `live`
                    // is unchanged; we just mark predicate columns needed.
                    predicate_columns(predicate, &mut needed);
                }
                PlanOp::Project(projection) => {
                    // Project narrows the live set: only columns referenced
                    // by the expressions survive. Output is the projection
                    // outputs — but those are derived, not source, columns.
                    // If a column is later needed again (unusual), a future
                    // op's column reference will mark it.
                    let mut referenced = std::collections::HashSet::new();
                    for (_, expr) in projection {
                        expr_columns(expr, &mut referenced);
                    }
                    // Mark referenced source columns as needed.
                    for name in &referenced {
                        if source_columns.contains(&**name) {
                            needed.insert(name.clone());
                        }
                    }
                    // If any referenced column is itself derived (not in
                    // source_columns), it must have been produced by a
                    // prior op, so the prior op already marked its inputs.
                    // The projection's outputs become the new live set for
                    // naming purposes, but since they're not source
                    // columns, they don't affect upload decisions.
                    // However, a later Sort/Slice needs *all* columns that
                    // physically exist on device at that point. We approximate
                    // by keeping union of source + prior derived live set:
                    // safe because uploading extra derived columns is
                    // impossible (they're produced on-device).
                    // The simplest correct approach: wide ops mark every
                    // currently-existing source column as needed.
                    // For now, keep source columns alive until a wide op.
                    let _ = &referenced;
                }
                PlanOp::Slice { .. } => {
                    // Slice is wide: it reads every resident column.
                    needed.extend(live.intersection(&source_columns).cloned());
                }
                PlanOp::Sort { by, .. } => {
                    // Sort is wide: it reads every resident column plus
                    // the sort key. Mark all live source columns.
                    needed.insert(by.clone());
                    needed.extend(live.intersection(&source_columns).cloned());
                }
                PlanOp::GroupBy { keys, aggs } => {
                    // GroupBy is narrow: reads keys + agg inputs.
                    for (_, input) in keys {
                        needed.insert(input.clone());
                    }
                    for agg in aggs {
                        if let Some(input) = agg.input_column() {
                            needed.insert(input.clone());
                        }
                    }
                    // Output is derived columns (group keys + agg results);
                    // the original resident columns can be freed, so we
                    // clear `live` to prevent false positives from later
                    // wide ops that don't actually see them.
                    live.clear();
                }
            }
        }
        // Preserve source order: deterministic column order in the result.
        match &self.source {
            PlanSource::DataFrame { columns, .. } | PlanSource::Parquet { columns, .. } => {
                columns.iter().filter(|name| needed.contains(&**name)).cloned().collect()
            }
        }
    }
}

/// All source-column names referenced by a predicate tree (recursively).
fn predicate_columns(predicate: &Predicate, sink: &mut std::collections::HashSet<PlSmallStr>) {
    match predicate {
        Predicate::Compare { column, .. } |
        Predicate::ColumnMask { column } |
        Predicate::NotColumn { column } => {
            sink.insert(column.clone());
        }
        Predicate::And(left, right) | Predicate::Or(left, right) => {
            predicate_columns(left, sink);
            predicate_columns(right, sink);
        }
    }
}

/// All source-column names referenced by an [`Expr`] tree (recursively).
fn expr_columns(expr: &Expr, sink: &mut std::collections::HashSet<PlSmallStr>) {
    match expr {
        Expr::Col(name) => {
            sink.insert(name.clone());
        }
        Expr::Lit(_) => {}
        Expr::BinOp(_, left, right) => {
            expr_columns(left, sink);
            expr_columns(right, sink);
        }
        Expr::Unary(_, inner) => {
            expr_columns(inner, sink);
        }
    }
}

/// Lower the optimized Polars IR at `root` into a [`CubePlan`].
///
/// Returns [`LowerError::Unsupported`] for anything outside the subset; the
/// caller must leave the plan untouched in that case.
pub fn lower(
    root: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> Result<CubePlan, LowerError> {
    let mut builder = Builder::new(expr_arena);
    builder.lower_node(root, lp_arena)?;
    builder.finish()
}

/// Child plan nodes of `node`, for hybrid top-down traversal.
///
/// Leaf sources (`DataFrameScan`, file `Scan`) and unknown shapes yield no
/// children; the hybrid driver simply does not descend into them.
#[must_use]
pub fn child_nodes(node: Node, lp_arena: &Arena<IR>) -> Vec<Node> {
    match lp_arena.get(node) {
        IR::Sink { input, .. } |
        IR::Filter { input, .. } |
        IR::Slice { input, .. } |
        IR::SimpleProjection { input, .. } |
        IR::Select { input, .. } |
        IR::Sort { input, .. } |
        IR::Cache { input, .. } |
        IR::GroupBy { input, .. } |
        IR::HStack { input, .. } |
        IR::Distinct { input, .. } |
        IR::MapFunction { input, .. } => vec![*input],
        IR::Gather { input, idxs, .. } => vec![*input, *idxs],
        IR::Join { input_left, input_right, .. } => vec![*input_left, *input_right],
        IR::Union { inputs, .. } |
        IR::HConcat { inputs, .. } |
        IR::SinkMultiple { inputs, .. } |
        IR::UnoptimizedDispatch { inputs, .. } => inputs.clone(),
        IR::ExtContext { input, contexts, .. } => {
            let mut children = Vec::with_capacity(contexts.len() + 1);
            children.push(*input);
            children.extend(contexts.iter().copied());
            children
        }
        IR::DataFrameScan { .. } | IR::Scan { .. } | IR::Invalid => Vec::new(),
    }
}

struct Builder<'a> {
    expr_arena: &'a Arena<AExpr>,
    source: Option<PlanSource>,
    ops: Vec<PlanOp>,
    schema: Vec<(PlSmallStr, DataType)>,
}

impl<'a> Builder<'a> {
    const fn new(expr_arena: &'a Arena<AExpr>) -> Self {
        Self { expr_arena, source: None, ops: Vec::new(), schema: Vec::new() }
    }

    fn finish(self) -> Result<CubePlan, LowerError> {
        let Some(source) = self.source else {
            return Err(unsupported("plan has no readable source"));
        };
        Ok(CubePlan { source, ops: self.ops })
    }

    fn lower_node(&mut self, node: Node, lp_arena: &Arena<IR>) -> Result<(), LowerError> {
        match lp_arena.get(node) {
            IR::Sink { input, payload } => {
                if !matches!(payload, SinkTypeIR::Memory) {
                    return Err(unsupported("only memory sinks are supported"));
                }
                self.lower_node(*input, lp_arena)?;
            }
            IR::DataFrameScan { df, output_schema, .. } => {
                self.lower_scan_source(df.clone(), output_schema.as_deref())?;
            }
            IR::Filter { input, predicate } => {
                self.lower_node(*input, lp_arena)?;
                let predicate = self.lower_predicate(predicate.node(), false)?;
                self.ops.push(PlanOp::Filter(predicate));
            }
            IR::SimpleProjection { input, columns } => {
                self.lower_node(*input, lp_arena)?;
                let mut projection = Vec::new();
                for name in columns.iter_names() {
                    self.require_column(name)?;
                    projection.push((name.clone(), Expr::Col(name.clone())));
                }
                if projection.is_empty() {
                    return Err(unsupported("empty projection"));
                }
                self.apply_projection(projection)?;
            }
            IR::Select { input, expr, .. } => {
                self.lower_node(*input, lp_arena)?;
                let mut projection = Vec::with_capacity(expr.len());
                for item in expr {
                    let output = output_name(item)?;
                    let lowered_expr = self.lower_expr(item.node())?;
                    projection.push((output, lowered_expr));
                }
                if projection.is_empty() {
                    return Err(unsupported("empty select"));
                }
                self.apply_projection(projection)?;
            }
            IR::Slice { input, offset, len } => {
                self.lower_node(*input, lp_arena)?;
                self.ops.push(PlanOp::Slice { offset: *offset, len: *len });
            }
            IR::GroupBy { input, keys, aggs, maintain_order, apply, .. } => {
                if apply.is_some() {
                    return Err(unsupported("group_by with python apply"));
                }
                if *maintain_order {
                    return Err(unsupported("group_by with maintain_order"));
                }
                self.lower_node(*input, lp_arena)?;
                let mut key_names = Vec::with_capacity(keys.len());
                for key in keys {
                    key_names.push((output_name(key)?, self.plain_column(key.node())?));
                }
                if key_names.is_empty() {
                    return Err(unsupported("group_by without keys"));
                }
                let mut lowered = Vec::with_capacity(aggs.len());
                for agg in aggs {
                    lowered.push(self.lower_aggregation(agg)?);
                }
                if lowered.is_empty() {
                    return Err(unsupported("group_by without aggregations"));
                }
                let mut seen = std::collections::HashSet::new();
                for (output, _) in &key_names {
                    if !seen.insert(output) {
                        return Err(unsupported("duplicate group_by key output"));
                    }
                }
                for agg in &lowered {
                    if !seen.insert(agg.output()) {
                        return Err(unsupported("duplicate group_by output"));
                    }
                }
                self.apply_group_by(key_names, lowered)?;
            }
            IR::Scan { sources, file_info, predicate, output_schema, scan_type, .. } => {
                self.lower_scan(sources, file_info, predicate, output_schema, scan_type)?;
            }
            IR::Cache { input, .. } => {
                self.lower_node(*input, lp_arena)?;
            }
            IR::Sort { input, by_column, slice, sort_options } => {
                self.lower_node(*input, lp_arena)?;
                // Reject multi-column sort and `maintain_order` for now. Single-key
                // sort covers the common path; the rest stays on the CPU engine.
                if by_column.len() != 1 {
                    return Err(unsupported("sort by multiple columns"));
                }
                let options: &SortMultipleOptions = sort_options;
                if options.maintain_order {
                    return Err(unsupported("sort with maintain_order"));
                }
                if options.limit.is_some() {
                    return Err(unsupported("sort with limit"));
                }
                let by = self.plain_column(by_column[0].node())?;
                let descending = options.descending.first().copied().unwrap_or(false);
                let nulls_last = options.nulls_last.first().copied().unwrap_or(false);
                let slice = slice.as_ref().map(|(offset, len, _)| (*offset, *len));
                self.ops.push(PlanOp::Sort { by, descending, nulls_last, slice });
            }
            IR::Join { .. } |
            IR::Gather { .. } |
            IR::HStack { .. } |
            IR::Distinct { .. } |
            IR::MapFunction { .. } |
            IR::Union { .. } |
            IR::HConcat { .. } |
            IR::ExtContext { .. } |
            IR::SinkMultiple { .. } |
            IR::UnoptimizedDispatch { .. } |
            IR::Invalid => {
                return Err(unsupported("node type outside the subset"));
            }
        }
        Ok(())
    }

    fn lower_scan_source(
        &mut self,
        frame: Arc<DataFrame>,
        projected: Option<&Schema>,
    ) -> Result<(), LowerError> {
        if self.source.is_some() {
            return Err(unsupported("multiple sources"));
        }
        let mut columns = Vec::new();
        let mut schema = Vec::new();
        match projected {
            Some(projection) => {
                for name in projection.iter_names() {
                    let dtype = frame
                        .column(name)
                        .map_err(|_| unsupported("projected column missing from frame"))?
                        .dtype()
                        .clone();
                    require_dtype(&dtype)?;
                    columns.push(name.clone());
                    schema.push((name.clone(), dtype));
                }
            }
            None => {
                for column in frame.columns() {
                    let dtype = column.dtype().clone();
                    require_dtype(&dtype)?;
                    columns.push(column.name().clone());
                    schema.push((column.name().clone(), dtype));
                }
            }
        }
        if columns.is_empty() {
            return Err(unsupported("source without columns"));
        }
        self.schema = schema;
        self.source = Some(PlanSource::DataFrame { frame, columns });
        Ok(())
    }

    fn lower_scan(
        &mut self,
        sources: &ScanSources,
        file_info: &FileInfo,
        predicate: &Option<ExprIR>,
        output_schema: &Option<SchemaRef>,
        scan_type: &FileScanIR,
    ) -> Result<(), LowerError> {
        #[cfg(feature = "parquet")]
        {
            if !matches!(scan_type, FileScanIR::Parquet { .. }) {
                return Err(unsupported("non-parquet file scan"));
            }
            let source = ScanSource::from_sources(sources.clone())
                .map_err(|_| unsupported("multi-source scan"))?;
            let ScanSource::Path(path) = source else {
                return Err(unsupported("non-path scan source"));
            };
            self.lower_parquet_source(path, file_info, output_schema.as_ref())?;
            if let Some(predicate) = predicate {
                let predicate = self.lower_predicate(predicate.node(), false)?;
                self.ops.push(PlanOp::Filter(predicate));
            }
            Ok(())
        }
        #[cfg(not(feature = "parquet"))]
        {
            // The parquet path above is the only one that mutates the
            // builder; without the feature nothing does. The reborrow keeps
            // the shared signature honest in both configurations.
            let _ = &mut *self;
            let _ = (sources, file_info, predicate, output_schema, scan_type);
            Err(unsupported("file scan needs the parquet feature"))
        }
    }

    #[cfg(feature = "parquet")]
    fn lower_parquet_source(
        &mut self,
        path: PlRefPath,
        file_info: &FileInfo,
        projected: Option<&SchemaRef>,
    ) -> Result<(), LowerError> {
        if self.source.is_some() {
            return Err(unsupported("multiple sources"));
        }
        let mut columns = Vec::new();
        let mut schema = Vec::new();
        match projected {
            Some(projection) => {
                for name in projection.iter_names() {
                    let dtype = file_info
                        .schema
                        .get(name)
                        .ok_or_else(|| unsupported("projected column missing from file"))?
                        .clone();
                    require_dtype(&dtype)?;
                    columns.push(name.clone());
                    schema.push((name.clone(), dtype));
                }
            }
            None => {
                for (name, dtype) in file_info.schema.iter() {
                    require_dtype(dtype)?;
                    columns.push(name.clone());
                    schema.push((name.clone(), dtype.clone()));
                }
            }
        }
        if columns.is_empty() {
            return Err(unsupported("source without columns"));
        }
        self.schema = schema;
        self.source = Some(PlanSource::Parquet {
            path: path.as_str().to_string(),
            columns,
            height: file_info.row_estimation.0.unwrap_or(usize::MAX),
        });
        Ok(())
    }

    fn require_column(&self, name: &PlSmallStr) -> Result<DataType, LowerError> {
        self.schema
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, dtype)| dtype.clone())
            .ok_or_else(|| unsupported("unknown column"))
    }

    fn plain_column(&self, node: Node) -> Result<PlSmallStr, LowerError> {
        match self.expr_arena.get(node) {
            AExpr::Column(name) => {
                self.require_column(name)?;
                Ok(name.clone())
            }
            _ => Err(unsupported("non-column expression where a column is required")),
        }
    }

    fn apply_projection(&mut self, projection: Vec<(PlSmallStr, Expr)>) -> Result<(), LowerError> {
        let mut schema = Vec::with_capacity(projection.len());
        let mut seen = std::collections::HashSet::new();
        for (output, expr) in &projection {
            if !seen.insert(output) {
                return Err(unsupported("duplicate output column in projection"));
            }
            let dtype = self.expr_dtype(expr)?;
            require_dtype(&dtype)?;
            schema.push((output.clone(), dtype));
        }
        self.schema = schema;
        self.ops.push(PlanOp::Project(projection));
        Ok(())
    }

    /// Infer the Polars [`DataType`] of a lowered [`Expr`] given the current
    /// schema. Mirrors Polars' type-promotion rules for the engine's subset.
    fn expr_dtype(&self, expr: &Expr) -> Result<DataType, LowerError> {
        match expr {
            Expr::Col(name) => self.require_column(name),
            Expr::Lit(lit) => Ok(lit.to_polars()),
            Expr::BinOp(op, left, right) => {
                let left_dtype = self.expr_dtype(left)?;
                let right_dtype = self.expr_dtype(right)?;
                // Comparison operators always yield `Boolean`.
                if matches!(
                    op,
                    BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq
                ) {
                    return Ok(DataType::Boolean);
                }
                promote_binary(left_dtype, right_dtype)
            }
            Expr::Unary(_, inner) => self.expr_dtype(inner),
        }
    }

    /// Translate a Polars `AExpr` into the engine's [`Expr`] subset.
    ///
    /// Supports plain columns, literals, binary arithmetic/comparison, and
    /// the unary math functions in [`UnaryOp`]. Anything else yields
    /// [`LowerError::Unsupported`].
    fn lower_expr(&self, node: Node) -> Result<Expr, LowerError> {
        match self.expr_arena.get(node) {
            AExpr::Column(name) => {
                self.require_column(name)?;
                Ok(Expr::Col(name.clone()))
            }
            AExpr::Literal(value) => {
                let scalar = self.lower_literal_to_scalar(value)?;
                Ok(Expr::Lit(scalar))
            }
            AExpr::BinaryExpr { left, op, right } => {
                let bin_op = BinOp::from_polars(*op)
                    .ok_or_else(|| unsupported("binary operator outside the subset"))?;
                let left = self.lower_expr(*left)?;
                let right = self.lower_expr(*right)?;
                Ok(Expr::BinOp(bin_op, Box::new(left), Box::new(right)))
            }
            AExpr::Function { input, function, .. } => {
                let op = match function {
                    IRFunctionExpr::Negate => UnaryOp::Negate,
                    IRFunctionExpr::Pow(f) => match f {
                        polars_plan::plans::IRPowFunction::Sqrt => UnaryOp::Sqrt,
                        _ => return Err(unsupported("pow function outside the subset")),
                    },
                    IRFunctionExpr::Abs => UnaryOp::Abs,
                    IRFunctionExpr::Ceil => UnaryOp::Ceil,
                    IRFunctionExpr::Floor => UnaryOp::Floor,
                    IRFunctionExpr::Exp => UnaryOp::Exp,
                    IRFunctionExpr::Log1p => UnaryOp::Log1p,
                    IRFunctionExpr::Sign => UnaryOp::Sign,
                    IRFunctionExpr::Trigonometry(f) => match f {
                        polars_plan::plans::IRTrigonometricFunction::Sin => UnaryOp::Sin,
                        polars_plan::plans::IRTrigonometricFunction::Cos => UnaryOp::Cos,
                        _ => return Err(unsupported("trig function outside the subset")),
                    },
                    _ => return Err(unsupported("function outside the subset")),
                };
                if input.len() != 1 {
                    return Err(unsupported("unary function with multiple inputs"));
                }
                let inner = self.lower_expr(input[0].node())?;
                // Float-only math ops (abs, ceil, floor, exp, sqrt, log1p,
                // sin, cos, sign) have no integer kernel — reject at lowering
                // so the engine cleanly falls back to the CPU.
                if matches!(
                    op,
                    UnaryOp::Abs |
                        UnaryOp::Ceil |
                        UnaryOp::Floor |
                        UnaryOp::Exp |
                        UnaryOp::Sqrt |
                        UnaryOp::Log1p |
                        UnaryOp::Sin |
                        UnaryOp::Cos |
                        UnaryOp::Sign
                ) && !matches!(self.expr_dtype(&inner)?, DataType::Float32 | DataType::Float64)
                {
                    return Err(unsupported("float-only math op on non-float column"));
                }
                Ok(Expr::Unary(op, Box::new(inner)))
            }
            AExpr::Cast { expr, dtype, .. } => {
                // Only allow identity-style casts (no-op) inside
                // expressions; real casts should fall back to CPU.
                let inner = self.lower_expr(*expr)?;
                let inner_dtype = self.expr_dtype(&inner)?;
                if &inner_dtype == dtype {
                    Ok(inner)
                } else {
                    Err(unsupported("non-identity cast in expression"))
                }
            }
            _ => Err(unsupported("expression shape outside the subset")),
        }
    }

    /// Translate a Polars [`LiteralValue`] into a [`ScalarLit`].
    fn lower_literal_to_scalar(&self, value: &LiteralValue) -> Result<ScalarLit, LowerError> {
        match value {
            LiteralValue::Scalar(scalar) => match scalar.value() {
                AnyValue::Int32(raw) => Ok(ScalarLit::I32(*raw)),
                AnyValue::Int64(raw) => Ok(ScalarLit::I64(*raw)),
                AnyValue::UInt32(raw) => Ok(ScalarLit::U32(*raw)),
                AnyValue::UInt64(raw) => Ok(ScalarLit::U64(*raw)),
                AnyValue::Float32(raw) => Ok(ScalarLit::F32(*raw)),
                AnyValue::Float64(raw) => Ok(ScalarLit::F64(*raw)),
                AnyValue::Boolean(raw) => Ok(ScalarLit::Bool(*raw)),
                _ => Err(unsupported("literal kind outside the subset")),
            },
            LiteralValue::Dyn(DynLiteralValue::Int(raw)) => i32::try_from(*raw)
                .map(ScalarLit::I32)
                .map_err(|_| unsupported("integer literal out of i32 range")),
            LiteralValue::Dyn(DynLiteralValue::Float(raw)) => Ok(ScalarLit::F64(*raw)),
            _ => Err(unsupported("literal kind outside the subset")),
        }
    }

    fn apply_group_by(
        &mut self,
        keys: Vec<(PlSmallStr, PlSmallStr)>,
        aggs: Vec<Aggregation>,
    ) -> Result<(), LowerError> {
        let mut schema = Vec::with_capacity(keys.len() + aggs.len());
        for (output, input) in &keys {
            schema.push((output.clone(), self.require_column(input)?));
        }
        for agg in &aggs {
            schema.push((agg.output().clone(), agg.dtype(self)?));
        }
        self.schema = schema;
        self.ops.push(PlanOp::GroupBy { keys, aggs });
        Ok(())
    }

    fn lower_aggregation(&self, agg: &ExprIR) -> Result<Aggregation, LowerError> {
        let output = output_name(agg)?;
        match self.expr_arena.get(agg.node()) {
            AExpr::Agg(inner) => match inner {
                IRAggExpr::Sum(input) => {
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Sum { input, output })
                }
                IRAggExpr::Min { input, propagate_nans } => {
                    reject_nans(*propagate_nans)?;
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Min { input, output })
                }
                IRAggExpr::Max { input, propagate_nans } => {
                    reject_nans(*propagate_nans)?;
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Max { input, output })
                }
                IRAggExpr::Mean(input) => {
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Mean { input, output })
                }
                IRAggExpr::Count { input, include_nulls } => {
                    let input = self.plain_column(*input)?;
                    if *include_nulls {
                        Ok(Aggregation::Len { output })
                    } else {
                        Ok(Aggregation::Count { input, output })
                    }
                }
                IRAggExpr::First(input) => {
                    let input = self.supported_column(*input)?;
                    Ok(Aggregation::First { input, output })
                }
                IRAggExpr::Last(input) => {
                    let input = self.supported_column(*input)?;
                    Ok(Aggregation::Last { input, output })
                }
                IRAggExpr::Std(input, ddof) => {
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Std { input, output, ddof: *ddof })
                }
                IRAggExpr::Var(input, ddof) => {
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Var { input, output, ddof: *ddof })
                }
                IRAggExpr::Median(input) => {
                    let input = self.numeric_column(*input)?;
                    Ok(Aggregation::Median { input, output })
                }
                _ => Err(unsupported("aggregation outside the subset")),
            },
            AExpr::Function { input, function, .. } => match function {
                IRFunctionExpr::Product => {
                    if input.len() != 1 {
                        return Err(unsupported("product with multiple inputs"));
                    }
                    let input_col = self.numeric_column(input[0].node())?;
                    Ok(Aggregation::Product { input: input_col, output })
                }
                _ => Err(unsupported("function outside the subset")),
            },
            AExpr::Len => Ok(Aggregation::Len { output }),
            _ => Err(unsupported("non-aggregation expression in group_by")),
        }
    }

    fn numeric_column(&self, node: Node) -> Result<PlSmallStr, LowerError> {
        let name = self.plain_column(node)?;
        let dtype = self.require_column(&name)?;
        if DType::from_polars(&dtype).is_none() {
            return Err(unsupported("aggregation over unsupported dtype"));
        }
        Ok(name)
    }

    fn supported_column(&self, node: Node) -> Result<PlSmallStr, LowerError> {
        let name = self.plain_column(node)?;
        let dtype = self.require_column(&name)?;
        require_dtype(&dtype)?;
        Ok(name)
    }

    /// Lower a predicate into negation normal form.
    ///
    /// `negated` tracks an odd number of enclosing `NOT`s (De Morgan): leaf
    /// comparisons flip operators, `AND`/`OR` swap, and bare boolean columns
    /// become [`Predicate::NotColumn`].
    fn lower_predicate(&self, node: Node, negated: bool) -> Result<Predicate, LowerError> {
        match self.expr_arena.get(node) {
            AExpr::BinaryExpr { left, op, right } => match op {
                Operator::Eq |
                Operator::NotEq |
                Operator::Lt |
                Operator::LtEq |
                Operator::Gt |
                Operator::GtEq => self.lower_comparison(*left, *op, *right, negated),
                Operator::And | Operator::LogicalAnd if !negated => Ok(Predicate::And(
                    Box::new(self.lower_predicate(*left, false)?),
                    Box::new(self.lower_predicate(*right, false)?),
                )),
                Operator::Or | Operator::LogicalOr if !negated => Ok(Predicate::Or(
                    Box::new(self.lower_predicate(*left, false)?),
                    Box::new(self.lower_predicate(*right, false)?),
                )),
                Operator::And | Operator::LogicalAnd => Ok(Predicate::Or(
                    Box::new(self.lower_predicate(*left, true)?),
                    Box::new(self.lower_predicate(*right, true)?),
                )),
                Operator::Or | Operator::LogicalOr => Ok(Predicate::And(
                    Box::new(self.lower_predicate(*left, true)?),
                    Box::new(self.lower_predicate(*right, true)?),
                )),
                _ => Err(unsupported("binary operator outside the subset")),
            },
            AExpr::Function { input, function, .. } => match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::Not) => {
                    if input.len() != 1 {
                        return Err(unsupported("not with multiple inputs"));
                    }
                    self.lower_predicate(input[0].node(), !negated)
                }
                _ => Err(unsupported("function outside the subset")),
            },
            AExpr::Column(name) => {
                let dtype = self.require_column(name)?;
                if dtype != DataType::Boolean {
                    return Err(unsupported("non-boolean column as predicate"));
                }
                if negated {
                    Ok(Predicate::NotColumn { column: name.clone() })
                } else {
                    Ok(Predicate::ColumnMask { column: name.clone() })
                }
            }
            _ => Err(unsupported("predicate shape outside the subset")),
        }
    }

    fn lower_comparison(
        &self,
        left: Node,
        op: Operator,
        right: Node,
        negated: bool,
    ) -> Result<Predicate, LowerError> {
        let cmp = match op {
            Operator::Eq => CmpOp::Eq,
            Operator::NotEq => CmpOp::NotEq,
            Operator::Lt => CmpOp::Lt,
            Operator::LtEq => CmpOp::LtEq,
            Operator::Gt => CmpOp::Gt,
            Operator::GtEq => CmpOp::GtEq,
            _ => return Err(unsupported("comparison operator outside the subset")),
        };
        let cmp = if negated { cmp.negate() } else { cmp };
        if let Ok(predicate) = self.column_against_literal(left, cmp, right) {
            return Ok(predicate);
        }
        self.column_against_literal(right, cmp.flip(), left)
    }

    /// Lower `column_node op literal_node`.
    ///
    /// The column side is either a plain column or an `f64` cast over an
    /// `Int32`/`UInt32` column (the widening the CPU engine applies when one
    /// side is a float literal).
    fn column_against_literal(
        &self,
        column_node: Node,
        op: CmpOp,
        literal_node: Node,
    ) -> Result<Predicate, LowerError> {
        let (column, widened) = match self.expr_arena.get(column_node) {
            AExpr::Column(name) => (name.clone(), None),
            AExpr::Cast { expr, dtype, .. } => {
                let AExpr::Column(name) = self.expr_arena.get(*expr) else {
                    return Err(unsupported("cast of a non-column in comparison"));
                };
                if *dtype != DataType::Float64 {
                    return Err(unsupported("cast to non-f64 in comparison"));
                }
                match self.require_column(name)? {
                    DataType::Int32 | DataType::UInt32 => (name.clone(), Some(DType::F64)),
                    _ => return Err(unsupported("widening cast of non-32-bit column")),
                }
            }
            _ => return Err(unsupported("comparison without a column side")),
        };
        let column_dtype = self.require_column(&column)?;
        let kernel_dtype = widened.unwrap_or(
            DType::from_polars(&column_dtype)
                .ok_or_else(|| unsupported("comparison over unsupported dtype"))?,
        );
        let literal = self.lower_literal(literal_node, &column_dtype, kernel_dtype)?;
        Ok(Predicate::Compare { column, kernel_dtype, op, literal })
    }

    fn lower_literal(
        &self,
        node: Node,
        column_dtype: &DataType,
        kernel_dtype: DType,
    ) -> Result<ScalarLit, LowerError> {
        let AExpr::Literal(value) = self.expr_arena.get(node) else {
            return Err(unsupported("comparison without a literal side"));
        };
        match value {
            LiteralValue::Scalar(scalar) => {
                scalar_against_dtype(scalar.value(), column_dtype, kernel_dtype)
            }
            LiteralValue::Dyn(DynLiteralValue::Int(raw)) => {
                int_against_dtype(*raw, column_dtype, kernel_dtype)
            }
            LiteralValue::Dyn(DynLiteralValue::Float(raw)) => {
                float_against_dtype(*raw, column_dtype, kernel_dtype)
            }
            _ => Err(unsupported("literal kind outside the subset")),
        }
    }
}

impl Aggregation {
    const fn output(&self) -> &PlSmallStr {
        match self {
            Self::Sum { output, .. } |
            Self::Min { output, .. } |
            Self::Max { output, .. } |
            Self::Mean { output, .. } |
            Self::Count { output, .. } |
            Self::First { output, .. } |
            Self::Last { output, .. } |
            Self::Product { output, .. } |
            Self::Std { output, .. } |
            Self::Var { output, .. } |
            Self::Median { output, .. } |
            Self::Len { output } => output,
        }
    }

    fn dtype(&self, builder: &Builder<'_>) -> Result<DataType, LowerError> {
        match self {
            // Sum/min/max/first/last/product keep the input dtype (verified
            // against the CPU oracle).
            Self::Sum { input, .. } |
            Self::Min { input, .. } |
            Self::Max { input, .. } |
            Self::First { input, .. } |
            Self::Last { input, .. } |
            Self::Product { input, .. } |
            Self::Median { input, .. } => Ok(builder.require_column(input)?),
            // Mean/std/var widen to `f64`.
            Self::Mean { .. } | Self::Std { .. } | Self::Var { .. } => Ok(DataType::Float64),
            Self::Count { .. } | Self::Len { .. } => Ok(DataType::UInt32),
        }
    }
}

fn output_name(expr: &ExprIR) -> Result<PlSmallStr, LowerError> {
    expr.output_name_inner()
        .get()
        .cloned()
        .ok_or_else(|| unsupported("expression without an output name"))
}

fn require_dtype(dtype: &DataType) -> Result<(), LowerError> {
    if DType::from_polars(dtype).is_none() {
        return Err(unsupported(&format!("dtype {dtype:?} outside the subset")));
    }
    Ok(())
}

fn reject_nans(propagate_nans: bool) -> Result<(), LowerError> {
    if propagate_nans {
        return Err(unsupported("min/max with propagate_nans"));
    }
    Ok(())
}

/// Validate a materialized scalar literal against the comparison.
///
/// `column_dtype` is the polars dtype of the column; `kernel_dtype` is the
/// dtype the comparison kernel runs in (equal except for the f64 widening).
fn scalar_against_dtype(
    value: &AnyValue<'_>,
    column_dtype: &DataType,
    kernel_dtype: DType,
) -> Result<ScalarLit, LowerError> {
    if kernel_dtype == DType::F64 && matches!(column_dtype, DataType::Int32 | DataType::UInt32) {
        if let Some(raw) = any_value_as_f64(value) {
            return Ok(ScalarLit::F64(raw));
        }
        return Err(unsupported("literal dtype incompatible with column dtype"));
    }
    match (value, column_dtype) {
        (AnyValue::Int32(raw), DataType::Int32) => Ok(ScalarLit::I32(*raw)),
        (AnyValue::Int64(raw), DataType::Int64) => Ok(ScalarLit::I64(*raw)),
        (AnyValue::UInt32(raw), DataType::UInt32) => Ok(ScalarLit::U32(*raw)),
        (AnyValue::UInt64(raw), DataType::UInt64) => Ok(ScalarLit::U64(*raw)),
        (AnyValue::Float32(raw), DataType::Float32) => Ok(ScalarLit::F32(*raw)),
        (AnyValue::Float64(raw), DataType::Float64) => Ok(ScalarLit::F64(*raw)),
        (AnyValue::Boolean(raw), DataType::Boolean) => Ok(ScalarLit::Bool(*raw)),
        (AnyValue::Int64(raw), DataType::Int32) => {
            int_against_dtype(i128::from(*raw), column_dtype, kernel_dtype)
        }
        (AnyValue::Int32(raw), DataType::Int64) => Ok(ScalarLit::I64(i64::from(*raw))),
        _ => Err(unsupported("literal dtype incompatible with column dtype")),
    }
}

fn any_value_as_f64(value: &AnyValue<'_>) -> Option<f64> {
    match value {
        AnyValue::Int32(raw) => Some(f64::from(*raw)),
        AnyValue::Int64(raw) => Some(*raw as f64),
        AnyValue::UInt32(raw) => Some(f64::from(*raw)),
        AnyValue::UInt64(raw) => Some(*raw as f64),
        AnyValue::Float32(raw) => Some(f64::from(*raw)),
        AnyValue::Float64(raw) => Some(*raw),
        _ => None,
    }
}

/// Polars-style type promotion for a binary arithmetic/algebraic operation.
///
/// For the engine's subset the rules reduce to: `Float64` absorbs everything,
/// `Float32` absorbs all integers, wider integers absorb narrower ones of the
/// same sign class, and mixed signedness widens to the next signed width.
fn promote_binary(left: DataType, right: DataType) -> Result<DataType, LowerError> {
    use DataType::*;
    if left == right {
        return Ok(left);
    }
    // Float64 absorbs all.
    if left == Float64 || right == Float64 {
        return Ok(Float64);
    }
    // Float32 absorbs integers.
    if left == Float32 || right == Float32 {
        return Ok(Float32);
    }
    // Both integers: widen to the wider of the two, preferring signed.
    let (lbits, l_signed) = int_width_signed(&left)?;
    let (rbits, r_signed) = int_width_signed(&right)?;
    if l_signed && r_signed {
        return Ok(if lbits >= rbits { left } else { right });
    }
    if !l_signed && !r_signed {
        return Ok(if lbits >= rbits { left } else { right });
    }
    // Mixed signedness: promote to the wider, preferring signed if equal.
    let max_bits = lbits.max(rbits);
    let promoted = if max_bits >= 64 {
        Int64
    } else if max_bits >= 32 {
        if lbits == 32 && rbits == 32 {
            // 32-bit signed + 32-bit unsigned -> Int64.
            Int64
        } else {
            Int64
        }
    } else {
        Int32
    };
    Ok(promoted)
}

fn int_width_signed(dtype: &DataType) -> Result<(u32, bool), LowerError> {
    match dtype {
        DataType::Int32 => Ok((32, true)),
        DataType::Int64 => Ok((64, true)),
        DataType::UInt32 => Ok((32, false)),
        DataType::UInt64 => Ok((64, false)),
        _ => Err(unsupported("non-integer operand in binary op")),
    }
}

fn int_against_dtype(
    raw: i128,
    column_dtype: &DataType,
    kernel_dtype: DType,
) -> Result<ScalarLit, LowerError> {
    match (column_dtype, kernel_dtype) {
        (DataType::Int32, DType::I32) => Ok(ScalarLit::I32(
            i32::try_from(raw).map_err(|_| unsupported("integer literal out of range"))?,
        )),
        (DataType::Int64, DType::I64) => Ok(ScalarLit::I64(
            i64::try_from(raw).map_err(|_| unsupported("integer literal out of range"))?,
        )),
        (DataType::UInt32, DType::U32) => Ok(ScalarLit::U32(
            u32::try_from(raw).map_err(|_| unsupported("integer literal out of range"))?,
        )),
        (DataType::UInt64, DType::U64) => Ok(ScalarLit::U64(
            u64::try_from(raw).map_err(|_| unsupported("integer literal out of range"))?,
        )),
        (DataType::Float32, DType::F32) => {
            let value = raw as f32;
            if f32_is_exact_int(value, raw) {
                Ok(ScalarLit::F32(value))
            } else {
                Err(unsupported("integer literal not exact in f32"))
            }
        }
        (DataType::Float64, DType::F64) => {
            let value = raw as f64;
            if f64_is_exact_int(value, raw) {
                Ok(ScalarLit::F64(value))
            } else {
                Err(unsupported("integer literal not exact in f64"))
            }
        }
        _ => Err(unsupported("integer literal against incompatible column")),
    }
}

fn float_against_dtype(
    raw: f64,
    column_dtype: &DataType,
    kernel_dtype: DType,
) -> Result<ScalarLit, LowerError> {
    match (column_dtype, kernel_dtype) {
        // i32/u32 convert to f64 exactly, so passing the literal through
        // matches Polars; an f64 column passes through unchanged.
        (DataType::Int32 | DataType::UInt32 | DataType::Float64, DType::F64) => {
            Ok(ScalarLit::F64(raw))
        }
        (DataType::Int64 | DataType::UInt64, _) => {
            Err(unsupported("float literal against 64-bit integer column"))
        }
        (DataType::Float32, DType::F32) => {
            let value = raw as f32;
            // Exact equality, not a tolerance: the question is whether the
            // literal survives the f64 -> f32 narrowing bit for bit. A margin
            // would accept literals Polars itself rejects.
            #[expect(clippy::float_cmp, reason = "exact round-trip check, not an approximate one")]
            let exact = f64::from(value) == raw;
            if exact || (value.is_nan() && raw.is_nan()) {
                Ok(ScalarLit::F32(value))
            } else {
                Err(unsupported("float literal not exact in f32"))
            }
        }
        _ => Err(unsupported("float literal against incompatible column")),
    }
}

const fn f32_is_exact_int(value: f32, raw: i128) -> bool {
    value as i128 == raw && raw >= -(1 << 24) && raw <= (1 << 24)
}

const fn f64_is_exact_int(value: f64, raw: i128) -> bool {
    value as i128 == raw && raw >= -9_007_199_254_740_992 && raw <= 9_007_199_254_740_992
}

fn unsupported(reason: &str) -> LowerError {
    LowerError::Unsupported { reason: reason.to_string() }
}

/// Property-based tests for invariants that must hold across randomized
/// inputs — operator algebra, float-roundtrip boundaries, and the tight
/// coupling between Polars dtypes and the engine subset.
#[cfg(test)]
mod property_tests {
    use polars_core::datatypes::TimeUnit;

    use super::*;

    /// `f32_is_exact_int` is true exactly for integers whose `f32` bit
    /// pattern round-trips and that sit within the ±2^24 inclusive range.
    #[test]
    fn f32_exact_int_boundary_is_exactly_plus_minus_2_24() {
        // Inside the boundary: exact representation.
        assert!(f32_is_exact_int(0.0, 0));
        assert!(f32_is_exact_int(1.0, 1));
        assert!(f32_is_exact_int(-1.0, -1));
        assert!(f32_is_exact_int(16_777_216.0, 16_777_216));
        assert!(f32_is_exact_int(-16_777_216.0, -16_777_216));
        // Outside: rejected even when bit-roundtrip coincidentally holds.
        assert!(!f32_is_exact_int(16_777_217.0, 16_777_217));
        assert!(!f32_is_exact_int(-16_777_217.0, -16_777_217));
        assert!(!f32_is_exact_int(16_777_218.0, 16_777_218));
    }

    /// `f64_is_exact_int` is true exactly for integers whose `f64` bit
    /// pattern round-trips and that sit within the ±2^53 inclusive range.
    #[test]
    fn f64_exact_int_boundary_is_exactly_plus_minus_2_53() {
        assert!(f64_is_exact_int(0.0, 0));
        assert!(f64_is_exact_int(9_007_199_254_740_992.0, 9_007_199_254_740_992));
        assert!(f64_is_exact_int(-9_007_199_254_740_992.0, -9_007_199_254_740_992));
        // 2^53 + 1 cannot be represented exactly.
        assert!(!f64_is_exact_int(9_007_199_254_740_993.0, 9_007_199_254_740_993));
        // 2^53 + 2 is representable but out of range.
        assert!(!f64_is_exact_int(9_007_199_254_740_994.0, 9_007_199_254_740_994));
    }

    /// `DType::from_polars` → `to_polars` round-trips for every supported
    /// dtype; unsupported ones yield `None`.
    #[test]
    #[expect(clippy::expect_used, reason = "test asserts supported-dtype invariant")]
    fn dtype_round_trip_supported_and_rejects_unsupported() {
        use polars_core::datatypes::DataType::*;
        for dtype in [Int32, Int64, UInt32, UInt64, Float32, Float64, Boolean] {
            let engine = DType::from_polars(&dtype).expect("supported dtype should always convert");
            assert_eq!(engine.to_polars(), dtype, "round-trip failed for {dtype:?}");
        }
        for dtype in [String, Null, Date, Datetime(TimeUnit::Microseconds, None)] {
            assert!(
                DType::from_polars(&dtype).is_none(),
                "unsupported dtype {dtype:?} should return None"
            );
        }
    }

    /// `Aggregation::dtype` always mirrors what the CPU oracle produces:
    /// aggregates that keep the input keep it; `Mean` widens to `f64`;
    /// `Count`/`Len` yield `UInt32`.
    #[test]
    #[expect(clippy::expect_used, reason = "test asserts valid aggregation schema")]
    fn aggregation_dtype_matches_cpu_semantics() {
        let arena = Arena::new();
        let mut builder = Builder::new(&arena);
        builder.schema = vec![
            (name("i32"), DataType::Int32),
            (name("i64"), DataType::Int64),
            (name("f32"), DataType::Float32),
            (name("f64"), DataType::Float64),
            (name("u32"), DataType::UInt32),
        ];
        let cases = [
            (Aggregation::Sum { input: name("i32"), output: name("s") }, DataType::Int32),
            (Aggregation::Min { input: name("f32"), output: name("m") }, DataType::Float32),
            (Aggregation::Max { input: name("f64"), output: name("m") }, DataType::Float64),
            (Aggregation::First { input: name("i64"), output: name("f") }, DataType::Int64),
            (Aggregation::Last { input: name("u32"), output: name("l") }, DataType::UInt32),
            (Aggregation::Mean { input: name("i32"), output: name("m") }, DataType::Float64),
            (Aggregation::Count { input: name("i64"), output: name("c") }, DataType::UInt32),
            (Aggregation::Len { output: name("n") }, DataType::UInt32),
        ];
        for (agg, expected) in cases {
            let dtype = agg.dtype(&builder).expect("dtype lookup");
            assert_eq!(dtype, expected, "Aggregation::{:?} should yield {expected:?}", agg);
        }
    }

    fn name(value: &str) -> PlSmallStr {
        PlSmallStr::from(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int32(value: i32) -> AnyValue<'static> {
        AnyValue::Int32(value)
    }

    #[test]
    fn int_literal_exact_in_f32_is_accepted() {
        assert_eq!(int_against_dtype(5, &DataType::Float32, DType::F32), Ok(ScalarLit::F32(5.0)));
        assert_eq!(
            int_against_dtype(-16_777_216, &DataType::Float32, DType::F32),
            Ok(ScalarLit::F32(-16_777_216.0))
        );
    }

    #[test]
    fn int_literal_inexact_in_f32_is_rejected() {
        // 2^24 + 1 has no exact f32 representation.
        assert!(int_against_dtype(16_777_217, &DataType::Float32, DType::F32).is_err());
        // 2^24 + 2 is exact but out of range: equality holds, the bound fails.
        assert!(int_against_dtype(16_777_218, &DataType::Float32, DType::F32).is_err());
        // Below the range with a failed equality: both conjuncts fail.
        assert!(int_against_dtype(-16_777_217, &DataType::Float32, DType::F32).is_err());
        assert!(int_against_dtype(i128::from(i64::MAX), &DataType::Float32, DType::F32).is_err());
    }

    #[test]
    fn int_literal_exact_in_f64_is_accepted() {
        assert_eq!(int_against_dtype(5, &DataType::Float64, DType::F64), Ok(ScalarLit::F64(5.0)));
        assert_eq!(
            int_against_dtype(9_007_199_254_740_992, &DataType::Float64, DType::F64),
            Ok(ScalarLit::F64(9_007_199_254_740_992.0))
        );
    }

    #[test]
    fn int_literal_inexact_in_f64_is_rejected() {
        // 2^53 + 1 has no exact f64 representation.
        assert!(int_against_dtype(9_007_199_254_740_993, &DataType::Float64, DType::F64).is_err());
        // 2^53 + 2 is exact but out of range.
        assert!(int_against_dtype(9_007_199_254_740_994, &DataType::Float64, DType::F64).is_err());
        // -(2^53 + 1) rounds up: equality fails while both bounds hold.
        assert!(int_against_dtype(-9_007_199_254_740_993, &DataType::Float64, DType::F64).is_err());
    }

    #[test]
    fn int_literal_range_checked_against_int_columns() {
        assert_eq!(int_against_dtype(7, &DataType::Int32, DType::I32), Ok(ScalarLit::I32(7)));
        assert_eq!(int_against_dtype(7, &DataType::Int64, DType::I64), Ok(ScalarLit::I64(7)));
        assert_eq!(int_against_dtype(7, &DataType::UInt32, DType::U32), Ok(ScalarLit::U32(7)));
        assert_eq!(int_against_dtype(7, &DataType::UInt64, DType::U64), Ok(ScalarLit::U64(7)));
        assert!(int_against_dtype(i128::from(i32::MAX) + 1, &DataType::Int32, DType::I32).is_err());
        assert!(int_against_dtype(-1, &DataType::UInt32, DType::U32).is_err());
        assert!(int_against_dtype(3, &DataType::Boolean, DType::Bool).is_err());
    }

    #[test]
    fn float_literal_exact_in_f32_is_accepted() {
        assert_eq!(
            float_against_dtype(0.5, &DataType::Float32, DType::F32),
            Ok(ScalarLit::F32(0.5))
        );
    }

    #[test]
    fn float_literal_inexact_in_f32_is_rejected() {
        // 0.1 narrows lossily; accepting it would compare a different value.
        assert!(float_against_dtype(0.1, &DataType::Float32, DType::F32).is_err());
    }

    #[test]
    fn float_nan_survives_f32_narrowing() {
        let result = float_against_dtype(f64::NAN, &DataType::Float32, DType::F32);
        assert!(matches!(result, Ok(ScalarLit::F32(value)) if value.is_nan()));
    }

    #[test]
    fn float_literal_against_64_bit_int_column_is_rejected() {
        assert!(float_against_dtype(1.0, &DataType::Int64, DType::I64).is_err());
        assert!(float_against_dtype(1.0, &DataType::UInt64, DType::U64).is_err());
        assert!(float_against_dtype(1.0, &DataType::Int64, DType::F64).is_err());
        assert!(float_against_dtype(1.0, &DataType::UInt64, DType::F64).is_err());
    }

    #[test]
    fn float_literal_passes_through_f64() {
        assert_eq!(
            float_against_dtype(0.1, &DataType::Float64, DType::F64),
            Ok(ScalarLit::F64(0.1))
        );
        assert_eq!(float_against_dtype(3.0, &DataType::Int32, DType::F64), Ok(ScalarLit::F64(3.0)));
        assert_eq!(
            float_against_dtype(3.0, &DataType::UInt32, DType::F64),
            Ok(ScalarLit::F64(3.0))
        );
    }

    #[test]
    fn scalar_literals_pass_through_matching_columns() {
        assert_eq!(
            scalar_against_dtype(&int32(4), &DataType::Int32, DType::I32),
            Ok(ScalarLit::I32(4))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Int64(4), &DataType::Int64, DType::I64),
            Ok(ScalarLit::I64(4))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::UInt32(4), &DataType::UInt32, DType::U32),
            Ok(ScalarLit::U32(4))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::UInt64(4), &DataType::UInt64, DType::U64),
            Ok(ScalarLit::U64(4))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Float32(0.5), &DataType::Float32, DType::F32),
            Ok(ScalarLit::F32(0.5))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Boolean(true), &DataType::Boolean, DType::Bool),
            Ok(ScalarLit::Bool(true))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Float64(1.5), &DataType::Float64, DType::F64),
            Ok(ScalarLit::F64(1.5))
        );
    }

    #[test]
    fn scalar_int64_narrows_into_int32_column_in_range() {
        assert_eq!(
            scalar_against_dtype(&AnyValue::Int64(9), &DataType::Int32, DType::I32),
            Ok(ScalarLit::I32(9))
        );
        assert!(
            scalar_against_dtype(
                &AnyValue::Int64(i64::from(i32::MAX) + 1),
                &DataType::Int32,
                DType::I32
            )
            .is_err()
        );
    }

    #[test]
    fn scalar_int32_widens_into_int64_column() {
        assert_eq!(
            scalar_against_dtype(&int32(-2), &DataType::Int64, DType::I64),
            Ok(ScalarLit::I64(-2))
        );
    }

    #[test]
    fn scalar_mismatched_literal_is_rejected() {
        assert!(scalar_against_dtype(&int32(1), &DataType::Boolean, DType::Bool).is_err());
        assert!(
            scalar_against_dtype(&AnyValue::String("x"), &DataType::String, DType::I32).is_err()
        );
    }

    #[test]
    fn scalar_widens_int_to_f64_kernel() {
        assert_eq!(
            scalar_against_dtype(&int32(8), &DataType::Int32, DType::F64),
            Ok(ScalarLit::F64(8.0))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Int64(8), &DataType::Int32, DType::F64),
            Ok(ScalarLit::F64(8.0))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::UInt32(3), &DataType::UInt32, DType::F64),
            Ok(ScalarLit::F64(3.0))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::UInt64(3), &DataType::UInt32, DType::F64),
            Ok(ScalarLit::F64(3.0))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Float32(0.5), &DataType::Int32, DType::F64),
            Ok(ScalarLit::F64(0.5))
        );
        assert_eq!(
            scalar_against_dtype(&AnyValue::Float64(0.5), &DataType::UInt32, DType::F64),
            Ok(ScalarLit::F64(0.5))
        );
    }

    #[test]
    fn support_predicates_reject_outside_subset() {
        assert!(require_dtype(&DataType::String).is_err());
        assert!(require_dtype(&DataType::Int32).is_ok());
        assert!(reject_nans(true).is_err());
        assert!(reject_nans(false).is_ok());
    }

    /// `flip` is an involution: flipping twice returns the original operator.
    /// This matters because `lower_comparison` tries `flip` as the second
    /// orientation — a non-involutional `flip` would silently swap operands.
    #[test]
    fn cmp_op_flip_is_involution() {
        for op in [CmpOp::Eq, CmpOp::NotEq, CmpOp::Lt, CmpOp::LtEq, CmpOp::Gt, CmpOp::GtEq] {
            assert_eq!(op.flip().flip(), op, "flip should be an involution for {op:?}");
        }
    }

    /// `negate` is an involution: negating twice returns the original operator.
    /// Predicate lowering relies on this when pushing `NOT` through leaves.
    #[test]
    fn cmp_op_negate_is_involution() {
        for op in [CmpOp::Eq, CmpOp::NotEq, CmpOp::Lt, CmpOp::LtEq, CmpOp::Gt, CmpOp::GtEq] {
            assert_eq!(op.negate().negate(), op, "negate should be an involution for {op:?}");
        }
    }

    /// `flip` and `negate` commute on symmetric operators (`Eq`/`NotEq`) and
    /// correctly restate asymmetric ones.
    #[test]
    fn cmp_op_flip_matches_negate_duality() {
        use CmpOp::*;
        // `Eq` and `NotEq` are symmetric: flip is identity.
        assert_eq!(Eq.flip(), Eq);
        assert_eq!(NotEq.flip(), NotEq);
        // Asymmetric operators flip to their mirror.
        assert_eq!(Lt.flip(), Gt);
        assert_eq!(LtEq.flip(), GtEq);
        assert_eq!(Gt.flip(), Lt);
        assert_eq!(GtEq.flip(), LtEq);
        // Negation maps to the complement operator.
        assert_eq!(Eq.negate(), NotEq);
        assert_eq!(NotEq.negate(), Eq);
        assert_eq!(Lt.negate(), GtEq);
        assert_eq!(LtEq.negate(), Gt);
        assert_eq!(Gt.negate(), LtEq);
        assert_eq!(GtEq.negate(), Lt);
    }

    #[test]
    fn aggregation_output_names_its_alias() {
        let agg = Aggregation::Sum {
            input: PlSmallStr::from_static("v"),
            output: PlSmallStr::from_static("s"),
        };
        assert_eq!(agg.output(), &PlSmallStr::from_static("s"));
        let agg = Aggregation::Len { output: PlSmallStr::from_static("n") };
        assert_eq!(agg.output(), &PlSmallStr::from_static("n"));
    }
}

#[cfg(test)]
mod predicate_tests {
    use polars_core::{
        prelude::{NamedFrom, Series},
        scalar::Scalar,
    };
    use polars_plan::plans::expr_ir::ExprIR;

    use super::*;

    pub(crate) fn name(value: &str) -> PlSmallStr {
        PlSmallStr::from(value)
    }

    #[test]
    fn all_comparison_operators_lower() {
        let cases = [
            (Operator::Eq, CmpOp::Eq),
            (Operator::NotEq, CmpOp::NotEq),
            (Operator::Lt, CmpOp::Lt),
            (Operator::LtEq, CmpOp::LtEq),
            (Operator::Gt, CmpOp::Gt),
            (Operator::GtEq, CmpOp::GtEq),
        ];
        for (op, expected) in cases {
            let mut arena = Arena::new();
            let left = arena.add(AExpr::Column(name("a")));
            let right = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(4))));
            let node = arena.add(AExpr::BinaryExpr { left, op, right });
            let builder = pred_builder(&arena);
            assert_eq!(
                builder.lower_predicate(node, false),
                Ok(Predicate::Compare {
                    column: name("a"),
                    kernel_dtype: DType::I32,
                    op: expected,
                    literal: ScalarLit::I32(4),
                })
            );
        }
    }

    #[test]
    fn and_or_lower_and_swap_under_negation() -> Result<(), LowerError> {
        // `a == 1 AND flag`, then the same tree negated (De Morgan).
        for negated in [false, true] {
            let mut arena = Arena::new();
            let column = arena.add(AExpr::Column(name("a")));
            let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(1))));
            let cmp =
                arena.add(AExpr::BinaryExpr { left: column, op: Operator::Eq, right: literal });
            let mask = arena.add(AExpr::Column(name("flag")));
            for op in [Operator::And, Operator::LogicalAnd, Operator::Or, Operator::LogicalOr] {
                let node = arena.add(AExpr::BinaryExpr { left: cmp, op, right: mask });
                let builder = pred_builder(&arena);
                let predicate = builder.lower_predicate(node, negated)?;
                // `AND` stays a conjunction unless negated; `OR` is the dual.
                let is_conjunction = matches!(op, Operator::And | Operator::LogicalAnd) != negated;
                if is_conjunction {
                    assert!(matches!(predicate, Predicate::And(..)));
                } else {
                    assert!(matches!(predicate, Predicate::Or(..)));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn not_flips_negation() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("flag")));
        let input = ExprIR::from_node(column, &arena);
        let node = arena.add(AExpr::Function {
            input: vec![input],
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            options: Default::default(),
        });
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::NotColumn { column: name("flag") })
        );
        assert_eq!(
            builder.lower_predicate(node, true),
            Ok(Predicate::ColumnMask { column: name("flag") })
        );
    }

    #[test]
    fn bare_boolean_column_is_a_mask() {
        let mut arena = Arena::new();
        let node = arena.add(AExpr::Column(name("flag")));
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::ColumnMask { column: name("flag") })
        );
    }

    #[test]
    fn non_boolean_column_as_predicate_is_rejected() {
        let mut arena = Arena::new();
        let node = arena.add(AExpr::Column(name("a")));
        let builder = pred_builder(&arena);
        assert!(builder.lower_predicate(node, false).is_err());
    }

    #[test]
    fn non_comparison_binary_is_rejected() {
        let mut arena = Arena::new();
        let left = arena.add(AExpr::Column(name("a")));
        let right = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(1))));
        let node = arena.add(AExpr::BinaryExpr { left, op: Operator::Plus, right });
        let builder = pred_builder(&arena);
        assert!(builder.lower_predicate(node, false).is_err());
    }

    #[test]
    fn widening_cast_comparison_runs_in_f64() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("a")));
        let cast = arena.add(AExpr::Cast {
            expr: column,
            dtype: DataType::Float64,
            options: polars_core::chunked_array::cast::CastOptions::Strict,
        });
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(1.5))));
        let node = arena.add(AExpr::BinaryExpr { left: cast, op: Operator::Gt, right: literal });
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::Compare {
                column: name("a"),
                kernel_dtype: DType::F64,
                op: CmpOp::Gt,
                literal: ScalarLit::F64(1.5),
            })
        );
    }

    #[test]
    fn float_literal_against_int64_column_is_rejected_in_predicate() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("big")));
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(1.0))));
        let node = arena.add(AExpr::BinaryExpr { left: column, op: Operator::Eq, right: literal });
        let builder = pred_builder(&arena);
        assert!(builder.lower_predicate(node, false).is_err());
    }

    #[test]
    fn scalar_literal_lowers_through() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("a")));
        let scalar = Scalar::new(DataType::Int32, AnyValue::Int32(4));
        let literal = arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)));
        let node = arena.add(AExpr::BinaryExpr { left: column, op: Operator::Lt, right: literal });
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::Compare {
                column: name("a"),
                kernel_dtype: DType::I32,
                op: CmpOp::Lt,
                literal: ScalarLit::I32(4),
            })
        );
    }

    #[test]
    fn supported_column_accepts_subset_and_rejects_the_rest() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("a")));
        let missing = arena.add(AExpr::Column(name("nope")));
        let stringy = arena.add(AExpr::Column(name("s")));
        let builder = pred_builder(&arena);
        assert_eq!(builder.supported_column(column), Ok(name("a")));
        assert!(builder.supported_column(missing).is_err());
        assert!(builder.supported_column(stringy).is_err());
    }

    #[test]
    fn output_name_reads_column_names() {
        let mut arena = Arena::new();
        let node = arena.add(AExpr::Column(name("a")));
        let expr = ExprIR::from_node(node, &arena);
        assert_eq!(output_name(&expr), Ok(name("a")));
    }

    #[test]
    fn column_against_literal_with_literal_on_left() {
        // `5 > a` must be lowered as `a < 5` via `flip()`.
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("a")));
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(5))));
        let node = arena.add(AExpr::BinaryExpr { left: literal, op: Operator::Gt, right: column });
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::Compare {
                column: name("a"),
                kernel_dtype: DType::I32,
                op: CmpOp::Lt,
                literal: ScalarLit::I32(5),
            })
        );
    }

    #[test]
    fn negated_comparison_flips_operator() {
        // `NOT (a == 1)` should produce `a != 1`.
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("a")));
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(1))));
        let cmp = arena.add(AExpr::BinaryExpr { left: column, op: Operator::Eq, right: literal });
        let input = ExprIR::from_node(cmp, &arena);
        let node = arena.add(AExpr::Function {
            input: vec![input],
            function: IRFunctionExpr::Boolean(IRBooleanFunction::Not),
            options: Default::default(),
        });
        let builder = pred_builder(&arena);
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::Compare {
                column: name("a"),
                kernel_dtype: DType::I32,
                op: CmpOp::NotEq,
                literal: ScalarLit::I32(1),
            })
        );
    }

    #[test]
    fn i64_literal_against_u64_column_lowers() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("u")));
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(3))));
        let node = arena.add(AExpr::BinaryExpr { left: column, op: Operator::Eq, right: literal });
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("u"), DataType::UInt64)];
        assert_eq!(
            builder.lower_predicate(node, false),
            Ok(Predicate::Compare {
                column: name("u"),
                kernel_dtype: DType::U64,
                op: CmpOp::Eq,
                literal: ScalarLit::U64(3),
            })
        );
    }

    #[test]
    fn f64_literal_against_f32_column_narrows_lossy_is_rejected() {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("f")));
        let literal = arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(0.1))));
        let node = arena.add(AExpr::BinaryExpr { left: column, op: Operator::Eq, right: literal });
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("f"), DataType::Float32)];
        assert!(builder.lower_predicate(node, false).is_err());
    }

    #[test]
    fn source_height_reports_frame_rows() -> Result<(), LowerError> {
        let frame =
            DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3]).into()])
                .map_err(|error| unsupported(&error.to_string()))?;
        let plan = CubePlan {
            source: PlanSource::DataFrame { frame: Arc::new(frame), columns: vec![name("a")] },
            ops: Vec::new(),
        };
        assert_eq!(plan.source_height(), 3);
        assert!(plan.ops().is_empty(), "fresh plan carries no ops");
        Ok(())
    }

    fn pred_builder(arena: &Arena<AExpr>) -> Builder<'_> {
        let mut builder = Builder::new(arena);
        builder.schema = vec![
            (name("a"), DataType::Int32),
            (name("f"), DataType::Float64),
            (name("flag"), DataType::Boolean),
            (name("big"), DataType::Int64),
            (name("s"), DataType::String),
        ];
        builder
    }
}

#[cfg(test)]
mod plan_tests {
    use polars_core::prelude::{NamedFrom, Series};
    use polars_plan::plans::expr_ir::OutputName;

    use super::*;
    use crate::predicate_tests::name;

    pub(crate) fn plan_frame() -> DataFrame {
        DataFrame::new_infer_height(vec![
            Series::new("a".into(), &[1_i32, 2, 3]).into(),
            Series::new("f".into(), &[Some(1.0_f64), None, Some(3.0)]).into(),
            Series::new("flag".into(), &[Some(true), Some(false), None]).into(),
        ])
        .unwrap_or(DataFrame::empty_with_height(0))
    }

    pub(crate) fn scan_node(frame: DataFrame, lp_arena: &mut Arena<IR>) -> Node {
        let schema = frame.schema().clone();
        lp_arena.add(IR::DataFrameScan { df: Arc::new(frame), schema, output_schema: None })
    }

    fn compare_node(
        expr_arena: &mut Arena<AExpr>,
        column: &str,
        op: Operator,
        literal: LiteralValue,
    ) -> Node {
        let left = expr_arena.add(AExpr::Column(name(column)));
        let right = expr_arena.add(AExpr::Literal(literal));
        expr_arena.add(AExpr::BinaryExpr { left, op, right })
    }

    #[test]
    fn filter_plan_lowers_end_to_end() -> Result<(), LowerError> {
        let frame = plan_frame();
        let mut expr_arena = Arena::new();
        let predicate = compare_node(
            &mut expr_arena,
            "a",
            Operator::Gt,
            LiteralValue::Dyn(DynLiteralValue::Int(1)),
        );
        let mut lp_arena = Arena::new();
        let scan = scan_node(frame, &mut lp_arena);
        let filter = lp_arena
            .add(IR::Filter { input: scan, predicate: ExprIR::from_node(predicate, &expr_arena) });
        let plan = lower(filter, &lp_arena, &expr_arena)?;
        assert_eq!(plan.source_height(), 3);
        assert_eq!(plan.ops(), &[PlanOp::Filter(Predicate::Compare {
            column: name("a"),
            kernel_dtype: DType::I32,
            op: CmpOp::Gt,
            literal: ScalarLit::I32(1),
        })]);
        Ok(())
    }

    #[test]
    fn child_nodes_reports_inputs() {
        let frame = plan_frame();
        let mut expr_arena = Arena::new();
        let predicate = compare_node(
            &mut expr_arena,
            "a",
            Operator::Eq,
            LiteralValue::Dyn(DynLiteralValue::Int(2)),
        );
        let mut lp_arena = Arena::new();
        let scan = scan_node(frame, &mut lp_arena);
        assert!(child_nodes(scan, &lp_arena).is_empty(), "scan nodes have no children");
        let filter = lp_arena
            .add(IR::Filter { input: scan, predicate: ExprIR::from_node(predicate, &expr_arena) });
        assert_eq!(child_nodes(filter, &lp_arena), vec![scan]);
    }

    #[test]
    fn memory_sink_passes_through() -> Result<(), LowerError> {
        let frame = plan_frame();
        let expr_arena = Arena::new();
        let mut lp_arena = Arena::new();
        let scan = scan_node(frame, &mut lp_arena);
        let sink = lp_arena.add(IR::Sink { input: scan, payload: SinkTypeIR::Memory });
        let plan = lower(sink, &lp_arena, &expr_arena)?;
        assert_eq!(plan.source_height(), 3);
        assert!(plan.ops().is_empty(), "sink adds no ops");
        Ok(())
    }

    #[test]
    fn apply_projection_rewrites_schema() -> Result<(), LowerError> {
        let arena = Arena::new();
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("a"), DataType::Int32), (name("f"), DataType::Float64)];
        builder.apply_projection(vec![
            (name("x"), Expr::Col(name("a"))),
            (name("y"), Expr::Col(name("f"))),
        ])?;
        assert_eq!(builder.schema, vec![
            (name("x"), DataType::Int32),
            (name("y"), DataType::Float64)
        ]);
        assert_eq!(builder.ops, [PlanOp::Project(vec![
            (name("x"), Expr::Col(name("a"))),
            (name("y"), Expr::Col(name("f"))),
        ])]);
        // Duplicate outputs and unknown inputs are rejected, not silently kept.
        assert!(
            builder
                .apply_projection(vec![
                    (name("x"), Expr::Col(name("a"))),
                    (name("x"), Expr::Col(name("f"))),
                ])
                .is_err()
        );
        assert!(builder.apply_projection(vec![(name("x"), Expr::Col(name("nope")))]).is_err());
        Ok(())
    }

    #[test]
    fn apply_group_by_rewrites_schema() -> Result<(), LowerError> {
        let arena = Arena::new();
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("a"), DataType::Int32), (name("f"), DataType::Float64)];
        let aggs = vec![Aggregation::Sum { input: name("f"), output: name("s") }];
        builder.apply_group_by(vec![(name("a"), name("a"))], aggs)?;
        assert_eq!(builder.schema, vec![
            (name("a"), DataType::Int32),
            (name("s"), DataType::Float64)
        ]);
        assert_eq!(builder.ops.len(), 1);
        assert!(matches!(builder.ops[0], PlanOp::GroupBy { .. }));
        Ok(())
    }

    #[test]
    fn all_aggregation_shapes_lower() -> Result<(), LowerError> {
        let mut arena = Arena::new();
        let column = arena.add(AExpr::Column(name("f")));
        // Every expression is built before the builder borrows the arena.
        let shapes: Vec<(AExpr, Aggregation, PlSmallStr)> = vec![
            (
                AExpr::Agg(IRAggExpr::Sum(column)),
                Aggregation::Sum { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Min { input: column, propagate_nans: false }),
                Aggregation::Min { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Max { input: column, propagate_nans: false }),
                Aggregation::Max { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Mean(column)),
                Aggregation::Mean { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Count { input: column, include_nulls: false }),
                Aggregation::Count { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Count { input: column, include_nulls: true }),
                Aggregation::Len { output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::First(column)),
                Aggregation::First { input: name("f"), output: name("s") },
                name("s"),
            ),
            (
                AExpr::Agg(IRAggExpr::Last(column)),
                Aggregation::Last { input: name("f"), output: name("s") },
                name("s"),
            ),
            // `len()` counts rows without an input column.
            (AExpr::Len, Aggregation::Len { output: name("n") }, name("n")),
        ];
        let mut accepted = Vec::with_capacity(shapes.len());
        for (shape, expected, output) in shapes {
            let node = arena.add(shape);
            accepted.push((node, expected, output));
        }
        // NaN propagation stays on the CPU engine; Median now lowers.
        let mut rejected = Vec::with_capacity(1);
        for shape in [AExpr::Agg(IRAggExpr::Min { input: column, propagate_nans: true })] {
            rejected.push(arena.add(shape));
        }
        let builder = schema_builder(&arena);
        for (node, expected, output) in accepted {
            let expr = ExprIR::new(node, OutputName::ColumnLhs(output));
            assert_eq!(builder.lower_aggregation(&expr)?, expected);
        }
        for node in rejected {
            let expr = ExprIR::new(node, OutputName::ColumnLhs(name("s")));
            assert!(builder.lower_aggregation(&expr).is_err());
        }
        Ok(())
    }

    #[test]
    fn float_literal_against_64_bit_reports_its_reason() -> Result<(), LowerError> {
        let Err(error) = float_against_dtype(1.0, &DataType::Int64, DType::I64) else {
            return Err(unsupported("64-bit float literal unexpectedly lowered"));
        };
        assert!(
            matches!(&error, LowerError::Unsupported { reason } if reason.contains("64-bit")),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn predictate_dtypes_match_kernel_for_all_supported() {
        // For every supported dtype, DType::from_polars + to_polars round-trips
        // and every byte-width/dtype variant covers the engine subset exactly.
        let cases = vec![
            (DataType::Int32, DType::I32),
            (DataType::Int64, DType::I64),
            (DataType::UInt32, DType::U32),
            (DataType::UInt64, DType::U64),
            (DataType::Float32, DType::F32),
            (DataType::Float64, DType::F64),
            (DataType::Boolean, DType::Bool),
        ];
        for (polars_dtype, engine_dtype) in cases {
            assert_eq!(DType::from_polars(&polars_dtype), Some(engine_dtype));
            assert_eq!(engine_dtype.to_polars(), polars_dtype);
        }
    }

    #[test]
    fn unsupported_dtypes_are_rejected_in_lower_path() {
        for dtype in [DataType::String, DataType::Null, DataType::Date] {
            assert!(require_dtype(&dtype).is_err(), "unsupported dtype {dtype:?} must be rejected");
        }
    }

    fn schema_builder(arena: &Arena<AExpr>) -> Builder<'_> {
        let mut builder = Builder::new(arena);
        builder.schema = vec![
            (name("a"), DataType::Int32),
            (name("f"), DataType::Float64),
            (name("flag"), DataType::Boolean),
        ];
        builder
    }
}

#[cfg(test)]
mod group_plan_tests {
    use polars_plan::{dsl::GroupbyOptions, plans::expr_ir::OutputName};

    use super::*;
    use crate::{
        plan_tests::{plan_frame, scan_node},
        predicate_tests::name,
    };

    fn group_node(
        lp_arena: &mut Arena<IR>,
        frame: DataFrame,
        keys: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
    ) -> Node {
        let schema = frame.schema().clone();
        let scan = scan_node(frame, lp_arena);
        lp_arena.add(IR::GroupBy {
            input: scan,
            keys,
            aggs,
            schema,
            maintain_order: false,
            options: Arc::new(GroupbyOptions::default()),
            apply: None,
        })
    }

    fn sum_agg(expr_arena: &mut Arena<AExpr>, output: &str) -> ExprIR {
        let column = expr_arena.add(AExpr::Column(name("f")));
        let node = expr_arena.add(AExpr::Agg(IRAggExpr::Sum(column)));
        ExprIR::new(node, OutputName::ColumnLhs(name(output)))
    }

    #[test]
    fn group_by_plan_lowers_end_to_end() -> Result<(), LowerError> {
        let mut expr_arena = Arena::new();
        let key_col = expr_arena.add(AExpr::Column(name("a")));
        let keys = vec![ExprIR::new(key_col, OutputName::ColumnLhs(name("a")))];
        let aggs = vec![sum_agg(&mut expr_arena, "s")];
        let mut lp_arena = Arena::new();
        let group = group_node(&mut lp_arena, plan_frame(), keys, aggs);
        let plan = lower(group, &lp_arena, &expr_arena)?;
        assert_eq!(plan.ops(), &[PlanOp::GroupBy {
            keys: vec![(name("a"), name("a"))],
            aggs: vec![Aggregation::Sum { input: name("f"), output: name("s") }],
        }]);
        Ok(())
    }

    #[test]
    fn group_by_rejects_duplicates_and_empty_parts() -> Result<(), LowerError> {
        // Duplicate key outputs are rejected, not silently merged.
        let mut expr_arena = Arena::new();
        let key_col = expr_arena.add(AExpr::Column(name("a")));
        let keys = vec![
            ExprIR::new(key_col, OutputName::ColumnLhs(name("a"))),
            ExprIR::new(key_col, OutputName::ColumnLhs(name("a"))),
        ];
        let aggs = vec![sum_agg(&mut expr_arena, "s")];
        let mut lp_arena = Arena::new();
        let group = group_node(&mut lp_arena, plan_frame(), keys, aggs);
        assert!(lower(group, &lp_arena, &expr_arena).is_err());
        // Duplicate aggregation outputs are rejected too.
        let mut expr_arena = Arena::new();
        let key_col = expr_arena.add(AExpr::Column(name("a")));
        let keys = vec![ExprIR::new(key_col, OutputName::ColumnLhs(name("a")))];
        let aggs = vec![sum_agg(&mut expr_arena, "s"), sum_agg(&mut expr_arena, "s")];
        let mut lp_arena = Arena::new();
        let group = group_node(&mut lp_arena, plan_frame(), keys, aggs);
        assert!(lower(group, &lp_arena, &expr_arena).is_err());
        // A group-by without keys or without aggregations is rejected.
        let mut expr_arena = Arena::new();
        let aggs = vec![sum_agg(&mut expr_arena, "s")];
        let mut lp_arena = Arena::new();
        let group = group_node(&mut lp_arena, plan_frame(), Vec::new(), aggs);
        assert!(lower(group, &lp_arena, &expr_arena).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod math_fn_tests {
    use super::*;
    use crate::predicate_tests::name;

    /// Helper to lower a unary `IRFunctionExpr` against a schema where `f` is
    /// `Float64` and `i` is `Int32`, returning the resulting `Expr`.
    fn lower_unary(
        arena: &mut Arena<AExpr>,
        function: IRFunctionExpr,
        column: &str,
    ) -> Result<Expr, LowerError> {
        let col = arena.add(AExpr::Column(name(column)));
        let input = ExprIR::from_node(col, arena);
        let node = arena.add(AExpr::Function {
            input: vec![input],
            function,
            options: Default::default(),
        });
        let mut builder = Builder::new(arena);
        builder.schema = vec![(name("f"), DataType::Float64), (name("i"), DataType::Int32)];
        builder.lower_expr(node)
    }

    /// Lower and unwrap with a clear panic message on failure.
    #[expect(clippy::expect_used)]
    fn lower_ok(arena: &mut Arena<AExpr>, function: IRFunctionExpr, column: &str) -> Expr {
        lower_unary(arena, function, column).expect("lowering should succeed")
    }

    #[test]
    fn abs_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Abs, "f"),
            Expr::Unary(UnaryOp::Abs, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn ceil_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Ceil, "f"),
            Expr::Unary(UnaryOp::Ceil, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn floor_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Floor, "f"),
            Expr::Unary(UnaryOp::Floor, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn exp_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Exp, "f"),
            Expr::Unary(UnaryOp::Exp, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn log1p_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Log1p, "f"),
            Expr::Unary(UnaryOp::Log1p, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn sign_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Sign, "f"),
            Expr::Unary(UnaryOp::Sign, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn sin_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(
                &mut arena,
                IRFunctionExpr::Trigonometry(polars_plan::plans::IRTrigonometricFunction::Sin),
                "f",
            ),
            Expr::Unary(UnaryOp::Sin, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn cos_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(
                &mut arena,
                IRFunctionExpr::Trigonometry(polars_plan::plans::IRTrigonometricFunction::Cos),
                "f",
            ),
            Expr::Unary(UnaryOp::Cos, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn sqrt_lowers_on_float_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Pow(polars_plan::plans::IRPowFunction::Sqrt), "f",),
            Expr::Unary(UnaryOp::Sqrt, Box::new(Expr::Col(name("f")))),
        );
    }

    #[test]
    fn negate_lowers_on_int_column() {
        let mut arena = Arena::new();
        assert_eq!(
            lower_ok(&mut arena, IRFunctionExpr::Negate, "i"),
            Expr::Unary(UnaryOp::Negate, Box::new(Expr::Col(name("i")))),
        );
    }

    #[test]
    fn tangent_is_rejected() {
        let mut arena = Arena::new();
        assert!(
            lower_unary(
                &mut arena,
                IRFunctionExpr::Trigonometry(polars_plan::plans::IRTrigonometricFunction::Tan),
                "f",
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_function_is_rejected() {
        let mut arena = Arena::new();
        assert!(lower_unary(&mut arena, IRFunctionExpr::Unique(false), "f").is_err());
    }

    #[test]
    fn unary_function_dtype_passthrough() {
        // The dtype of a unary math function is the dtype of its input.
        let expr = Expr::Unary(UnaryOp::Abs, Box::new(Expr::Col(name("f"))));
        let arena = Arena::new();
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("f"), DataType::Float64)];
        #[expect(clippy::expect_used)]
        let dtype = builder.expr_dtype(&expr).expect("expr_dtype should succeed");
        assert_eq!(dtype, DataType::Float64);
    }

    #[test]
    fn unary_function_over_int_preserves_int() {
        let expr = Expr::Unary(UnaryOp::Negate, Box::new(Expr::Col(name("i"))));
        let arena = Arena::new();
        let mut builder = Builder::new(&arena);
        builder.schema = vec![(name("i"), DataType::Int32)];
        #[expect(clippy::expect_used)]
        let dtype = builder.expr_dtype(&expr).expect("expr_dtype should succeed");
        assert_eq!(dtype, DataType::Int32);
    }

    #[test]
    fn float_only_math_ops_reject_int_column() {
        // abs, ceil, floor, exp, sqrt, log1p, sin, cos, sign — all
        // float-only — must be rejected at lowering when applied to an
        // integer column so the engine falls back to the CPU cleanly.
        let float_only_fns = vec![
            IRFunctionExpr::Abs,
            IRFunctionExpr::Ceil,
            IRFunctionExpr::Floor,
            IRFunctionExpr::Exp,
            IRFunctionExpr::Log1p,
            IRFunctionExpr::Sign,
            IRFunctionExpr::Pow(polars_plan::plans::IRPowFunction::Sqrt),
            IRFunctionExpr::Trigonometry(polars_plan::plans::IRTrigonometricFunction::Sin),
            IRFunctionExpr::Trigonometry(polars_plan::plans::IRTrigonometricFunction::Cos),
        ];
        for function in float_only_fns {
            let mut arena = Arena::new();
            assert!(
                lower_unary(&mut arena, function.clone(), "i").is_err(),
                "expected {function:?} on int column to be rejected"
            );
        }
    }

    #[test]
    fn negate_is_not_float_only_and_accepts_int_column() {
        let mut arena = Arena::new();
        assert!(lower_unary(&mut arena, IRFunctionExpr::Negate, "i").is_ok());
    }
}
