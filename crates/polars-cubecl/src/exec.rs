//! [`CubePlan`] execution on the selected CubeCL backend.
//!
//! The executor ingests source columns to the device once, applies each
//! pipeline step through the kernel launch wrappers, and downloads the final
//! frame. Data stays resident between steps; the only host round-trips are
//! ingest, the tiny per-thread tally prefix sums of filtering, key download
//! plus group layout for group-by, and the final download.
//!
//! This module contains no `unsafe`: all buffer wrapping lives in the
//! launch wrappers of `polars-cubecl-kernels`.

use cubecl::prelude::Client;
use polars_core::frame::DataFrame;
use polars_cubecl_buffer::{
    BufferError, DType, DeviceColumn, HostValues, download, download_columns, download_frame,
    series_to_host, upload,
};
use polars_cubecl_kernels::{
    BinOp as KernelBinOp, CmpOp as KernelCmpOp, KernelError, UnaryOp as KernelUnaryOp,
    launch_cast_i32_to_f64, launch_cast_u32_to_f64,
};
use polars_cubecl_lower::{
    BinOp, CmpOp, CubePlan, Expr, PlanOp, PlanSource, Predicate, ScalarLit, UnaryOp,
};
use polars_cubecl_runtime::{RuntimeContext, supports_f64};
use polars_utils::pl_str::PlSmallStr;
use thiserror::Error;

use crate::buffer_pool;
use crate::groupby;

/// Error returned when a lowered plan cannot be executed.
#[derive(Debug, Error)]
pub enum ExecError {
    /// Host/device transfer failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// A kernel launch could not be set up.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// Kernel execution or a readback failed at runtime.
    #[error("kernel execution failed: {0}")]
    Launch(String),
    /// The plan is internally inconsistent (lowering validated it, so this
    /// indicates an engine bug, never a user error).
    #[error("inconsistent plan during execution: {0}")]
    Inconsistent(String),
    /// The plan lowered but this executor revision does not run the op yet.
    #[error("lowered op not executable yet: {0}")]
    NotExecutable(String),
    /// The plan needs an operation this backend cannot run (e.g. `f64`
    /// arithmetic on a backend without `double`). The caller turns this into
    /// CPU fallback (or a strict-mode error in tests).
    #[error("operation not supported by this backend: {0}")]
    BackendUnsupported(String),
}

/// Execute a lowered plan, returning the result frame and the ops that ran.
pub fn execute_plan(
    ctx: &RuntimeContext,
    plan: &CubePlan,
) -> Result<(DataFrame, Vec<crate::Op>), ExecError> {
    let client = ctx.client();
    let f64_ok = supports_f64(ctx.kind());
    let mut columns = ingest(client, plan.source(), plan)?;
    let mut ops = Vec::with_capacity(plan.ops().len());
    for op in plan.ops() {
        match op {
            PlanOp::Filter(predicate) => {
                columns = apply_filter(client, f64_ok, columns, predicate)?;
                ops.push(crate::Op::Filter);
            }
            PlanOp::Project(projection) => {
                columns = apply_project(client, f64_ok, columns, projection)?;
                ops.push(crate::Op::Project);
            }
            PlanOp::Slice { offset, len } => {
                columns = apply_slice(client, columns, *offset, *len)?;
                ops.push(crate::Op::Slice);
            }
            PlanOp::GroupBy { keys, aggs } => {
                columns = groupby::apply(client, f64_ok, columns, keys, aggs)?;
                ops.push(crate::Op::GroupBy);
            }
            PlanOp::Sort { by, descending, nulls_last, slice } => {
                columns = apply_sort(client, columns, by, *descending, *nulls_last, *slice)?;
                ops.push(crate::Op::Sort);
            }
        }
    }
    let frame = download_frame(client, &columns)?;
    // Recycle device buffers into the pool for reuse by subsequent plans.
    // After download, columns are no longer needed; their handles go back to
    // the thread-local pool so the next call's alloc() avoids cudaMalloc.
    recycle_device_columns(&columns);
    Ok((frame, ops))
}

/// Recycle every buffer backing a plan's device columns. Called after the
/// result has been downloaded; the columns are about to be dropped, so
/// pooling their handles turns a `cudaMalloc`+`cudaFree` pair per op into
/// `acquire`+`recycle` of an existing buffer.
fn recycle_device_columns(columns: &[DeviceColumn]) {
    for col in columns {
        let value_bytes = col.len().saturating_mul(dtype_size(col.dtype()));
        if value_bytes > 0 {
            recycle_handle(col.values().clone(), value_bytes);
        }
        if col.len() > 0 {
            recycle_handle(col.validity().clone(), col.len());
        }
    }
}

fn ingest(
    client: &Client,
    source: &PlanSource,
    plan: &CubePlan,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let needed = plan.needed_source_columns();
    if needed.is_empty() {
        // No source columns survive to any op — this can happen for a pure
        // zero-column plan (unreachable in practice because lowering rejects
        // it, but be defensive).
        return Ok(Vec::new());
    }
    match source {
        PlanSource::DataFrame { frame, .. } => ingest_frame(client, frame, &needed),
        PlanSource::Parquet { path, .. } => {
            let frame = read_parquet_file(path)?;
            ingest_frame(client, &frame, &needed)
        }
    }
}

fn ingest_frame(
    client: &Client,
    frame: &DataFrame,
    columns: &[PlSmallStr],
) -> Result<Vec<DeviceColumn>, ExecError> {
    let mut device = Vec::with_capacity(columns.len());
    for name in columns {
        let series =
            frame.column(name).map_err(|error| ExecError::Inconsistent(error.to_string()))?;
        device.push(upload(client, &series_to_host(series.as_materialized_series())?));
    }
    Ok(device)
}

/// Read a parquet file on the host; the device then ingests only the
/// projected columns, so file IO is the only unaccelerated step.
#[cfg(feature = "parquet")]
fn read_parquet_file(path: &str) -> Result<DataFrame, ExecError> {
    use polars_lazy::frame::LazyFrame;
    use polars_utils::pl_path::PlRefPath;
    LazyFrame::scan_parquet(PlRefPath::new(path), Default::default())
        .and_then(|lf| lf.collect())
        .map_err(|error| ExecError::Launch(format!("parquet read failed: {error}")))
}

/// Without the `parquet` feature the executor cannot read files; lowering
/// already rejects scans there, so this is unreachable in practice.
#[cfg(not(feature = "parquet"))]
fn read_parquet_file(path: &str) -> Result<DataFrame, ExecError> {
    let _ = path;
    Err(ExecError::NotExecutable("parquet read needs the parquet feature".to_string()))
}

pub(crate) fn find_column(columns: &[DeviceColumn], name: &PlSmallStr) -> Result<usize, ExecError> {
    columns
        .iter()
        .position(|column| column.name() == name)
        .ok_or_else(|| ExecError::Inconsistent(format!("missing column {name}")))
}

fn height_of(columns: &[DeviceColumn]) -> usize {
    columns.first().map_or(0, DeviceColumn::len)
}

/// Allocate `bytes` on the device.
///
/// Some backends (notably Metal) refuse zero-byte allocations, so empty
/// buffers get a one-byte stand-in. Lengths are always tracked separately
/// in [`DeviceColumn`], and kernels never touch rows beyond the real length,
/// so the padding is never read or written.
///
/// Pooled: returns a buffer from the thread-local pool when one of matching
/// size is available, avoiding the ~10-100 µs CUDA `cudaMalloc` cost on hot
/// paths. Recycle via [`recycle_handle`] when the buffer is no longer needed.
pub(crate) fn alloc(client: &Client, bytes: usize) -> cubecl_runtime::server::Handle {
    buffer_pool::acquire(client, bytes)
}

/// Return a kernel output buffer to the pool for reuse by a future call to
/// [`alloc`]. Skip on zero-byte (padded) buffers: re-pooling them saves
/// nothing because they'll just round up to the same 64-byte bucket.
pub(crate) fn recycle_handle(handle: cubecl_runtime::server::Handle, bytes: usize) {
    if bytes > 0 {
        buffer_pool::recycle(handle, bytes);
    }
}

pub(crate) const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
        DType::Bool => 1,
    }
}

/// Elementwise comparison of `values` against a single literal.
/// Lengths are tracked separately everywhere, so the padding is inert.
pub(crate) fn create_slice(client: &Client, bytes: &[u8]) -> cubecl_runtime::server::Handle {
    client.create_from_slice(if bytes.is_empty() { &[0] } else { bytes })
}

/// Evaluate one projection: each output column is an [`Expr`] tree over the
/// resident input columns. Trees are evaluated bottom-up; intermediate
/// results stay resident between steps.
fn apply_project(
    client: &Client,
    f64_ok: bool,
    columns: Vec<DeviceColumn>,
    projection: &[(PlSmallStr, Expr)],
) -> Result<Vec<DeviceColumn>, ExecError> {
    let height = height_of(&columns);
    let mut output = Vec::with_capacity(projection.len());
    for (output_name, expr) in projection {
        let result = eval_expr(client, f64_ok, &columns, expr, height)?;
        output.push(DeviceColumn::from_handles(
            output_name.clone(),
            result.dtype(),
            result.values().clone(),
            result.validity().clone(),
            result.len(),
        ));
    }
    Ok(output)
}

/// Recursively evaluate an [`Expr`] tree against the resident columns.
fn eval_expr(
    client: &Client,
    f64_ok: bool,
    columns: &[DeviceColumn],
    expr: &Expr,
    height: usize,
) -> Result<DeviceColumn, ExecError> {
    match expr {
        Expr::Col(name) => {
            let index = find_column(columns, name)?;
            Ok(columns[index].clone())
        }
        Expr::Lit(lit) => eval_literal(client, *lit, height),
        Expr::BinOp(op, left, right) => {
            eval_binop(client, f64_ok, columns, *op, left, right, height)
        }
        Expr::Unary(op, inner) => eval_unary(client, f64_ok, columns, *op, inner, height),
    }
}

/// Materialize a scalar literal into a full device column: every row holds
/// the literal value, and the validity mask is all-valid.
fn eval_literal(client: &Client, lit: ScalarLit, height: usize) -> Result<DeviceColumn, ExecError> {
    let dtype = match lit {
        ScalarLit::I32(_) => DType::I32,
        ScalarLit::I64(_) => DType::I64,
        ScalarLit::U32(_) => DType::U32,
        ScalarLit::U64(_) => DType::U64,
        ScalarLit::F32(_) => DType::F32,
        ScalarLit::F64(_) => DType::F64,
        ScalarLit::Bool(_) => DType::Bool,
    };
    // Build a host-side dense buffer of the scalar repeated `height` times,
    // then upload it. This avoids needing a dedicated fill kernel.
    let raw = scalar_to_bytes(lit);
    let mut value_bytes = Vec::with_capacity(height.saturating_mul(raw.len()));
    for _ in 0..height {
        value_bytes.extend_from_slice(&raw);
    }
    let values = client.create_from_slice(if value_bytes.is_empty() { &[0] } else { &value_bytes });
    let validity_bytes = vec![1_u8; height];
    let validity =
        client.create_from_slice(if validity_bytes.is_empty() { &[0] } else { &validity_bytes });
    Ok(DeviceColumn::from_handles(PlSmallStr::from_static(""), dtype, values, validity, height))
}

/// Encode a [`ScalarLit`] as little-endian bytes for host-side broadcast.
fn scalar_to_bytes(lit: ScalarLit) -> Vec<u8> {
    match lit {
        ScalarLit::I32(v) => v.to_le_bytes().to_vec(),
        ScalarLit::I64(v) => v.to_le_bytes().to_vec(),
        ScalarLit::U32(v) => v.to_le_bytes().to_vec(),
        ScalarLit::U64(v) => v.to_le_bytes().to_vec(),
        ScalarLit::F32(v) => v.to_le_bytes().to_vec(),
        ScalarLit::F64(v) => v.to_le_bytes().to_vec(),
        ScalarLit::Bool(v) => vec![u8::from(v)],
    }
}

/// Evaluate a binary operation on two (recursively evaluated) sub-expressions.
///
/// Both sides are evaluated in full; the result is compared or combined
/// elementwise. Literal fast-paths are not yet implemented — when they are,
/// they should avoid materialising a full-height literal column.
fn eval_binop(
    client: &Client,
    f64_ok: bool,
    columns: &[DeviceColumn],
    op: BinOp,
    left: &Expr,
    right: &Expr,
    height: usize,
) -> Result<DeviceColumn, ExecError> {
    let lhs = eval_expr(client, f64_ok, columns, left, height)?;
    let rhs = eval_expr(client, f64_ok, columns, right, height)?;

    // Comparisons always yield `Bool`.
    if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
        return eval_col_cmp(client, f64_ok, &lhs, &rhs, cmp_op(op), height);
    }

    // Arithmetic: promote to the wider dtype.
    let out_dtype = promote_dtype(lhs.dtype(), rhs.dtype())?;
    if out_dtype == DType::F64 && !f64_ok {
        return Err(ExecError::BackendUnsupported(
            "f64 arithmetic needs double support".to_string(),
        ));
    }

    let lhs = maybe_cast(client, &lhs, out_dtype)?;
    let rhs = maybe_cast(client, &rhs, out_dtype)?;

    let out = alloc(client, height.saturating_mul(dtype_size(out_dtype)));
    let out_valid = alloc(client, height);
    dispatch_binop(client, &lhs, &rhs, &out, &out_valid, height, kernel_binop(op), out_dtype)?;
    Ok(DeviceColumn::from_handles(PlSmallStr::from_static(""), out_dtype, out, out_valid, height))
}

/// Evaluate a unary operation on a (recursively evaluated) sub-expression.
fn eval_unary(
    client: &Client,
    f64_ok: bool,
    columns: &[DeviceColumn],
    op: UnaryOp,
    inner: &Expr,
    height: usize,
) -> Result<DeviceColumn, ExecError> {
    let input = eval_expr(client, f64_ok, columns, inner, height)?;
    let dtype = input.dtype();

    // Float-only math functions require float dtypes.
    match op {
        UnaryOp::Abs |
        UnaryOp::Ceil |
        UnaryOp::Floor |
        UnaryOp::Exp |
        UnaryOp::Sqrt |
        UnaryOp::Log1p |
        UnaryOp::Sin |
        UnaryOp::Cos |
        UnaryOp::Sign => {
            if !matches!(dtype, DType::F32 | DType::F64) {
                return Err(ExecError::NotExecutable(format!(
                    "{op:?} requires a float dtype, got {dtype:?}",
                )));
            }
        }
        UnaryOp::Negate => {}
    }
    if dtype == DType::F64 && !f64_ok {
        return Err(ExecError::BackendUnsupported("f64 unary op needs double support".to_string()));
    }

    let out = alloc(client, height.saturating_mul(dtype_size(dtype)));
    let out_valid = alloc(client, height);
    dispatch_unaryop(client, &input, &out, &out_valid, height, kernel_unary_op(op), dtype)?;
    Ok(DeviceColumn::from_handles(PlSmallStr::from_static(""), dtype, out, out_valid, height))
}

/// Evaluate a column-vs-column comparison.
fn eval_col_cmp(
    client: &Client,
    f64_ok: bool,
    lhs: &DeviceColumn,
    rhs: &DeviceColumn,
    op: CmpOp,
    height: usize,
) -> Result<DeviceColumn, ExecError> {
    let out_dtype = promote_dtype(lhs.dtype(), rhs.dtype())?;
    if out_dtype == DType::F64 && !f64_ok {
        return Err(ExecError::BackendUnsupported(
            "f64 comparison needs double support".to_string(),
        ));
    }
    let lhs = maybe_cast(client, lhs, out_dtype)?;
    let rhs = maybe_cast(client, rhs, out_dtype)?;

    let out = alloc(client, height);
    let out_valid = alloc(client, height);
    dispatch_col_cmp(client, &lhs, &rhs, &out, &out_valid, height, kernel_cmp_op(op), out_dtype)?;
    Ok(DeviceColumn::from_handles(PlSmallStr::from_static(""), DType::Bool, out, out_valid, height))
}

// --- Kernel dispatch -----------------------------------------------------

/// Dispatch a binary arithmetic kernel for `dtype`.
fn dispatch_binop(
    client: &Client,
    lhs: &DeviceColumn,
    rhs: &DeviceColumn,
    out: &cubecl_runtime::server::Handle,
    out_valid: &cubecl_runtime::server::Handle,
    height: usize,
    op: KernelBinOp,
    dtype: DType,
) -> Result<(), ExecError> {
    match dtype {
        DType::I32 => polars_cubecl_kernels::launch_binop::<i32>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_binop::<i64>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_binop::<u32>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U64 => polars_cubecl_kernels::launch_binop::<u64>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_binop_f32(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F64 => polars_cubecl_kernels::launch_binop_f64(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::Bool => {
            return Err(ExecError::NotExecutable(
                "binary arithmetic on boolean not supported".to_string(),
            ))
        }
    }
    Ok(())
}

/// Dispatch a unary kernel for `dtype`.
fn dispatch_unaryop(
    client: &Client,
    input: &DeviceColumn,
    out: &cubecl_runtime::server::Handle,
    out_valid: &cubecl_runtime::server::Handle,
    height: usize,
    op: KernelUnaryOp,
    dtype: DType,
) -> Result<(), ExecError> {
    match dtype {
        DType::I32 => polars_cubecl_kernels::launch_unaryop::<i32>(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_unaryop::<i64>(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_unaryop::<u32>(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U64 => polars_cubecl_kernels::launch_unaryop::<u64>(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_unaryop_f32(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F64 => polars_cubecl_kernels::launch_unaryop_f64(
            client,
            input.values(),
            input.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::Bool => {
            return Err(ExecError::NotExecutable("unary ops on boolean not supported".to_string()))
        }
    }
    Ok(())
}

/// Dispatch a column-vs-column comparison kernel for `dtype`.
fn dispatch_col_cmp(
    client: &Client,
    lhs: &DeviceColumn,
    rhs: &DeviceColumn,
    out: &cubecl_runtime::server::Handle,
    out_valid: &cubecl_runtime::server::Handle,
    height: usize,
    op: KernelCmpOp,
    dtype: DType,
) -> Result<(), ExecError> {
    match dtype {
        DType::I32 => polars_cubecl_kernels::launch_col_cmp::<i32>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_col_cmp::<i64>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_col_cmp::<u32>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::U64 => polars_cubecl_kernels::launch_col_cmp::<u64>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_col_cmp_f32(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::F64 => polars_cubecl_kernels::launch_col_cmp_f64(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
        DType::Bool => polars_cubecl_kernels::launch_col_cmp::<u8>(
            client,
            lhs.values(),
            lhs.validity(),
            rhs.values(),
            rhs.validity(),
            out,
            out_valid,
            height,
            op,
        )?,
    }
    Ok(())
}

// --- Type promotion ------------------------------------------------------

/// Type promotion for the engine's subset over [`DType`]. Mirrors the
/// `promote_binary` logic in the lower crate.
fn promote_dtype(left: DType, right: DType) -> Result<DType, ExecError> {
    if left == right {
        return Ok(left);
    }
    if left == DType::F64 || right == DType::F64 {
        return Ok(DType::F64);
    }
    if left == DType::F32 || right == DType::F32 {
        return Ok(DType::F32);
    }
    let (lb, ls) = int_width(left)?;
    let (rb, rs) = int_width(right)?;
    if ls == rs {
        return Ok(if lb >= rb { left } else { right });
    }
    // Mixed signedness widens to at least `I64`.
    Ok(DType::I64)
}

fn int_width(dtype: DType) -> Result<(u32, bool), ExecError> {
    match dtype {
        DType::I32 => Ok((32, true)),
        DType::I64 => Ok((64, true)),
        DType::U32 => Ok((32, false)),
        DType::U64 => Ok((64, false)),
        _ => Err(ExecError::Inconsistent("non-integer operand in binary op".to_string())),
    }
}

/// Cast `column` to `target` dtype. Only casts with an existing kernel are
/// supported; everything else falls back to the caller.
fn maybe_cast(
    client: &Client,
    column: &DeviceColumn,
    target: DType,
) -> Result<DeviceColumn, ExecError> {
    if column.dtype() == target {
        return Ok(column.clone());
    }
    match (column.dtype(), target) {
        (DType::I32, DType::F64) => {
            let out = alloc(client, column.len() * 8);
            launch_cast_i32_to_f64(client, column.values(), &out, column.len())?;
            Ok(DeviceColumn::from_handles(
                column.name().clone(),
                target,
                out,
                column.validity().clone(),
                column.len(),
            ))
        }
        (DType::U32, DType::F64) => {
            let out = alloc(client, column.len() * 8);
            launch_cast_u32_to_f64(client, column.values(), &out, column.len())?;
            Ok(DeviceColumn::from_handles(
                column.name().clone(),
                target,
                out,
                column.validity().clone(),
                column.len(),
            ))
        }
        _ => Err(ExecError::Inconsistent(format!(
            "unsupported cast from {:?} to {:?}",
            column.dtype(),
            target,
        ))),
    }
}

// --- Op mapping ----------------------------------------------------------

/// Map a lower [`BinOp`] to a [`KernelBinOp`] for arithmetic dispatch.
fn kernel_binop(op: BinOp) -> KernelBinOp {
    match op {
        BinOp::Add => KernelBinOp::Add,
        BinOp::Sub => KernelBinOp::Sub,
        BinOp::Mul => KernelBinOp::Mul,
        BinOp::Div => KernelBinOp::Div,
        BinOp::FloorDiv => KernelBinOp::FloorDiv,
        BinOp::Mod => KernelBinOp::Mod,
        BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
            unreachable!("comparison ops dispatched via eval_col_cmp")
        }
    }
}

/// Map a lower [`BinOp`] comparison to a [`CmpOp`] (lower enum).
fn cmp_op(op: BinOp) -> CmpOp {
    match op {
        BinOp::Eq => CmpOp::Eq,
        BinOp::NotEq => CmpOp::NotEq,
        BinOp::Lt => CmpOp::Lt,
        BinOp::LtEq => CmpOp::LtEq,
        BinOp::Gt => CmpOp::Gt,
        BinOp::GtEq => CmpOp::GtEq,
        _ => unreachable!("cmp_op called on arithmetic op"),
    }
}

/// Map a lower [`CmpOp`] to a [`KernelCmpOp`] (kernels enum).
fn kernel_cmp_op(op: CmpOp) -> KernelCmpOp {
    match op {
        CmpOp::Eq => KernelCmpOp::Eq,
        CmpOp::NotEq => KernelCmpOp::NotEq,
        CmpOp::Lt => KernelCmpOp::Lt,
        CmpOp::LtEq => KernelCmpOp::LtEq,
        CmpOp::Gt => KernelCmpOp::Gt,
        CmpOp::GtEq => KernelCmpOp::GtEq,
    }
}

/// Map a lower [`UnaryOp`] to a [`KernelUnaryOp`] (kernels enum).
fn kernel_unary_op(op: UnaryOp) -> KernelUnaryOp {
    match op {
        UnaryOp::Abs => KernelUnaryOp::Abs,
        UnaryOp::Ceil => KernelUnaryOp::Ceil,
        UnaryOp::Floor => KernelUnaryOp::Floor,
        UnaryOp::Exp => KernelUnaryOp::Exp,
        UnaryOp::Sqrt => KernelUnaryOp::Sqrt,
        UnaryOp::Log1p => KernelUnaryOp::Log1p,
        UnaryOp::Sin => KernelUnaryOp::Sin,
        UnaryOp::Cos => KernelUnaryOp::Cos,
        UnaryOp::Sign => KernelUnaryOp::Sign,
        UnaryOp::Negate => KernelUnaryOp::Negate,
    }
}

fn apply_slice(
    client: &Client,
    columns: Vec<DeviceColumn>,
    offset: i64,
    len: u32,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let height = height_of(&columns);
    let start = if offset >= 0 {
        (offset as usize).min(height)
    } else {
        height.saturating_sub(offset.unsigned_abs() as usize)
    };
    let end = start.saturating_add(len as usize).min(height);
    let count = end.saturating_sub(start);
    let mut output = Vec::with_capacity(columns.len());
    for column in &columns {
        let values = slice_column(client, column, start, count)?;
        let validity = slice_mask(client, column.validity(), column.len(), start, count)?;
        output.push(DeviceColumn::from_handles(
            column.name().clone(),
            column.dtype(),
            values,
            validity,
            count,
        ));
    }
    Ok(output)
}

fn apply_filter(
    client: &Client,
    f64_ok: bool,
    columns: Vec<DeviceColumn>,
    predicate: &Predicate,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let height = height_of(&columns);
    if height == 0 {
        return Ok(columns);
    }
    let mask = eval_predicate(client, f64_ok, predicate, &columns, height)?;
    let (indices, kept) = compact_indices(client, &mask, height)?;
    let mut output = Vec::with_capacity(columns.len());
    for column in &columns {
        let (values, validity) = gather_values_mask(client, column, &indices, kept)?;
        output.push(DeviceColumn::from_handles(
            column.name().clone(),
            column.dtype(),
            values,
            validity,
            kept,
        ));
    }
    Ok(output)
}

/// Sort all columns by a single key column.
///
/// The engine downloads the key column, computes the sort permutation on the
/// host, then uploads the index vector and reorders every column through the
/// fused gather infrastructure (values + validity in one launch per column).
/// A trailing `(offset, len)` window is applied after sorting.
///
/// Uses GPU radix sort for large u32/u64/i32/i64 key columns; falls back to
/// host sorting for other types or small inputs where the DMA round-trips
/// are cheaper than extra kernel launches.
fn apply_sort(
    client: &Client,
    columns: Vec<DeviceColumn>,
    by: &PlSmallStr,
    descending: bool,
    nulls_last: bool,
    slice: Option<(i64, usize)>,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let height = height_of(&columns);
    if height == 0 {
        return Ok(columns);
    }

    // Fast path for small row counts: download everything and sort on the
    // host. Avoids N-column × kernel launch overhead, which dominates for
    // tiny frames (e.g. group_by output with thousands of groups).
    const SORT_HOST_MAX: usize = 1024;
    if height <= SORT_HOST_MAX {
        return apply_sort_host(client, columns, by, descending, nulls_last, slice);
    }

    let key = columns
        .iter()
        .find(|column| column.name() == by)
        .ok_or_else(|| ExecError::Inconsistent(format!("sort key {by:?} missing from columns")))?;

    // GPU radix sort is currently disabled. The device-side prefix-sum kernel
    // (`radix_offsets_kernel`) runs single-threaded and is orders of magnitude
    // slower than a CPU prefix sum for typical block counts (~1K-10K). To make
    // the GPU path win, it needs a multi-threaded parallel prefix-sum kernel
    // (work-efficient Blelloch scan) — tracked in Opt 3. Until then, all
    // downloads go through the host-cooperative path below.
    let indices = if false && matches!(key.dtype(), DType::U32 | DType::U64 | DType::I32 | DType::I64)
        && height >= _GPU_SORT_THRESHOLD
    {
        gpu_radix_sort_key_indices(client, key, descending, nulls_last)?
    } else {
        // Host-cooperative path: download key, sort permutation on CPU,
        // upload indices to device.
        let key_host = download(client, key)?;
        let mut permutation: Vec<u32> = (0..height).map(|i| i as u32).collect();
        sort_permutation(
            key_host.values(),
            key_host.validity(),
            descending,
            nulls_last,
            &mut permutation,
        );
        client.create_from_slice(bytemuck::cast_slice(&permutation))
    };

    // Reorder every column using the fused gather kernel (values + validity).
    apply_sort_gather(client, columns, &indices, height, slice)
}

/// Sort small frames entirely on the host: download the key column, compute
/// the permutation, then reindex every column on the host too. One sync.
fn apply_sort_host(
    client: &Client,
    columns: Vec<DeviceColumn>,
    by: &PlSmallStr,
    descending: bool,
    nulls_last: bool,
    slice: Option<(i64, usize)>,
) -> Result<Vec<DeviceColumn>, ExecError> {
    // Batch-download all columns in one read-back round-trip.
    let hosts = download_columns(client, &columns)?;
    let key_host = hosts
        .iter()
        .find(|column| column.name() == by)
        .ok_or_else(|| ExecError::Inconsistent(format!("sort key {by:?} missing")))?;
    let height = key_host.len();
    let mut permutation: Vec<u32> = (0..height).map(|i| i as u32).collect();
    sort_permutation(
        key_host.values(),
        key_host.validity(),
        descending,
        nulls_last,
        &mut permutation,
    );

    // Apply the slice window to the permutation.
    let permutation = if let Some((offset, len)) = slice {
        let start = if offset >= 0 {
            (offset as usize).min(height)
        } else {
            height.saturating_sub(offset.unsigned_abs() as usize)
        };
        let end = start.saturating_add(len).min(height);
        permutation[start..end].to_vec()
    } else {
        permutation
    };
    let new_height = permutation.len();

    // Reindex each column on the host; re-upload all.
    let mut output = Vec::with_capacity(columns.len());
    for (host, column) in hosts.into_iter().zip(columns.iter()) {
        let values = permute_values(host.values(), &permutation);
        let host_validity = host.validity();
        let validity: Vec<u8> = permutation.iter().map(|&i| host_validity[i as usize]).collect();
        let values_handle = client.create_from_slice(bytemuck::cast_slice(&values.to_bytes()));
        let validity_handle = client.create_from_slice(&validity[..new_height]);
        output.push(DeviceColumn::from_handles(
            column.name().clone(),
            column.dtype(),
            values_handle,
            validity_handle,
            new_height,
        ));
    }
    Ok(output)
}

/// Permute `values` by `permutation`, returning a new `HostValues`.
fn permute_values(values: &HostValues, permutation: &[u32]) -> HostValues {
    match values {
        HostValues::I32(v) => HostValues::I32(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::I64(v) => HostValues::I64(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::U32(v) => HostValues::U32(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::U64(v) => HostValues::U64(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::F32(v) => HostValues::F32(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::F64(v) => HostValues::F64(permutation.iter().map(|&i| v[i as usize]).collect()),
        HostValues::Bool(v) => HostValues::Bool(permutation.iter().map(|&i| v[i as usize]).collect()),
    }
}

/// GPU gather: reorder every column by `indices` on the device, then
/// optionally apply a trailing slice window.
fn apply_sort_gather(
    client: &Client,
    columns: Vec<DeviceColumn>,
    indices: &cubecl_runtime::server::Handle,
    height: usize,
    slice: Option<(i64, usize)>,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let mut output = Vec::with_capacity(columns.len());
    for column in &columns {
        let (values, validity) = gather_values_mask(client, column, indices, height)?;
        output.push(DeviceColumn::from_handles(
            column.name().clone(),
            column.dtype(),
            values,
            validity,
            height,
        ));
    }
    if let Some((offset, len)) = slice {
        output = apply_slice(client, output, offset, len as u32)?;
    }
    Ok(output)
}

/// Below this row count, the key DMA transfer + CPU sort is faster than the
/// many kernel launches GPU radix sort needs (8 passes × sync points). GPU
/// radix sort wins only at very large sizes where its O(n) passes beat the
/// CPU's O(n log n) even with the sync overhead. RTX 3090 Ti testing shows
/// this crossover is well above 10 M rows; until the GPU path eliminates
/// per-pass `sync`, we keep all sorts on the host-cooperative path.
pub(crate) const _GPU_SORT_THRESHOLD: usize = 256 * 1024;

/// Sort the indices `[0, height)` by the key column on the GPU using LSD
/// radix sort. Returns a device buffer of sorted `u32` indices.
///
/// The key column stays on the device throughout. The only host round-trips
/// are the tiny `num_blocks × 256`-word histograms computed per digit pass
/// (a few hundred bytes each), never the full key column nor the full
/// permutation vector.
///
/// ## Type handling
///
/// | Key dtype | Transform applied before sort | `descending` flip |
/// |-----------|-------------------------------|--------------------|
/// | `I32`     | sign-flip (`x ^ 1<<31`)       | NOT the sign-flip  |
/// | `I64`     | sign-flip (`x ^ 1<<63`)       | NOT the sign-flip  |
/// | `U32`     | identity (already unsigned)   | NOT the raw value  |
/// | `U64`     | identity (already unsigned)   | NOT the raw value  |
///
/// For `descending`, the sign-flipped (signed) or plain (unsigned) key is
/// additionally bitwise-NOT'd, which reverses unsigned ordering and turns the
/// natural ascending radix-sort into a correct descending sort.
///
/// Nulls: the GPU scatter kernel is not mask-aware, so this routine falls
/// back to the host-sort path when any null is present in the key column.
fn gpu_radix_sort_key_indices(
    client: &Client,
    key: &DeviceColumn,
    descending: bool,
    nulls_last: bool,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    use polars_cubecl_kernels::{
        RADIX_BINS, RADIX_PASSES_U32, RADIX_PASSES_U64, launch_fill_index,
        launch_not_u32, launch_not_u64, launch_radix_histogram_u32, launch_radix_histogram_u64,
        launch_radix_offsets, launch_radix_scatter_u32, launch_radix_scatter_u64,
        launch_sign_flip_u32, launch_sign_flip_u64,
    };

    let height = key.len();
    assert!(height > 0, "apply_sort short-circuits height == 0");

    // Only integer keys (signed or unsigned, 32/64-bit) get the GPU path.
    let (passes, elem_size) = match key.dtype() {
        DType::I32 => (RADIX_PASSES_U32, 4usize),
        DType::I64 => (RADIX_PASSES_U64, 8),
        DType::U32 => (RADIX_PASSES_U32, 4),
        DType::U64 => (RADIX_PASSES_U64, 8),
        _ => return gpu_sort_host_fallback(client, key, descending, nulls_last),
    };

    // Columns with nulls fall back: the scatter kernel does not honour the
    // validity mask, so a sentinel pre-pass would be required.
    if key_validity_has_nulls(client, key, height)? {
        return gpu_sort_host_fallback(client, key, descending, nulls_last);
    }

    // Topology: one block-thread per histogram block, each scans `grid` rows.
    // Keep num_blocks small so the histogram download is cheap.
    let num_blocks = height.div_ceil(1024).max(1);
    let grid = height.div_ceil(num_blocks);
    let passes_usize = passes as usize;

    // Ping-pong buffers for LSD passes. `keys_a`/`keys_b` hold either the
    // sign-flipped signed key (for I32/I64) or a device copy of the raw
    // unsigned key (for U32/U64); the sign-flip/Copy and NOT kernels write
    // to a *different* buffer than they read (no in-place support), so we
    // use the ping-pong pair.
    let mut keys_a = alloc(client, height * elem_size);
    let mut keys_b = alloc(client, height * elem_size);
    let indices = alloc(client, height * 4);
    let indices_scratch = alloc(client, height * 4);
    let histograms = alloc(client, num_blocks * RADIX_BINS as usize * 4);

    // Step 1: prepare a sortable key buffer.
    //   - Signed types: sign-flip → keys_a (so i32/i64 sort as unsigned).
    //   - Unsigned types: identity. We read directly into keys_a via the
    //     ping-pong's second buffer, then swap pointers so keys_a is the
    //     working source.
    // For descending: additionally NOT the prepared key right after, which
    // reverses unsigned ordering within the same buffer pair.
    match key.dtype() {
        DType::I32 => {
            launch_sign_flip_u32(client, key.values(), height, &keys_a)?;
            if descending {
                launch_not_u32(client, &keys_a, height, &keys_b)?;
                std::mem::swap(&mut keys_a, &mut keys_b);
            }
        }
        DType::I64 => {
            launch_sign_flip_u64(client, key.values(), height, &keys_a)?;
            if descending {
                launch_not_u64(client, &keys_a, height, &keys_b)?;
                std::mem::swap(&mut keys_a, &mut keys_b);
            }
        }
        DType::U32 => {
            // Raw unsigned values are already in sortable order; use them
            // directly as the first source to avoid an extra copy.
            // If descending, NOT into keys_a; otherwise skip — `keys_a`
            // remains uninitialised but used as *destination* in pass 0.
            if descending {
                launch_not_u32(client, key.values(), height, &keys_a)?;
            }
            // For ascending U32, we use `key.values()` directly as src by
            // aliasing: a thin copy would cost a kernel pass. Instead, treat
            // the raw key as the initial source; the first scatter pass
            // writes to `keys_b` — exactly what the ping-pong expects.
        }
        DType::U64 => {
            if descending {
                launch_not_u64(client, key.values(), height, &keys_a)?;
            }
        }
        _ => unreachable!(),
    }

    launch_fill_index(client, &indices, height)?;

    // Allocate a scratch buffer for device-side offset computation. Reused
    // across passes to avoid repeated alloc/free.
    let scatter_offsets = alloc(client, num_blocks * RADIX_BINS as usize * 4);

    // Step 2: LSD radix sort.
    // Unsigned ascending is a special case: the raw key buffer (`key.values()`)
    // is the initial source. All other paths use keys_a.
    let mut src_keys = if matches!(key.dtype(), DType::U32 | DType::U64) && !descending {
        key.values().clone()
    } else {
        keys_a
    };
    let mut dst_keys = keys_b;
    let mut src_indices = indices;
    let mut dst_indices = indices_scratch;

    for pass_idx in 0..passes_usize {
        let shift_u32 = (pass_idx as u32) * 8;
        let shift_u64 = shift_u32 as u64;

        match key.dtype() {
            DType::I32 | DType::U32 => {
                launch_radix_histogram_u32(client, &src_keys, height, &histograms, num_blocks, grid, shift_u32)?;
                // Device-side prefix sum: no per-pass CUDA sync needed.
                launch_radix_offsets(client, &histograms, &scatter_offsets, num_blocks)?;
                launch_radix_scatter_u32(
                    client, &src_keys, height, &src_indices, &scatter_offsets, &dst_keys,
                    &dst_indices, num_blocks, grid, shift_u32,
                )?;
            }
            DType::I64 | DType::U64 => {
                launch_radix_histogram_u64(client, &src_keys, height, &histograms, num_blocks, grid, shift_u64)?;
                launch_radix_offsets(client, &histograms, &scatter_offsets, num_blocks)?;
                launch_radix_scatter_u64(
                    client, &src_keys, height, &src_indices, &scatter_offsets, &dst_keys,
                    &dst_indices, num_blocks, grid, shift_u64,
                )?;
            }
            _ => unreachable!(),
        }

        std::mem::swap(&mut src_keys, &mut dst_keys);
        std::mem::swap(&mut src_indices, &mut dst_indices);
    }

    // `src_indices` now holds the sorted permutation; recycle the scratch pair.
    recycle_handle(dst_keys, height * elem_size);
    recycle_handle(dst_indices, height * 4);
    recycle_handle(histograms, num_blocks * RADIX_BINS as usize * 4);
    recycle_handle(scatter_offsets, num_blocks * RADIX_BINS as usize * 4);
    Ok(src_indices)
}

/// Download the key validity mask and check whether any entry is null (`0`).
fn key_validity_has_nulls(
    client: &Client,
    key: &DeviceColumn,
    height: usize,
) -> Result<bool, ExecError> {
    let bytes = client
        .read_one(key.validity().clone())
        .map_err(|error| ExecError::Launch(error.to_string()))?;
    Ok(bytes.iter().take(height).any(|&b| b == 0))
}

/// Host fallback: download the key column, compute the permutation on the
/// CPU, and upload the index buffer to the device.
fn gpu_sort_host_fallback(
    client: &Client,
    key: &DeviceColumn,
    descending: bool,
    nulls_last: bool,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    let key_host = download(client, key)?;
    let height = key.len();
    let mut permutation: Vec<u32> = (0..height).map(|i| i as u32).collect();
    sort_permutation(key_host.values(), &key_host.validity(), descending, nulls_last, &mut permutation);
    Ok(client.create_from_slice(bytemuck::cast_slice(&permutation)))
}

/// In-place compute a (stable) permutation that sorts `values` according to
/// the given key column and null policy. `permutation` is mutated into the
/// sorted index vector.
fn sort_permutation(
    values: &polars_cubecl_buffer::HostValues,
    validity: &[u8],
    descending: bool,
    nulls_last: bool,
    permutation: &mut [u32],
) {
    // `Valid(idx)` vs `Null(idx)`: null ordering is decided first (nulls go
    // last when `nulls_last`), then by the natural order of the value,
    // optionally reversed when `descending`. The comparator is stable on
    // equal keys, which is what Polars `sort` promises.
    match values {
        polars_cubecl_buffer::HostValues::I32(v) => {
            permutation.sort_by(|a, b| {
                compare_keys(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::I64(v) => {
            permutation.sort_by(|a, b| {
                compare_keys(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::U32(v) => {
            permutation.sort_by(|a, b| {
                compare_keys(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::U64(v) => {
            permutation.sort_by(|a, b| {
                compare_keys(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::F32(v) => {
            permutation.sort_by(|a, b| {
                compare_keys_f32(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::F64(v) => {
            permutation.sort_by(|a, b| {
                compare_keys_f64(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
        polars_cubecl_buffer::HostValues::Bool(v) => {
            permutation.sort_by(|a, b| {
                compare_keys(v, validity, *a as usize, *b as usize, descending, nulls_last)
            });
        }
    }
}

/// Compare two rows `a` and `b` of a column of totally-ordered values.
fn compare_keys<T: Ord>(
    values: &[T],
    validity: &[u8],
    a: usize,
    b: usize,
    descending: bool,
    nulls_last: bool,
) -> std::cmp::Ordering {
    let va = validity[a] != 0;
    let vb = validity[b] != 0;
    let base = match (va, vb) {
        (true, true) => values[a].cmp(&values[b]),
        (false, false) => std::cmp::Ordering::Equal,
        (false, true) => {
            if nulls_last {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
        (true, false) => {
            if nulls_last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
    };
    if descending { base.reverse() } else { base }
}

/// Float comparison that matches Polars' sort ordering: `NaN` sorts after
/// every other value (and above `+inf`), regardless of the `descending` and
/// `nulls_last` flags.
fn compare_keys_f32(
    values: &[f32],
    validity: &[u8],
    a: usize,
    b: usize,
    descending: bool,
    nulls_last: bool,
) -> std::cmp::Ordering {
    compare_float_keys(values, validity, a, b, descending, nulls_last)
}

fn compare_keys_f64(
    values: &[f64],
    validity: &[u8],
    a: usize,
    b: usize,
    descending: bool,
    nulls_last: bool,
) -> std::cmp::Ordering {
    compare_float_keys(values, validity, a, b, descending, nulls_last)
}

fn compare_float_keys<T: Copy + PartialOrd>(
    values: &[T],
    validity: &[u8],
    a: usize,
    b: usize,
    descending: bool,
    nulls_last: bool,
) -> std::cmp::Ordering
where
    f64: From<T>,
{
    let va = validity[a] != 0;
    let vb = validity[b] != 0;
    let base = match (va, vb) {
        (true, true) => {
            let fa = f64::from(values[a]);
            let fb = f64::from(values[b]);
            float_sort_ord(fa).cmp(&float_sort_ord(fb))
        }
        (false, false) => std::cmp::Ordering::Equal,
        (false, true) => {
            if nulls_last {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
        (true, false) => {
            if nulls_last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
    };
    if descending { base.reverse() } else { base }
}

/// Map an `f64` to a `u64` key that sorts in the same order as Polars'
/// (finite values by sign+magnitude, `-0.0 == +0.0`, `NaN` last).
fn float_sort_ord(value: f64) -> u64 {
    if value.is_nan() {
        return u64::MAX;
    }
    let bits = value.to_bits();
    // Flip the sign bit and, for negatives, the rest of the bits as well
    // (IEEE 754 trick to make the bit pattern sort as unsigned == float order).
    if bits >> 63 == 1 { !bits } else { bits ^ (1 << 63) }
}

fn eval_predicate(
    client: &Client,
    f64_ok: bool,
    predicate: &Predicate,
    columns: &[DeviceColumn],
    height: usize,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    match predicate {
        Predicate::Compare { column, kernel_dtype, op, literal } => {
            if *kernel_dtype == DType::F64 && !f64_ok {
                return Err(ExecError::BackendUnsupported(
                    "f64 comparison needs double support".to_string(),
                ));
            }
            let index = find_column(columns, column)?;
            let source = &columns[index];
            let values = maybe_widen(client, source, *kernel_dtype)?;
            let literal = upload_literal(client, *kernel_dtype, *literal)?;
            let out = alloc(client, height);
            launch_compare(
                client,
                *kernel_dtype,
                &values,
                source.validity(),
                &literal,
                &out,
                height,
                *op,
            )?;
            Ok(out)
        }
        Predicate::ColumnMask { column } => {
            let index = find_column(columns, column)?;
            let source = &columns[index];
            let out = alloc(client, height);
            polars_cubecl_kernels::launch_logic(
                client,
                source.values(),
                source.validity(),
                &out,
                height,
                polars_cubecl_kernels::LogicOp::And,
            )?;
            Ok(out)
        }
        Predicate::And(left, right) => {
            let lhs = eval_predicate(client, f64_ok, left, columns, height)?;
            let rhs = eval_predicate(client, f64_ok, right, columns, height)?;
            let out = alloc(client, height);
            polars_cubecl_kernels::launch_logic(
                client,
                &lhs,
                &rhs,
                &out,
                height,
                polars_cubecl_kernels::LogicOp::And,
            )?;
            Ok(out)
        }
        Predicate::Or(left, right) => {
            let lhs = eval_predicate(client, f64_ok, left, columns, height)?;
            let rhs = eval_predicate(client, f64_ok, right, columns, height)?;
            let out = alloc(client, height);
            polars_cubecl_kernels::launch_logic(
                client,
                &lhs,
                &rhs,
                &out,
                height,
                polars_cubecl_kernels::LogicOp::Or,
            )?;
            Ok(out)
        }
        Predicate::NotColumn { column } => {
            let index = find_column(columns, column)?;
            let source = &columns[index];
            let negated = alloc(client, height);
            polars_cubecl_kernels::launch_not(client, source.values(), &negated, height)?;
            let out = alloc(client, height);
            polars_cubecl_kernels::launch_logic(
                client,
                &negated,
                source.validity(),
                &out,
                height,
                polars_cubecl_kernels::LogicOp::And,
            )?;
            Ok(out)
        }
    }
}

/// Widen an `Int32`/`UInt32` column to `f64` when the comparison kernel runs
/// in `Float64`; otherwise reuse the resident values handle.
fn maybe_widen(
    client: &Client,
    column: &DeviceColumn,
    kernel_dtype: DType,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    if column.dtype() == kernel_dtype {
        return Ok(column.values().clone());
    }
    match (column.dtype(), kernel_dtype) {
        (DType::I32, DType::F64) => {
            let out = alloc(client, column.len() * 8);
            launch_cast_i32_to_f64(client, column.values(), &out, column.len())?;
            Ok(out)
        }
        (DType::U32, DType::F64) => {
            let out = alloc(client, column.len() * 8);
            launch_cast_u32_to_f64(client, column.values(), &out, column.len())?;
            Ok(out)
        }
        _ => Err(ExecError::Inconsistent("unsupported comparison widening".to_string())),
    }
}

fn upload_literal(
    client: &Client,
    dtype: DType,
    literal: ScalarLit,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    let mut buf = [0_u8; 8];
    let len = match (dtype, literal) {
        (DType::I32, ScalarLit::I32(value)) => {
            buf[..4].copy_from_slice(&value.to_le_bytes());
            4
        }
        (DType::I64, ScalarLit::I64(value)) => {
            buf.copy_from_slice(&value.to_le_bytes());
            8
        }
        (DType::U32, ScalarLit::U32(value)) => {
            buf[..4].copy_from_slice(&value.to_le_bytes());
            4
        }
        (DType::U64, ScalarLit::U64(value)) => {
            buf.copy_from_slice(&value.to_le_bytes());
            8
        }
        (DType::F32, ScalarLit::F32(value)) => {
            buf[..4].copy_from_slice(&value.to_le_bytes());
            4
        }
        (DType::F64, ScalarLit::F64(value)) => {
            buf.copy_from_slice(&value.to_le_bytes());
            8
        }
        (DType::Bool, ScalarLit::Bool(value)) => {
            buf[0] = u8::from(value);
            1
        }
        _ => {
            return Err(ExecError::Inconsistent(
                "literal dtype does not match comparison dtype".to_string(),
            ));
        }
    };
    Ok(client.create_from_slice(&buf[..len]))
}

fn launch_compare(
    client: &Client,
    dtype: DType,
    values: &cubecl_runtime::server::Handle,
    valid: &cubecl_runtime::server::Handle,
    lit: &cubecl_runtime::server::Handle,
    out: &cubecl_runtime::server::Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), ExecError> {
    let op = map_cmp_op(op);
    match dtype {
        DType::I32 => {
            polars_cubecl_kernels::launch_cmp::<i32>(client, values, valid, lit, out, height, op)?;
        }
        DType::I64 => {
            polars_cubecl_kernels::launch_cmp::<i64>(client, values, valid, lit, out, height, op)?;
        }
        DType::U32 => {
            polars_cubecl_kernels::launch_cmp::<u32>(client, values, valid, lit, out, height, op)?;
        }
        DType::U64 => {
            polars_cubecl_kernels::launch_cmp::<u64>(client, values, valid, lit, out, height, op)?;
        }
        DType::F32 => {
            polars_cubecl_kernels::launch_cmp_f32(client, values, valid, lit, out, height, op)?;
        }
        DType::F64 => {
            polars_cubecl_kernels::launch_cmp_f64(client, values, valid, lit, out, height, op)?;
        }
        DType::Bool => {
            polars_cubecl_kernels::launch_cmp::<u8>(client, values, valid, lit, out, height, op)?;
        }
    }
    Ok(())
}

const fn map_cmp_op(op: CmpOp) -> polars_cubecl_kernels::CmpOp {
    match op {
        CmpOp::Eq => polars_cubecl_kernels::CmpOp::Eq,
        CmpOp::NotEq => polars_cubecl_kernels::CmpOp::NotEq,
        CmpOp::Lt => polars_cubecl_kernels::CmpOp::Lt,
        CmpOp::LtEq => polars_cubecl_kernels::CmpOp::LtEq,
        CmpOp::Gt => polars_cubecl_kernels::CmpOp::Gt,
        CmpOp::GtEq => polars_cubecl_kernels::CmpOp::GtEq,
    }
}

/// Stable compaction of `mask` into sorted row indices plus the kept count.
///
/// Threads own disjoint contiguous row chunks; the host prefix-sums the
/// per-thread tallies into scatter bases. No cross-thread communication
/// happens on the device, so the result is deterministic on every backend.
fn compact_indices(
    client: &Client,
    mask: &cubecl_runtime::server::Handle,
    height: usize,
) -> Result<(cubecl_runtime::server::Handle, usize), ExecError> {
    use cubecl::prelude::CubeElement;
    let threads = polars_cubecl_kernels::filter_threads(height);
    let zeros = vec![0_u32; threads];
    let counts = client.create_from_slice(bytemuck::cast_slice(&zeros));
    polars_cubecl_kernels::launch_count(client, mask, &counts, height, threads)?;
    let counts_bytes =
        client.read_one(counts).map_err(|error| ExecError::Launch(error.to_string()))?;
    let counts = u32::from_bytes(&counts_bytes);
    let mut bases = Vec::with_capacity(threads);
    let mut total: u32 = 0;
    for count in counts.iter().take(threads) {
        bases.push(total);
        total = total.saturating_add(*count);
    }
    let kept = total as usize;
    let bases_handle = client.create_from_slice(bytemuck::cast_slice(&bases));
    let indices = alloc(client, kept.saturating_mul(4));
    polars_cubecl_kernels::launch_scatter(
        client,
        mask,
        &bases_handle,
        &indices,
        height,
        threads,
        kept,
    )?;
    Ok((indices, kept))
}

/// Fused gather of both values and validity mask through `indices` in a
/// single kernel launch. Replaces the separate `gather_column` + `gather_mask`
/// calls used previously, halving kernel-launch overhead per column.
fn gather_values_mask(
    client: &Client,
    column: &DeviceColumn,
    indices: &cubecl_runtime::server::Handle,
    kept: usize,
) -> Result<(cubecl_runtime::server::Handle, cubecl_runtime::server::Handle), ExecError> {
    if kept == 0 {
        return Ok((alloc(client, 0), alloc(client, 0)));
    }
    let out_values = alloc(client, kept.saturating_mul(dtype_size(column.dtype())));
    let out_validity = alloc(client, kept);
    match column.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_gather_values_mask::<i32>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_gather_values_mask::<i64>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_gather_values_mask::<u32>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
        DType::U64 | DType::F64 => polars_cubecl_kernels::launch_gather_values_mask::<u64>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_gather_values_mask::<f32>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
        DType::Bool => polars_cubecl_kernels::launch_gather_values_mask::<u8>(
            client,
            column.values(),
            column.len(),
            column.validity(),
            indices,
            &out_values,
            &out_validity,
            kept,
        )?,
    }
    Ok((out_values, out_validity))
}

fn slice_column(
    client: &Client,
    column: &DeviceColumn,
    start: usize,
    count: usize,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    if count == 0 {
        return Ok(alloc(client, 0));
    }
    let out = alloc(client, count.saturating_mul(dtype_size(column.dtype())));
    match column.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_slice::<i32>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_slice::<i64>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_slice::<u32>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
        // `F64` shares the `u64` kernel: see `gather_column`.
        DType::U64 | DType::F64 => polars_cubecl_kernels::launch_slice::<u64>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_slice::<f32>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
        DType::Bool => polars_cubecl_kernels::launch_slice::<u8>(
            client,
            column.values(),
            column.len(),
            &out,
            start,
            count,
        )?,
    }
    Ok(out)
}

fn slice_mask(
    client: &Client,
    handle: &cubecl_runtime::server::Handle,
    source_len: usize,
    start: usize,
    count: usize,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    if count == 0 {
        return Ok(alloc(client, 0));
    }
    let out = alloc(client, count);
    polars_cubecl_kernels::launch_slice::<u8>(client, handle, source_len, &out, start, count)?;
    Ok(out)
}
