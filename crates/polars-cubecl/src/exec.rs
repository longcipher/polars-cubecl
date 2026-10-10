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
    BufferError, DType, DeviceColumn, download_frame, series_to_host, upload,
};
use polars_cubecl_kernels::{KernelError, launch_cast_i32_to_f64, launch_cast_u32_to_f64};
use polars_cubecl_lower::{CmpOp, CubePlan, PlanOp, PlanSource, Predicate, ScalarLit};
use polars_cubecl_runtime::{RuntimeContext, supports_f64};
use polars_utils::pl_str::PlSmallStr;
use thiserror::Error;

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
    let mut columns = ingest(client, plan.source())?;
    let mut ops = Vec::with_capacity(plan.ops().len());
    for op in plan.ops() {
        match op {
            PlanOp::Filter(predicate) => {
                columns = apply_filter(client, f64_ok, columns, predicate)?;
                ops.push(crate::Op::Filter);
            }
            PlanOp::Project(projection) => {
                columns = apply_project(columns, projection)?;
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
        }
    }
    let frame = download_frame(client, &columns)?;
    Ok((frame, ops))
}

fn ingest(client: &Client, source: &PlanSource) -> Result<Vec<DeviceColumn>, ExecError> {
    match source {
        PlanSource::DataFrame { frame, columns } => ingest_frame(client, frame, columns),
        PlanSource::Parquet { path, columns, .. } => {
            let frame = read_parquet_file(path)?;
            ingest_frame(client, &frame, columns)
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
pub(crate) fn alloc(client: &Client, bytes: usize) -> cubecl_runtime::server::Handle {
    client.empty(bytes.max(1))
}

/// Upload bytes, padding empties: some backends refuse zero-byte buffers.
/// Lengths are tracked separately everywhere, so the padding is inert.
pub(crate) fn create_slice(client: &Client, bytes: &[u8]) -> cubecl_runtime::server::Handle {
    client.create_from_slice(if bytes.is_empty() { &[0] } else { bytes })
}

fn apply_project(
    columns: Vec<DeviceColumn>,
    projection: &[(PlSmallStr, PlSmallStr)],
) -> Result<Vec<DeviceColumn>, ExecError> {
    let mut output = Vec::with_capacity(projection.len());
    for (output_name, input_name) in projection {
        let index = find_column(&columns, input_name)?;
        let source = &columns[index];
        output.push(DeviceColumn::from_handles(
            output_name.clone(),
            source.dtype(),
            source.values().clone(),
            source.validity().clone(),
            source.len(),
        ));
    }
    Ok(output)
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
        let values = gather_column(client, column, &indices, kept)?;
        let validity = gather_mask(client, column.validity(), column.len(), &indices, kept)?;
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

fn gather_column(
    client: &Client,
    column: &DeviceColumn,
    indices: &cubecl_runtime::server::Handle,
    kept: usize,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    if kept == 0 {
        return Ok(alloc(client, 0));
    }
    let out = alloc(client, kept.saturating_mul(dtype_size(column.dtype())));
    match column.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_gather::<i32>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_gather::<i64>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_gather::<u32>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
        // `F64` shares the `u64` kernel: movement is bitwise, so no
        // arithmetic (and therefore no MSL `double`) is required.
        DType::U64 | DType::F64 => polars_cubecl_kernels::launch_gather::<u64>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_gather::<f32>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
        DType::Bool => polars_cubecl_kernels::launch_gather::<u8>(
            client,
            column.values(),
            column.len(),
            indices,
            &out,
            kept,
        )?,
    }
    Ok(out)
}

fn gather_mask(
    client: &Client,
    handle: &cubecl_runtime::server::Handle,
    source_len: usize,
    indices: &cubecl_runtime::server::Handle,
    kept: usize,
) -> Result<cubecl_runtime::server::Handle, ExecError> {
    if kept == 0 {
        return Ok(alloc(client, 0));
    }
    let out = alloc(client, kept);
    polars_cubecl_kernels::launch_gather::<u8>(client, handle, source_len, indices, &out, kept)?;
    Ok(out)
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

pub(crate) const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
        DType::Bool => 1,
    }
}
