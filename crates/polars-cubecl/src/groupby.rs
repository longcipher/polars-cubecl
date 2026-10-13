//! Grouped aggregation: host-side grouping, device-side reductions.
//!
//! Group ids are assigned on the host in first-seen order with an exact
//! hash map over normalized key tuples (float `NaN`s canonicalized, `-0.0`
//! fused into `+0.0`, nulls hashing as their own group). The device then
//! inverts the row-to-group mapping into contiguous segments and reduces
//! each segment with one serial thread. Device code uses only `u32` atomics
//! and independent per-thread loops: no barriers, no shared memory, no float
//! atomics, so it is deterministic and portable across backends.
//!
//! Integer sums are exact regardless of slot order; float sums may vary in
//! the last ulps with slot order, inside the comparator tolerance.

use cubecl::prelude::Client;
use cubecl_runtime::server::Handle;
use polars_cubecl_buffer::{
    DType, DeviceColumn, HostColumn, HostValues, download, download_columns,
};
use polars_cubecl_lower::Aggregation;
use polars_utils::pl_str::PlSmallStr;
use rustc_hash::FxHashMap;

use crate::exec::{ExecError, alloc, create_slice, dtype_size, find_column, recycle_handle};

/// A normalized group-key component: `None` is a null key part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeyVal {
    /// Nullable `i32`.
    I32(Option<i32>),
    /// Nullable `i64`.
    I64(Option<i64>),
    /// Nullable `u32`.
    U32(Option<u32>),
    /// Nullable `u64`.
    U64(Option<u64>),
    /// Nullable `f32` as normalized bits.
    F32(Option<u32>),
    /// Nullable `f64` as normalized bits.
    F64(Option<u64>),
    /// Nullable boolean as `u8`.
    Bool(Option<u8>),
}

impl KeyVal {
    fn of(dtype: DType, values: &HostValues, validity: &[u8], row: usize) -> Self {
        if validity[row] == 0 {
            return Self::null(dtype);
        }
        match (dtype, values) {
            (DType::I32, HostValues::I32(found)) => Self::I32(Some(found[row])),
            (DType::I64, HostValues::I64(found)) => Self::I64(Some(found[row])),
            (DType::U32, HostValues::U32(found)) => Self::U32(Some(found[row])),
            (DType::U64, HostValues::U64(found)) => Self::U64(Some(found[row])),
            (DType::F32, HostValues::F32(found)) => Self::F32(Some(normalize_f32(found[row]))),
            (DType::F64, HostValues::F64(found)) => Self::F64(Some(normalize_f64(found[row]))),
            (DType::Bool, HostValues::Bool(found)) => Self::Bool(Some(found[row])),
            _ => Self::null(dtype),
        }
    }

    const fn null(dtype: DType) -> Self {
        match dtype {
            DType::I32 => Self::I32(None),
            DType::I64 => Self::I64(None),
            DType::U32 => Self::U32(None),
            DType::U64 => Self::U64(None),
            DType::F32 => Self::F32(None),
            DType::F64 => Self::F64(None),
            DType::Bool => Self::Bool(None),
        }
    }
}

const fn normalize_f32(value: f32) -> u32 {
    if value.is_nan() {
        0x7FC0_0000
    } else if value == 0.0 {
        0x0000_0000
    } else {
        value.to_bits()
    }
}

const fn normalize_f64(value: f64) -> u64 {
    if value.is_nan() {
        0x7FF8_0000_0000_0000
    } else if value == 0.0 {
        0x0000_0000_0000_0000
    } else {
        value.to_bits()
    }
}

/// Grouping of `height` rows: dense first-seen ids, per-group sizes, and the
/// row-order first/last row of every group.
///
/// When `gpu_assignment` is `Some`, the caller skips the host-side
/// download and re-upload of the assignment vector. `assignment` must be
/// empty in that case.
struct Grouping {
    height: usize,
    assignment: Vec<u32>,
    gpu_assignment: Option<Handle>,
    sizes: Vec<u32>,
    first_rows: Vec<u32>,
    last_rows: Vec<u32>,
}

fn group_rows(key_columns: &[HostColumn]) -> Grouping {
    let height = key_columns.first().map_or(0, HostColumn::len);
    if key_columns.len() == 1 {
        let col = &key_columns[0];
        let validity = col.validity();
        // Fast path: dense-range integer keys with no nulls → direct array
        // indexing. Avoids the FxHashMap of the general path, which is the
        // dominant CPU cost of group-by for small-range keys.
        if height > 0 && validity.iter().take(height).all(|&v| v != 0) {
            match col.dtype() {
                DType::I32 => {
                    let HostValues::I32(values) = col.values() else { unreachable!() };
                    let (min, max) = values
                        .iter()
                        .take(height)
                        .fold((i32::MAX, i32::MIN), |(min, max), &v| (min.min(v), max.max(v)));
                    let range = (max as i64 - min as i64 + 1).max(0) as usize;
                    if range > 0 && range <= 65536 {
                        return group_single_dense_i32(values, height, min, range);
                    }
                }
                DType::I64 => {
                    let HostValues::I64(i64_values) = col.values() else { unreachable!() };
                    let (min, max) = i64_values
                        .iter()
                        .take(height)
                        .fold((i64::MAX, i64::MIN), |(min, max), &v| (min.min(v), max.max(v)));
                    let range = (max - min + 1).max(0) as usize;
                    if range > 0 && range <= 65536 {
                        return group_single_dense_i64(i64_values, height, min, range);
                    }
                }
                _ => {}
            }
        }
        group_single(col, height)
    } else {
        group_multi(key_columns, height)
    }
}

/// Dense direct-indexing grouping for non-null `i32` keys whose value range
/// fits in a small array. Avoids the `FxHashMap` of the general path.
fn group_single_dense_i32(values: &[i32], height: usize, min: i32, range: usize) -> Grouping {
    // Map raw key → dense group id via direct indexing.
    let mut dense_id: Vec<Option<u32>> = vec![None; range];
    let mut assignment = Vec::with_capacity(height);
    let mut sizes = Vec::new();
    let mut first_rows = Vec::new();
    let mut last_rows = Vec::new();
    let min = min as i64;
    for row in 0..height {
        let bucket = (values[row] as i64 - min) as usize;
        let next = sizes.len() as u32;
        let id = match &mut dense_id[bucket] {
            Some(id) => *id,
            slot => {
                *slot = Some(next);
                sizes.push(0);
                first_rows.push(row as u32);
                last_rows.push(row as u32);
                next
            }
        };
        sizes[id as usize] += 1;
        last_rows[id as usize] = row as u32;
        assignment.push(id);
    }
    Grouping { height, assignment, gpu_assignment: None, sizes, first_rows, last_rows }
}

/// Dense direct-indexing grouping for non-null `i64` keys; same approach as
/// `group_single_dense_i32` but with `i64` keys.
fn group_single_dense_i64(values: &[i64], height: usize, min: i64, range: usize) -> Grouping {
    let mut dense_id: Vec<Option<u32>> = vec![None; range];
    let mut assignment = Vec::with_capacity(height);
    let mut sizes = Vec::new();
    let mut first_rows = Vec::new();
    let mut last_rows = Vec::new();
    for row in 0..height {
        let bucket = (values[row] - min) as usize;
        let next = sizes.len() as u32;
        let id = match &mut dense_id[bucket] {
            Some(id) => *id,
            slot => {
                *slot = Some(next);
                sizes.push(0);
                first_rows.push(row as u32);
                last_rows.push(row as u32);
                next
            }
        };
        sizes[id as usize] += 1;
        last_rows[id as usize] = row as u32;
        assignment.push(id);
    }
    Grouping { height, assignment, gpu_assignment: None, sizes, first_rows, last_rows }
}

fn group_single(column: &HostColumn, height: usize) -> Grouping {
    let mut ids: FxHashMap<KeyVal, u32> = FxHashMap::default();
    let mut assignment = Vec::with_capacity(height);
    let mut sizes = Vec::new();
    let mut first_rows = Vec::new();
    let mut last_rows = Vec::new();
    for row in 0..height {
        let key = KeyVal::of(column.dtype(), column.values(), column.validity(), row);
        let next = sizes.len() as u32;
        let id = *ids.entry(key).or_insert(next);
        if (id as usize) == sizes.len() {
            sizes.push(0);
            first_rows.push(row as u32);
            last_rows.push(row as u32);
        }
        sizes[id as usize] += 1;
        last_rows[id as usize] = row as u32;
        assignment.push(id);
    }
    Grouping { height, assignment, gpu_assignment: None, sizes, first_rows, last_rows }
}

fn group_multi(key_columns: &[HostColumn], height: usize) -> Grouping {
    let mut ids: FxHashMap<Vec<KeyVal>, u32> = FxHashMap::default();
    let mut assignment = Vec::with_capacity(height);
    let mut sizes = Vec::new();
    let mut first_rows = Vec::new();
    let mut last_rows = Vec::new();
    let mut tuple = Vec::with_capacity(key_columns.len());
    for row in 0..height {
        tuple.clear();
        for column in key_columns {
            tuple.push(KeyVal::of(column.dtype(), column.values(), column.validity(), row));
        }
        let next = sizes.len() as u32;
        let id = *ids.entry(tuple.clone()).or_insert(next);
        if (id as usize) == sizes.len() {
            sizes.push(0);
            first_rows.push(row as u32);
            last_rows.push(row as u32);
        }
        sizes[id as usize] += 1;
        last_rows[id as usize] = row as u32;
        assignment.push(id);
    }
    Grouping { height, assignment, gpu_assignment: None, sizes, first_rows, last_rows }
}

/// Maximum key range for which the GPU histogram path is used. Beyond this,
/// the per-bucket `Vec<Atomic<u32>>` is too large to alloc + download.
const GPU_HISTOGRAM_MAX_BUCKETS: usize = 65536;

/// Try to group small-range `i32` keys without downloading the key column.
///
/// Returns `Ok(None)` if the key has nulls or the range is too wide (fall back
/// to the host path). Returns `Ok(Some(grouping))` on success.
///
/// Pipeline (all on GPU until step 5):
/// 1. Download key validity, check for nulls (tiny readback).
/// 2. `launch_reduce_min::<i32>` with one group → global min.
/// 3. `launch_reduce_max::<i32>` with one group → global max.
/// 4. Read back min+max (8 bytes, one sync).
/// 5. Launch GPU histogram; read back counts (`num_buckets * 4` bytes).
/// 6. Build Grouping from counts + a GPU kernel that computes per-row `assignment[row] = (keys[row]
///    - min_val) as u32` on the device.
fn try_gpu_histogram_grouping(
    client: &Client,
    key_col: &DeviceColumn,
    height: usize,
) -> Result<Option<Grouping>, ExecError> {
    // 1. Check nulls in the validity mask.
    let valid_bytes = client
        .read_one(key_col.validity().clone())
        .map_err(|error| ExecError::Launch(error.to_string()))?;
    if valid_bytes.iter().take(height).any(|&b| b == 0) {
        return Ok(None); // nulls: fall back to host path
    }

    // 2–3. Fused global min+max in a single launch (CubeK-style kernel fusion).
    //    Direct mode: reads values[k] directly in [0, height) — no slot
    //    indirection needed for a contiguous segment. Also avoids the
    //    buffer-pool uninit-read hazard from stale slot data.
    //    One kernel launch instead of two — eliminates ~500µs CUDA startup.

    let min_val_handle = alloc(client, 4);
    let max_val_handle = alloc(client, 4);
    let minmax_valid_handle = alloc(client, 1);
    polars_cubecl_kernels::launch_reduce_min_max_direct::<i32>(
        client,
        key_col.values(),
        key_col.validity(),
        height,
        0,
        height as u32,
        &min_val_handle,
        &max_val_handle,
        &minmax_valid_handle,
    )?;

    // 4. Read back min + max in one batched sync (was 2 separate read_one calls).
    let batch = client
        .read(vec![min_val_handle, max_val_handle])
        .map_err(|e| ExecError::Launch(e.to_string()))?;
    let min_val = bytemuck::cast_slice::<u8, i32>(&batch[0][..4])[0];
    let max_val = bytemuck::cast_slice::<u8, i32>(&batch[1][..4])[0];

    let range = (max_val as i64 - min_val as i64 + 1).max(0) as usize;
    if range == 0 || range > GPU_HISTOGRAM_MAX_BUCKETS {
        return Ok(None);
    }

    // 5–6. Fused GPU histogram + group-id assignment in a single launch
    //    (CubeK-style kernel fusion). Computes bucket counts and per-row
    //    assignment with one read of keys, saving ~500µs CUDA launch overhead
    //    vs separate kernels. The `out[row] = bucket` writes are
    //    deterministic (each thread writes its own index); first/last rows
    //    are computed on the CPU from the read-back assignment, avoiding
    //    non-determinism from atomic counters in group_invert.
    let num_buckets = range;
    let min_scalar: Handle = client.create_from_slice(bytemuck::cast_slice(&[min_val]));
    let histo_counts = alloc(client, num_buckets.saturating_mul(4));
    let assignment_handle = alloc(client, height.saturating_mul(4));
    polars_cubecl_kernels::launch_gpu_histogram_and_assign(
        client,
        key_col.values(),
        &min_scalar,
        &histo_counts,
        &assignment_handle,
        height,
        num_buckets,
    )?;
    // Read back counts + assignment in one batched sync (was 2 separate reads).
    let batch = client
        .read(vec![histo_counts, assignment_handle.clone()])
        .map_err(|e| ExecError::Launch(e.to_string()))?;
    let counts: &[u32] = bytemuck::cast_slice(&batch[0]);
    let counts: Vec<u32> = counts.iter().take(num_buckets).copied().collect();

    // `out[row] = bucket` is written by exactly one thread per row, so the
    // result is reproducible across CPU and GPU backends.
    let assign_raw: &[u32] = bytemuck::cast_slice(&batch[1]);

    // Build Grouping sizes, first_rows, last_rows on the CPU from the
    // deterministic assignment. The slots-based path in finish_grouping
    // uses bucket id directly (= dense group id from the histogram), so we
    // must use the same mapping here: per-bucket first/last, then filter to
    // non-empty buckets in bucket id order.
    let mut raw_first_rows = vec![u32::MAX; num_buckets];
    let mut raw_last_rows = vec![u32::MAX; num_buckets];
    for row in 0..height {
        let bucket = assign_raw[row] as usize;
        if raw_first_rows[bucket] == u32::MAX {
            raw_first_rows[bucket] = row as u32;
        }
        raw_last_rows[bucket] = row as u32;
    }
    let sizes: Vec<u32> = counts.iter().copied().filter(|&c| c > 0).collect();
    let num_groups = sizes.len();
    let mut first_rows = Vec::with_capacity(num_groups);
    let mut last_rows = Vec::with_capacity(num_groups);
    for bucket in 0..num_buckets {
        if counts[bucket] > 0 {
            first_rows.push(raw_first_rows[bucket]);
            last_rows.push(raw_last_rows[bucket]);
        }
    }

    // assignment_handle stays on the GPU — finish_grouping reads it
    // directly, avoiding a redundant upload.

    // Recycle scratch handles.
    recycle_handle(min_scalar, 4);
    recycle_handle(minmax_valid_handle, 1);

    Ok(Some(Grouping {
        height,
        assignment: Vec::new(),
        gpu_assignment: Some(assignment_handle),
        sizes,
        first_rows,
        last_rows,
    }))
}

/// Apply one `GroupBy` pipeline step.
///
/// `f64_ok` reports whether this backend compiles `f64` arithmetic: movement
/// over `f64` data always runs (via bitwise `u64` kernels), but reductions
/// that need true `double` (`sum/min/max/mean` over `f64`, `mean` over
/// anything) return [`ExecError::BackendUnsupported`] when it is false, so
/// the caller falls back to the CPU engine.
pub fn apply(
    client: &Client,
    f64_ok: bool,
    columns: Vec<DeviceColumn>,
    keys: &[(PlSmallStr, PlSmallStr)],
    aggs: &[Aggregation],
) -> Result<Vec<DeviceColumn>, ExecError> {
    let height = columns.first().map_or(0, DeviceColumn::len);
    let key_device_cols: Vec<DeviceColumn> = keys
        .iter()
        .map(|(_, input)| -> Result<DeviceColumn, ExecError> {
            Ok(columns[find_column(&columns, input)?].clone())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // GPU histogram fast path: for single-column i32 keys with small range,
    // avoid downloading the key column entirely. Compute min/max + histogram
    // on the GPU; only counts (num_buckets * 4 bytes) round-trip to the host.
    if keys.len() == 1 {
        let key_col = &key_device_cols[0];
        if key_col.dtype() == DType::I32 && height > 0 {
            if let Some(grouping) = try_gpu_histogram_grouping(client, key_col, height)? {
                return finish_grouping(client, f64_ok, columns, keys, aggs, grouping);
            }
        }
    }

    // Fallback: batch-download all key columns to host for grouping.
    let key_hosts = download_columns(client, &key_device_cols)?;
    let grouping = group_rows(&key_hosts);
    finish_grouping(client, f64_ok, columns, keys, aggs, grouping)
}

/// Shared tail path: given a `Grouping`, perform the invert + reductions.
///
/// When `grouping.gpu_assignment` is present (GPU histogram fast path),
/// group-by-sum uses a direct atomic-add kernel that bypasses the
/// group_invert + slots indirection — saving ~500µs CUDA launch overhead
/// and one slots buffer allocation.
fn finish_grouping(
    client: &Client,
    f64_ok: bool,
    columns: Vec<DeviceColumn>,
    keys: &[(PlSmallStr, PlSmallStr)],
    aggs: &[Aggregation],
    mut grouping: Grouping,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let num_groups = grouping.sizes.len();
    let height = grouping.height;

    // Use the GPU-resident assignment handle directly when available;
    // otherwise upload the host-side Vec<u32> to the device.
    let assignment = match grouping.gpu_assignment {
        Some(ref handle) => handle.clone(),
        None => create_slice(client, bytemuck::cast_slice(&grouping.assignment)),
    };
    let mut seg_off = Vec::with_capacity(num_groups);
    let mut edge = 0_u32;
    for size in &grouping.sizes {
        seg_off.push(edge);
        edge = edge.saturating_add(*size);
    }
    let seg_off_handle = create_slice(client, bytemuck::cast_slice(&seg_off));
    let seg_len_handle = create_slice(client, bytemuck::cast_slice(&grouping.sizes));

    // If every agg is Sum, we can skip group_invert entirely and use the
    // direct-from-assignment atomic-add kernel (~500µs saved vs slot path).
    // Otherwise, fall back to the traditional group_invert + slot-based path.
    let all_sum = aggs.iter().all(|a| matches!(a, Aggregation::Sum { .. }));

    if grouping.gpu_assignment.is_some() && all_sum {
        // GPU histogram fast path, all-sum: skip group_invert entirely.
        // Use direct atomic-add from the assignment vector — saves ~500µs
        // CUDA launch overhead (group_invert) + one slots buffer.
        let mut output = Vec::with_capacity(keys.len() + aggs.len());
        output.extend(materialize_keys(client, &columns, keys, &grouping.first_rows, num_groups)?);
        for agg in aggs {
            let Aggregation::Sum { input, output: name } = agg else {
                unreachable!("all agg are Sum by check above");
            };
            let col_index = find_column(&columns, input)?;
            let source = &columns[col_index];
            let dtype = source.dtype();
            // Allocate + zero-fill output buffers in one step. The buffers
            // are used as Atomic<T> / Atomic<u8> accumulators by the kernel.
            macro_rules! do_sum {
                ($t:ty) => {{
                    let num_val_bytes = num_groups.saturating_mul(dtype_size(dtype));
                    let val_buf = vec![0u8; num_val_bytes];
                    let vld_buf = vec![0u8; num_groups];
                    let out_values = create_slice(client, &val_buf);
                    let out_valid = create_slice(client, &vld_buf);
                    polars_cubecl_kernels::launch_reduce_sum_direct::<$t>(
                        client,
                        source.values(),
                        source.validity(),
                        &assignment,
                        &out_values,
                        &out_valid,
                        height,
                        num_groups,
                    )?;
                    (out_values, out_valid)
                }};
            }
            let (out_values, out_valid) = match dtype {
                DType::I32 => do_sum!(i32),
                DType::I64 => do_sum!(i64),
                DType::U32 => do_sum!(u32),
                DType::U64 => do_sum!(u64),
                DType::F32 => do_sum!(f32),
                DType::F64 => do_sum!(f64),
                DType::Bool => do_sum!(u8),
                _ => {
                    return Err(ExecError::NotExecutable(format!("direct sum for dtype {dtype:?}")))
                }
            };
            output.push(DeviceColumn::from_handles(
                name.clone(),
                dtype,
                out_values,
                out_valid,
                num_groups,
            ));
        }
        return Ok(output);
    }

    let zeros = vec![0_u32; num_groups];
    let seq = create_slice(client, bytemuck::cast_slice(&zeros));
    let slots = alloc(client, height.saturating_mul(4));
    polars_cubecl_kernels::launch_group_invert(
        client,
        &assignment,
        &seg_off_handle,
        &seq,
        &slots,
        height,
        num_groups,
    )?;

    // For the GPU histogram path, first_rows and last_rows were not
    // computed upfront. Derive both now from the slots array: within each
    // group, group_invert preserves the original row order, so the first/
    // last slots correspond to the first/last group rows.
    if num_groups > 0 && grouping.first_rows.is_empty() && grouping.last_rows.is_empty() {
        let slots_bytes =
            client.read_one(slots.clone()).map_err(|error| ExecError::Launch(error.to_string()))?;
        let slots_raw: &[u32] = bytemuck::cast_slice(&slots_bytes);
        let mut first_rows = Vec::with_capacity(num_groups);
        let mut last_rows = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let first_slot_pos = seg_off[g] as usize;
            let last_slot_pos = (seg_off[g] + grouping.sizes[g] - 1) as usize;
            first_rows.push(slots_raw[first_slot_pos]);
            last_rows.push(slots_raw[last_slot_pos]);
        }
        grouping.first_rows = first_rows;
        grouping.last_rows = last_rows;
    }

    // Defensive: if for any reason we still have empty first/last rows
    // (e.g. host-path grouping returned empty), populate with null values
    // so downstream gather kernels do not index out of bounds.
    if grouping.first_rows.is_empty() && num_groups > 0 {
        grouping.first_rows = vec![0; num_groups];
        grouping.last_rows = vec![0; num_groups];
    }

    let mut output = Vec::with_capacity(keys.len() + aggs.len());
    output.extend(materialize_keys(client, &columns, keys, &grouping.first_rows, num_groups)?);
    for agg in aggs {
        output.push(reduce_agg(client, f64_ok, &columns, agg, &grouping, &Segmentation {
            slots: &slots,
            seg_off: &seg_off_handle,
            seg_len: &seg_len_handle,
            num_groups,
        })?);
    }
    Ok(output)
}

/// Every group's keys come from its row-order first row (all rows in a group
/// share the key tuple, so any row would do; the first is deterministic).
fn materialize_keys(
    client: &Client,
    columns: &[DeviceColumn],
    keys: &[(PlSmallStr, PlSmallStr)],
    first_rows: &[u32],
    num_groups: usize,
) -> Result<Vec<DeviceColumn>, ExecError> {
    let picked = create_slice(client, bytemuck::cast_slice(first_rows));
    let mut output = Vec::with_capacity(keys.len());
    for (output_name, input_name) in keys {
        let index = find_column(columns, input_name)?;
        let source = &columns[index];
        let values = alloc(client, num_groups.saturating_mul(dtype_size(source.dtype())));
        let validity = alloc(client, num_groups);
        gather_picked(client, source, &picked, &values, &validity, num_groups)?;
        output.push(DeviceColumn::from_handles(
            output_name.clone(),
            source.dtype(),
            values,
            validity,
            num_groups,
        ));
    }
    Ok(output)
}

fn gather_picked(
    client: &Client,
    source: &DeviceColumn,
    picked: &Handle,
    values: &Handle,
    validity: &Handle,
    num_groups: usize,
) -> Result<(), ExecError> {
    if num_groups == 0 {
        return Ok(());
    }
    match source.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_gather::<i32>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
        DType::I64 => polars_cubecl_kernels::launch_gather::<i64>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
        DType::U32 => polars_cubecl_kernels::launch_gather::<u32>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
        // `F64` shares the `u64` kernel: movement is bitwise, so no
        // arithmetic (and therefore no MSL `double`) is required. See
        // `gather_column` in `exec.rs`.
        DType::U64 | DType::F64 => polars_cubecl_kernels::launch_gather::<u64>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
        DType::F32 => polars_cubecl_kernels::launch_gather::<f32>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
        DType::Bool => polars_cubecl_kernels::launch_gather::<u8>(
            client,
            source.values(),
            source.len(),
            picked,
            values,
            num_groups,
        )?,
    }
    polars_cubecl_kernels::launch_gather::<u8>(
        client,
        source.validity(),
        source.len(),
        picked,
        validity,
        num_groups,
    )?;
    Ok(())
}

/// Group segmentation shared by every reduction in one group-by step: the
/// slot permutation built by `launch_group_invert`, plus per-group offsets,
/// lengths, and the group count.
struct Segmentation<'a> {
    /// Row slots, grouped by key.
    slots: &'a Handle,
    /// Per-group start offset into `slots`.
    seg_off: &'a Handle,
    /// Per-group length in slots.
    seg_len: &'a Handle,
    /// Number of groups.
    num_groups: usize,
}

fn reduce_agg(
    client: &Client,
    f64_ok: bool,
    columns: &[DeviceColumn],
    agg: &Aggregation,
    grouping: &Grouping,
    seg: &Segmentation<'_>,
) -> Result<DeviceColumn, ExecError> {
    // `f64` arithmetic (and `mean`/`std`/`var`, whose output is always `f64`)
    // needs true `double`; movement-only aggs (`first`/`last`/`count`/`len`)
    // run anywhere. `product` preserves input dtype, `median` preserves it
    // but needs no arithmetic on the device.
    let needs_f64 = match agg {
        Aggregation::Sum { input, .. } |
        Aggregation::Min { input, .. } |
        Aggregation::Max { input, .. } |
        Aggregation::Product { input, .. } => {
            let index = find_column(columns, input)?;
            columns[index].dtype() == DType::F64
        }
        Aggregation::Mean { .. } | Aggregation::Std { .. } | Aggregation::Var { .. } => true,
        Aggregation::Count { .. } |
        Aggregation::First { .. } |
        Aggregation::Last { .. } |
        Aggregation::Len { .. } |
        Aggregation::Median { .. } => false,
    };
    if needs_f64 && !f64_ok {
        return Err(ExecError::BackendUnsupported(
            "f64 aggregation needs double support".to_string(),
        ));
    }
    match agg {
        Aggregation::Sum { input, output } => {
            let (dtype, values, validity) = reduce_sum(client, columns, input, seg)?;
            Ok(DeviceColumn::from_handles(output.clone(), dtype, values, validity, seg.num_groups))
        }
        Aggregation::Min { input, output } => {
            let index = find_column(columns, input)?;
            let dtype = columns[index].dtype();
            let (values, validity) = reduce_min_max(client, columns, input, seg, true)?;
            Ok(DeviceColumn::from_handles(output.clone(), dtype, values, validity, seg.num_groups))
        }
        Aggregation::Max { input, output } => {
            let index = find_column(columns, input)?;
            let dtype = columns[index].dtype();
            let (values, validity) = reduce_min_max(client, columns, input, seg, false)?;
            Ok(DeviceColumn::from_handles(output.clone(), dtype, values, validity, seg.num_groups))
        }
        Aggregation::Mean { input, output } => {
            let index = find_column(columns, input)?;
            let source = &columns[index];
            let out_values = alloc(client, seg.num_groups.saturating_mul(8));
            let out_valid = alloc(client, seg.num_groups);
            launch_mean_for(client, source, seg, &out_values, &out_valid)?;
            Ok(DeviceColumn::from_handles(
                output.clone(),
                DType::F64,
                out_values,
                out_valid,
                seg.num_groups,
            ))
        }
        Aggregation::Count { input, output } => {
            let index = find_column(columns, input)?;
            let source = &columns[index];
            let out_values = alloc(client, seg.num_groups.saturating_mul(4));
            let out_valid = alloc(client, seg.num_groups);
            polars_cubecl_kernels::launch_reduce_count(
                client,
                source.validity(),
                source.len(),
                polars_cubecl_kernels::ReduceParams {
                    slots: seg.slots,
                    seg_off: seg.seg_off,
                    seg_len: seg.seg_len,
                    num_groups: seg.num_groups,
                    out_values: &out_values,
                    out_valid: &out_valid,
                },
            )?;
            Ok(DeviceColumn::from_handles(
                output.clone(),
                DType::U32,
                out_values,
                out_valid,
                seg.num_groups,
            ))
        }
        Aggregation::First { input, output } => {
            pick_end(client, columns, input, output, &grouping.first_rows, seg.num_groups)
        }
        Aggregation::Last { input, output } => {
            pick_end(client, columns, input, output, &grouping.last_rows, seg.num_groups)
        }
        Aggregation::Product { input, output } => {
            let (dtype, values, validity) = reduce_product(client, columns, input, seg)?;
            Ok(DeviceColumn::from_handles(output.clone(), dtype, values, validity, seg.num_groups))
        }
        Aggregation::Std { input, output, ddof } => {
            let (values, validity) = reduce_std(client, columns, input, seg, *ddof)?;
            Ok(DeviceColumn::from_handles(
                output.clone(),
                DType::F64,
                values,
                validity,
                seg.num_groups,
            ))
        }
        Aggregation::Var { input, output, ddof } => {
            let (values, validity) = reduce_var(client, columns, input, seg, *ddof)?;
            Ok(DeviceColumn::from_handles(
                output.clone(),
                DType::F64,
                values,
                validity,
                seg.num_groups,
            ))
        }
        Aggregation::Median { input, output } => reduce_median(client, columns, input, output, seg),
        Aggregation::Len { output } => {
            let out_values = create_slice(client, bytemuck::cast_slice(&grouping.sizes));
            let ones = vec![1_u8; seg.num_groups];
            let out_valid = create_slice(client, &ones);
            Ok(DeviceColumn::from_handles(
                output.clone(),
                DType::U32,
                out_values,
                out_valid,
                seg.num_groups,
            ))
        }
    }
}

/// Grouped sum in the input dtype (the CPU engine does not widen small
/// sums). Returns the output dtype with its buffers.
///
/// NOTE: the parallel reduce_sum kernel (reduce_sum_parallel) was
/// prototyped but benchmarks showed it was ~17% slower at 1M rows/64 groups
/// (5.73 ms vs 4.89 ms serial). The CUDA launch overhead of 64 blocks plus
/// the 256 serial global-memory reads by thread 0 outweighed the benefit of
/// higher thread count. The parallel kernel is retained in the kernel crate
/// for future use when a cross-block barrier is available.
fn reduce_sum(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    seg: &Segmentation<'_>,
) -> Result<(DType, Handle, Handle), ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let out_valid = alloc(client, seg.num_groups);
    macro_rules! run {
        ($t:ty, $dtype:expr) => {{
            let out_values = alloc(client, seg.num_groups.saturating_mul(dtype_size($dtype)));
            polars_cubecl_kernels::launch_reduce_sum::<$t>(
                client,
                source.values(),
                source.validity(),
                source.len(),
                polars_cubecl_kernels::ReduceParams {
                    slots: seg.slots,
                    seg_off: seg.seg_off,
                    seg_len: seg.seg_len,
                    num_groups: seg.num_groups,
                    out_values: &out_values,
                    out_valid: &out_valid,
                },
            )?;
            ($dtype, out_values)
        }};
    }
    let (dtype, out_values) = match source.dtype() {
        DType::I32 => run!(i32, DType::I32),
        DType::I64 => run!(i64, DType::I64),
        DType::U32 => run!(u32, DType::U32),
        DType::U64 => run!(u64, DType::U64),
        DType::F32 => run!(f32, DType::F32),
        DType::F64 => run!(f64, DType::F64),
        DType::Bool => {
            return Err(ExecError::Inconsistent("sum over booleans".to_string()));
        }
    };
    Ok((dtype, out_values, out_valid))
}

/// Grouped minimum (`find_min == true`) or maximum.
fn reduce_min_max(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    seg: &Segmentation<'_>,
    find_min: bool,
) -> Result<(Handle, Handle), ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let out_values = alloc(client, seg.num_groups.saturating_mul(dtype_size(source.dtype())));
    let out_valid = alloc(client, seg.num_groups);
    macro_rules! run {
        ($t:ty, $kernel:ident) => {
            polars_cubecl_kernels::$kernel::<$t>(
                client,
                source.values(),
                source.validity(),
                source.len(),
                polars_cubecl_kernels::ReduceParams {
                    slots: seg.slots,
                    seg_off: seg.seg_off,
                    seg_len: seg.seg_len,
                    num_groups: seg.num_groups,
                    out_values: &out_values,
                    out_valid: &out_valid,
                },
            )?
        };
    }
    macro_rules! run_float {
        ($kernel:ident) => {
            polars_cubecl_kernels::$kernel(
                client,
                source.values(),
                source.validity(),
                source.len(),
                polars_cubecl_kernels::ReduceParams {
                    slots: seg.slots,
                    seg_off: seg.seg_off,
                    seg_len: seg.seg_len,
                    num_groups: seg.num_groups,
                    out_values: &out_values,
                    out_valid: &out_valid,
                },
            )?
        };
    }
    if find_min {
        match source.dtype() {
            DType::I32 => run!(i32, launch_reduce_min),
            DType::I64 => run!(i64, launch_reduce_min),
            DType::U32 => run!(u32, launch_reduce_min),
            DType::U64 => run!(u64, launch_reduce_min),
            DType::F32 => run_float!(launch_reduce_min_f32),
            DType::F64 => run_float!(launch_reduce_min_f64),
            DType::Bool => run!(u8, launch_reduce_min),
        }
    } else {
        match source.dtype() {
            DType::I32 => run!(i32, launch_reduce_max),
            DType::I64 => run!(i64, launch_reduce_max),
            DType::U32 => run!(u32, launch_reduce_max),
            DType::U64 => run!(u64, launch_reduce_max),
            DType::F32 => run_float!(launch_reduce_max_f32),
            DType::F64 => run_float!(launch_reduce_max_f64),
            DType::Bool => run!(u8, launch_reduce_max),
        }
    }
    Ok((out_values, out_valid))
}

fn launch_mean_for(
    client: &Client,
    source: &DeviceColumn,
    seg: &Segmentation<'_>,
    out_values: &Handle,
    out_valid: &Handle,
) -> Result<(), ExecError> {
    match source.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_reduce_mean::<i32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::I64 => polars_cubecl_kernels::launch_reduce_mean::<i64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::U32 => polars_cubecl_kernels::launch_reduce_mean::<u32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::U64 => polars_cubecl_kernels::launch_reduce_mean::<u64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::F32 => polars_cubecl_kernels::launch_reduce_mean::<f32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::F64 => polars_cubecl_kernels::launch_reduce_mean::<f64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
        DType::Bool => polars_cubecl_kernels::launch_reduce_mean::<u8>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            polars_cubecl_kernels::ReduceParams {
                slots: seg.slots,
                seg_off: seg.seg_off,
                seg_len: seg.seg_len,
                num_groups: seg.num_groups,
                out_values,
                out_valid,
            },
        )?,
    }
    Ok(())
}

/// First/last value of every group in row order, null-preserving. Positions
/// come from the host grouping, so no device indirection is involved.
fn pick_end(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    output: &PlSmallStr,
    rows: &[u32],
    num_groups: usize,
) -> Result<DeviceColumn, ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let positions_handle = create_slice(client, bytemuck::cast_slice(rows));
    let values = alloc(client, num_groups.saturating_mul(dtype_size(source.dtype())));
    let validity = alloc(client, num_groups);
    gather_picked(client, source, &positions_handle, &values, &validity, num_groups)?;
    Ok(DeviceColumn::from_handles(output.clone(), source.dtype(), values, validity, num_groups))
}

/// Grouped product in the input dtype. Returns the output dtype with its buffers.
fn reduce_product(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    seg: &Segmentation<'_>,
) -> Result<(DType, Handle, Handle), ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let out_valid = alloc(client, seg.num_groups);
    macro_rules! run {
        ($t:ty, $dtype:expr) => {{
            let out_values = alloc(client, seg.num_groups.saturating_mul(dtype_size($dtype)));
            polars_cubecl_kernels::launch_reduce_product::<$t>(
                client,
                source.values(),
                source.validity(),
                source.len(),
                polars_cubecl_kernels::ReduceParams {
                    slots: seg.slots,
                    seg_off: seg.seg_off,
                    seg_len: seg.seg_len,
                    num_groups: seg.num_groups,
                    out_values: &out_values,
                    out_valid: &out_valid,
                },
            )?;
            ($dtype, out_values)
        }};
    }
    let (dtype, out_values) = match source.dtype() {
        DType::I32 => run!(i32, DType::I32),
        DType::I64 => run!(i64, DType::I64),
        DType::U32 => run!(u32, DType::U32),
        DType::U64 => run!(u64, DType::U64),
        DType::F32 => run!(f32, DType::F32),
        DType::F64 => run!(f64, DType::F64),
        DType::Bool => {
            return Err(ExecError::Inconsistent("product over booleans".to_string()));
        }
    };
    Ok((dtype, out_values, out_valid))
}

/// Grouped standard deviation in `f64`. Dispatches to the type-specific kernel.
fn reduce_std(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    seg: &Segmentation<'_>,
    ddof: u8,
) -> Result<(Handle, Handle), ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let out_values = alloc(client, seg.num_groups.saturating_mul(8));
    let out_valid = alloc(client, seg.num_groups);
    match source.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_reduce_std::<i32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::I64 => polars_cubecl_kernels::launch_reduce_std::<i64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::U32 => polars_cubecl_kernels::launch_reduce_std::<u32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::U64 => polars_cubecl_kernels::launch_reduce_std::<u64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::F32 => polars_cubecl_kernels::launch_reduce_std::<f32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::F64 => polars_cubecl_kernels::launch_reduce_std::<f64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::Bool => polars_cubecl_kernels::launch_reduce_std::<u8>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
    }
    Ok((out_values, out_valid))
}

/// Grouped variance in `f64`. Dispatches to the type-specific kernel.
fn reduce_var(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    seg: &Segmentation<'_>,
    ddof: u8,
) -> Result<(Handle, Handle), ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let out_values = alloc(client, seg.num_groups.saturating_mul(8));
    let out_valid = alloc(client, seg.num_groups);
    match source.dtype() {
        DType::I32 => polars_cubecl_kernels::launch_reduce_var::<i32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::I64 => polars_cubecl_kernels::launch_reduce_var::<i64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::U32 => polars_cubecl_kernels::launch_reduce_var::<u32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::U64 => polars_cubecl_kernels::launch_reduce_var::<u64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::F32 => polars_cubecl_kernels::launch_reduce_var::<f32>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::F64 => polars_cubecl_kernels::launch_reduce_var::<f64>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
        DType::Bool => polars_cubecl_kernels::launch_reduce_var::<u8>(
            client,
            source.values(),
            source.validity(),
            source.len(),
            ddof,
            params(seg, &out_values, &out_valid),
        )?,
    }
    Ok((out_values, out_valid))
}

fn params<'a>(
    seg: &Segmentation<'a>,
    out_values: &'a Handle,
    out_valid: &'a Handle,
) -> polars_cubecl_kernels::ReduceParams<'a> {
    polars_cubecl_kernels::ReduceParams {
        slots: seg.slots,
        seg_off: seg.seg_off,
        seg_len: seg.seg_len,
        num_groups: seg.num_groups,
        out_values,
        out_valid,
    }
}

/// Grouped median, computed host-side: gather valid values per group, sort,
/// and pick the middle (or average of two middles) element.
///
/// The raw device handles for `slots`, `seg_off`, and `seg_len` are read back
/// with `client.read_one` and interpreted as `u32` slices.
fn reduce_median(
    client: &Client,
    columns: &[DeviceColumn],
    input: &PlSmallStr,
    output: &PlSmallStr,
    seg: &Segmentation<'_>,
) -> Result<DeviceColumn, ExecError> {
    let index = find_column(columns, input)?;
    let source = &columns[index];
    let host = download(client, source)?;

    // Read raw slot/segment data from device handles.
    let slots_bytes =
        client.read_one(seg.slots.clone()).map_err(|e| ExecError::Launch(e.to_string()))?;
    let seg_off_bytes =
        client.read_one(seg.seg_off.clone()).map_err(|e| ExecError::Launch(e.to_string()))?;
    let seg_len_bytes =
        client.read_one(seg.seg_len.clone()).map_err(|e| ExecError::Launch(e.to_string()))?;
    let slots_u32: &[u32] = bytemuck::cast_slice(&slots_bytes);
    let seg_off: &[u32] = bytemuck::cast_slice(&seg_off_bytes);
    let seg_len: &[u32] = bytemuck::cast_slice(&seg_len_bytes);

    // Gather valid values per group from the slot permutation.
    let mut group_values: Vec<Vec<f64>> = vec![Vec::new(); seg.num_groups];
    for group in 0..seg.num_groups {
        let base = seg_off[group] as usize;
        let len = seg_len[group] as usize;
        for k in 0..len {
            let row = slots_u32[base + k] as usize;
            if host.validity()[row] != 0 {
                let val = match (source.dtype(), host.values()) {
                    (DType::I32, HostValues::I32(v)) => f64::from(v[row]),
                    (DType::I64, HostValues::I64(v)) => v[row] as f64,
                    (DType::U32, HostValues::U32(v)) => f64::from(v[row]),
                    (DType::U64, HostValues::U64(v)) => v[row] as f64,
                    (DType::F32, HostValues::F32(v)) => f64::from(v[row]),
                    (DType::F64, HostValues::F64(v)) => v[row],
                    (DType::Bool, HostValues::Bool(v)) => f64::from(v[row]),
                    _ => 0.0,
                };
                group_values[group].push(val);
            }
        }
    }

    // Compute median per group.
    let mut out_vals = vec![0.0f64; seg.num_groups];
    let mut out_valid = vec![0u8; seg.num_groups];
    for group in 0..seg.num_groups {
        let vals = &mut group_values[group];
        if vals.is_empty() {
            out_vals[group] = 0.0;
            out_valid[group] = 0;
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let median =
            if n % 2 == 1 { vals[n / 2] } else { f64::midpoint(vals[n / 2 - 1], vals[n / 2]) };
        out_vals[group] = median;
        out_valid[group] = 1;
    }

    // Median is always output as f64 (matches Polars semantics).
    let out_valid_buf = create_slice(client, &out_valid);
    let out_values_buf = create_slice(client, bytemuck::cast_slice(&out_vals));

    Ok(DeviceColumn::from_handles(
        output.clone(),
        DType::F64,
        out_values_buf,
        out_valid_buf,
        seg.num_groups,
    ))
}
