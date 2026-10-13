//! `#[cube]` kernels for the engine's core operators, plus safe launch wrappers.
//!
//! One kernel is written once and compiled per backend by CubeCL. Topology
//! constants are read, never hard-coded: `ABSOLUTE_POS` (`usize`) addresses
//! rows and `UNIT_POS` (`u32`) addresses units inside a cube, so the same
//! source runs on a 32-lane warp, a 64-lane wavefront, and the plane-of-1 CPU
//! backend.
//!
//! The raw kernels live in the private [`inner`] module because the `#[cube]`
//! macro generates public helper items that cannot carry doc comments.
//! Callers use the documented `launch_*` wrappers below, which own the whole
//! launch protocol: buffer wrapping (the only `unsafe` in the engine,
//! justified once here), cube topology, and empty-input elision.
//!
//! Comparison and logic operators are intentionally one kernel each rather
//! than a comptime-dispatched generic: that keeps every kernel a straight
//! line of code with no macro-generated dispatch tables.
//!
//! Two pedantic lints are expected crate-wide because the `#[cube]` macro
//! does not expand `From`/`const fn` the way plain Rust does:
//! `bool`→`u8` must use `as u8` (`u8::from` does not expand in-kernel), and
//! macro-generated kernels cannot be `const`.

#![expect(clippy::cast_lossless, clippy::missing_const_for_fn)]

use cubecl::prelude::{BufferArg, Client, CubeCount, CubeDim, Numeric};
use cubecl_runtime::server::Handle;
pub use inner::{BinOp, CmpOp, LogicOp, UnaryOp};
use thiserror::Error;

/// Threads per cube for elementwise kernels.
pub const BLOCK: u32 = 256;

/// Rows processed serially by one thread in the strip-mined filter kernels.
/// Threads own contiguous row chunks, so access stays coalesced; all
/// coordination happens through host-side prefix sums, never through
/// cross-unit shared memory or barriers.
pub const STRIDE_ROWS: usize = 16;

/// Thread count covering `height` rows at [`STRIDE_ROWS`] rows per thread.
#[must_use]
pub const fn filter_threads(height: usize) -> usize {
    height.div_ceil(STRIDE_ROWS)
}

/// Error returned when a launch cannot be set up on the host side.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    /// The row count does not fit the device address space.
    #[error("row count out of range for a kernel launch")]
    RowCount,
}

/// Wrap a device buffer for a kernel launch.
///
/// # Safety
///
/// The handle must reference a live device allocation holding at least `len`
/// elements of the kernel's element type. Every wrapper below upholds this by
/// construction: handles are freshly allocated for exactly the launched
/// shape, and `Handle` keeps the allocation alive.
unsafe fn wrap(handle: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
}

fn count_for(height: usize) -> Result<Option<CubeCount>, KernelError> {
    if height == 0 {
        return Ok(None);
    }
    let blocks = height.div_ceil(BLOCK as usize);
    let blocks = u32::try_from(blocks).map_err(|_| KernelError::RowCount)?;
    Ok(Some(CubeCount::Static(blocks, 1, 1)))
}

const fn dim() -> CubeDim {
    CubeDim::new_1d(BLOCK)
}

/// Warm-up representative kernel paths so the first *real* launch does not
/// pay backend first-compilation or context initialisation costs.
///
/// For CUDA that means PTX→cubin JIT and module load; for the CPU backend it
/// warms the thread pool. Idempotent and safe to call more than once.
///
/// # Errors
///
/// Returns the first [`KernelError`] if any warm-up launch cannot be set up
/// (never happens for any linked backend: the input is always ≤ [`BLOCK`]).
pub fn warm_up(client: &Client) -> Result<(), KernelError> {
    const N: usize = 8;
    // Helper: allocate a zeroed buffer of the right byte size.
    macro_rules! zeros {
        ($ty:ty) => {
            client.create_from_slice(bytemuck::cast_slice(&[0 as $ty; N]))
        };
    }
    let v_i32 = zeros!(i32);
    let v_i64 = zeros!(i64);
    let v_u32 = zeros!(u32);
    let v_f32 = zeros!(f32);
    let v_f64 = zeros!(f64);
    let valid = client.create_from_slice(&[1u8; N]);
    let out_u32 = zeros!(u32);
    let out_i32 = zeros!(i32);
    let out_i64 = zeros!(i64);
    let out_f64 = zeros!(f64);
    let out_bool = zeros!(u8);
    let lit_i64 = zeros!(i64);
    let lit_f64 = zeros!(f64);

    // Comparison kernels.
    launch_cmp::<i32>(client, &v_i32, &valid, &lit_i64, &out_bool, N, CmpOp::Gt)?;
    launch_cmp::<i64>(client, &v_i64, &valid, &lit_i64, &out_bool, N, CmpOp::Eq)?;
    launch_cmp_f32(client, &v_f32, &valid, &lit_f64, &out_bool, N, CmpOp::Lt)?;
    launch_cmp_f64(client, &v_f64, &valid, &lit_f64, &out_bool, N, CmpOp::Gt)?;

    // Column-vs-column comparison.
    launch_col_cmp::<i32>(
        client,
        &v_i32,
        &valid,
        &v_i32,
        &valid,
        &out_bool,
        &out_bool,
        N,
        CmpOp::Gt,
    )?;
    launch_col_cmp_f64(client, &v_f64, &valid, &v_f64, &valid, &out_bool, &out_bool, N, CmpOp::Eq)?;

    // Boolean logic kernels.
    launch_logic(client, &valid, &valid, &out_bool, N, inner::LogicOp::And)?;
    launch_not(client, &valid, &out_bool, N)?;

    // Binary arithmetic kernels.
    launch_binop::<i32>(
        client,
        &v_i32,
        &valid,
        &v_i32,
        &valid,
        &out_i32,
        &out_bool,
        N,
        inner::BinOp::Add,
    )?;
    launch_binop::<i64>(
        client,
        &v_i64,
        &valid,
        &v_i64,
        &valid,
        &out_i64,
        &out_bool,
        N,
        inner::BinOp::Mul,
    )?;
    launch_binop_f64(
        client,
        &v_f64,
        &valid,
        &v_f64,
        &valid,
        &out_f64,
        &out_bool,
        N,
        inner::BinOp::Add,
    )?;

    // Unary kernels.
    launch_unaryop::<i32>(client, &v_i32, &valid, &out_i32, &out_bool, N, inner::UnaryOp::Abs)?;
    launch_unaryop_f64(client, &v_f64, &valid, &out_f64, &out_bool, N, inner::UnaryOp::Exp)?;

    // Filter helpers: count + gather + slice.
    launch_count(client, &valid, &out_u32, N, 1)?;
    launch_gather::<i64>(client, &v_i64, N, &v_u32, &out_i64, N)?;
    launch_slice::<i64>(client, &v_i64, N, &out_i64, 0, N)?;

    // Group-by / segmented reduce kernels.
    let slots = zeros!(u32);
    let seg_off = client.create_from_slice(bytemuck::cast_slice(&[0u32; 4]));
    let seg_len = client.create_from_slice(bytemuck::cast_slice(&[0u32; 4]));
    let groups = zeros!(u32);
    let group_counts = zeros!(u32);
    launch_group_count(client, &groups, &group_counts, N, 4)?;
    launch_group_invert(client, &groups, &seg_off, &seg_len, &slots, N, 4)?;
    let r_params = ReduceParams {
        slots: &slots,
        seg_off: &seg_off,
        seg_len: &seg_len,
        out_values: &out_i64,
        out_valid: &out_bool,
        num_groups: 4,
    };
    launch_reduce_sum::<i64>(client, &v_i64, &valid, N, r_params)?;
    launch_reduce_min::<i64>(client, &v_i64, &valid, N, r_params)?;
    launch_reduce_mean::<f64>(client, &v_f64, &valid, N, r_params)?;
    launch_reduce_count(client, &valid, N, ReduceParams {
        slots: &slots,
        seg_off: &seg_off,
        seg_len: &seg_len,
        out_values: &out_u32,
        out_valid: &out_bool,
        num_groups: 4,
    })?;

    // Cast kernels.
    launch_cast_i32_to_f64(client, &v_i32, &out_f64, N)?;
    launch_cast_u32_to_f64(client, &v_u32, &out_f64, N)?;

    // Force synchronisation so all warm-ups complete before returning.
    // Dropping the boxed future is sufficient: the synchronous work
    // (command submission, stream execution) happens before the future
    // is created in every CubeCL backend.
    drop(client.sync());
    Ok(())
}

/// Elementwise comparison of `values` against a single literal.
///
/// A null input row (validity `0`) always yields `0`, matching Polars: a null
/// comparison never passes a filter. Serves integer and boolean types; floats
/// go through [`launch_cmp_f32`]/[`launch_cmp_f64`].
pub fn launch_cmp<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    lit: &Handle,
    out: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    // SAFETY: see [`wrap`]; all four handles are freshly allocated for
    // exactly `height` (one for `lit`) elements of `T`/`u8`.
    unsafe {
        match op {
            CmpOp::Eq => inner::cmp_eq_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::NotEq => inner::cmp_ne_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Lt => inner::cmp_lt_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::LtEq => inner::cmp_le_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Gt => inner::cmp_gt_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::GtEq => inner::cmp_ge_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f32` comparison under the total order Polars uses.
///
/// Backed by the concrete `is_nan()`-based kernels, which survive the
/// fast-math folding that breaks the `x != x` idiom on some backends.
pub fn launch_cmp_f32(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    lit: &Handle,
    out: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        match op {
            CmpOp::Eq => inner::cmp_f32_eq_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::NotEq => inner::cmp_f32_ne_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Lt => inner::cmp_f32_lt_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::LtEq => inner::cmp_f32_le_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Gt => inner::cmp_f32_gt_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::GtEq => inner::cmp_f32_ge_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f64` comparison under the total order Polars uses.
///
/// Same fast-math rationale as [`launch_cmp_f32`]. Requires a backend with
/// `double` support; the executor gates this (see `supports_f64`).
pub fn launch_cmp_f64(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    lit: &Handle,
    out: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        match op {
            CmpOp::Eq => inner::cmp_f64_eq_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::NotEq => inner::cmp_f64_ne_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Lt => inner::cmp_f64_lt_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::LtEq => inner::cmp_f64_le_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::Gt => inner::cmp_f64_gt_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
            CmpOp::GtEq => inner::cmp_f64_ge_kernel::launch(
                client,
                count,
                dim,
                wrap(values, height),
                wrap(valid, height),
                wrap(lit, 1),
                wrap(out, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise boolean combination of two masks.
pub fn launch_logic(
    client: &Client,
    lhs: &Handle,
    rhs: &Handle,
    out: &Handle,
    height: usize,
    op: LogicOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    // SAFETY: see [`wrap`]; all three handles hold exactly `height` bytes.
    unsafe {
        match op {
            LogicOp::And => inner::logic_and_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(rhs, height),
                wrap(out, height),
            ),
            LogicOp::Or => inner::logic_or_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(rhs, height),
                wrap(out, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise boolean negation of a mask.
pub fn launch_not(
    client: &Client,
    input: &Handle,
    out: &Handle,
    height: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; both handles hold exactly `height` bytes.
    unsafe {
        inner::not_kernel::launch(client, count, dim(), wrap(input, height), wrap(out, height));
    }
    Ok(())
}

/// Count passing rows per strip-mined thread chunk into `counts`.
///
/// Thread `t` serially counts rows `[t * STRIDE_ROWS, (t + 1) * STRIDE_ROWS)`.
/// Chunks are disjoint, so the writes are race-free and the result is
/// deterministic on every backend without any synchronization.
pub fn launch_count(
    client: &Client,
    mask: &Handle,
    counts: &Handle,
    height: usize,
    threads: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(threads)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `mask` holds `height` bytes, `counts` one `u32`
    // per thread.
    unsafe {
        inner::count_kernel::launch(
            client,
            count,
            dim(),
            wrap(mask, height),
            wrap(counts, threads),
            STRIDE_ROWS,
        );
    }
    Ok(())
}

/// Scatter the indices of passing rows into `indices`.
///
/// `bases[t]` is the exclusive-prefix base of thread `t` (computed on the
/// host from [`launch_count`] output). Each thread serially emits its own
/// passing rows in row order, so the concatenation is the stable compaction
/// of the mask. `indices` must hold `kept` `u32` entries.
pub fn launch_scatter(
    client: &Client,
    mask: &Handle,
    bases: &Handle,
    indices: &Handle,
    height: usize,
    threads: usize,
    kept: usize,
) -> Result<(), KernelError> {
    if kept == 0 {
        return Ok(());
    }
    let Some(count) = count_for(threads)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::scatter_kernel::launch(
            client,
            count,
            dim(),
            wrap(mask, height),
            wrap(bases, threads),
            wrap(indices, kept),
            STRIDE_ROWS,
        );
    }
    Ok(())
}

/// Gather `values` through `indices`: `out[i] = values[indices[i]]`.
pub fn launch_gather<T: Numeric>(
    client: &Client,
    values: &Handle,
    values_len: usize,
    indices: &Handle,
    out: &Handle,
    kept: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(kept)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `values` holds `values_len` elements of `T`,
    // `indices` and `out` hold `kept` `u32`/elements.
    unsafe {
        inner::gather_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(indices, kept),
            wrap(out, kept),
        );
    }
    Ok(())
}

/// Fused gather of both values and validity mask through `indices` in a
/// single kernel launch.
///
/// This replaces the two separate `launch_gather` calls (one for values, one
/// for the u8 validity mask) used by filter compaction with a single launch,
/// halving the kernel-launch overhead per column during filtering.
pub fn launch_gather_values_mask<T: Numeric>(
    client: &Client,
    values: &Handle,
    values_len: usize,
    validity: &Handle,
    indices: &Handle,
    out_values: &Handle,
    out_validity: &Handle,
    kept: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(kept)? else { return Ok(()) };
    unsafe {
        inner::gather_values_mask_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(validity, values_len),
            wrap(indices, kept),
            wrap(out_values, kept),
            wrap(out_validity, kept),
        );
    }
    Ok(())
}

/// Slice `values` into `out`: `out[i] = values[offset + i]`.
pub fn launch_slice<T: Numeric>(
    client: &Client,
    values: &Handle,
    values_len: usize,
    out: &Handle,
    offset: usize,
    count: usize,
) -> Result<(), KernelError> {
    let Some(cubes) = count_for(count)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `values` holds `values_len` elements of `T` with
    // `offset + count <= values_len`, `out` holds `count`.
    unsafe {
        inner::slice_kernel::launch::<T>(
            client,
            cubes,
            dim(),
            wrap(values, values_len),
            wrap(out, count),
            offset,
        );
    }
    Ok(())
}

/// Widen `i32` values to `f64` (exact); validity passes through unchanged.
pub fn launch_cast_i32_to_f64(
    client: &Client,
    values: &Handle,
    out: &Handle,
    height: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `values` holds `height` `i32`, `out` `height` `f64`.
    unsafe {
        inner::cast_i32_to_f64_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, height),
            wrap(out, height),
        );
    }
    Ok(())
}

/// Widen `u32` values to `f64` (exact); validity passes through unchanged.
pub fn launch_cast_u32_to_f64(
    client: &Client,
    values: &Handle,
    out: &Handle,
    height: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `values` holds `height` `u32`, `out` `height` `f64`.
    unsafe {
        inner::cast_u32_to_f64_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, height),
            wrap(out, height),
        );
    }
    Ok(())
}

/// Count rows per group: `counts[groups[row]] += 1`.
///
/// Per-group sums commute, so the atomic increments are deterministic. Only
/// `u32` atomics are used, which every backend provides.
pub fn launch_group_count(
    client: &Client,
    groups: &Handle,
    counts: &Handle,
    height: usize,
    num_groups: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `groups` holds `height` ids below `num_groups`,
    // `counts` holds `num_groups` `u32`.
    unsafe {
        inner::group_count_kernel::launch(
            client,
            count,
            dim(),
            wrap(groups, height),
            wrap(counts, num_groups),
        );
    }
    Ok(())
}

/// GPU histogram: count occurrences of raw `i32` keys directly into per-bucket
/// counters on the device. Avoids downloading the key column to the host for
/// grouping; the host only reads back `num_buckets * 4` bytes of counts.
///
/// # Panics
///
/// Panics at compile time if any key falls outside `[min_val, min_val + num_buckets)`.
/// The caller validates the range before launch.
pub fn launch_gpu_histogram(
    client: &Client,
    keys: &Handle,
    min_val_handle: &Handle,
    counts_handle: &Handle,
    height: usize,
    num_buckets: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `keys` holds `height` i32 values, `min_val` holds
    // one scalar, `counts` holds `num_buckets` Atomic<u32>.
    unsafe {
        inner::gpu_histogram_kernel::launch(
            client,
            count,
            dim(),
            wrap(keys, height),
            wrap(min_val_handle, 1),
            wrap(counts_handle, num_buckets),
        );
    }
    Ok(())
}

/// Compute per-row group IDs from raw `i32` keys, entirely on the GPU.
/// `out[row] = (keys[row] - min_val) as u32`. Avoids downloading the key.
pub fn launch_keys_to_group_i32(
    client: &Client,
    keys: &Handle,
    min_val_handle: &Handle,
    out: &Handle,
    height: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; `keys` is i32, `min_val` is a scalar, `out` is u32.
    unsafe {
        inner::keys_to_group_i32_kernel::launch(
            client,
            count,
            dim(),
            wrap(keys, height),
            wrap(min_val_handle, 1),
            wrap(out, height),
        );
    }
    Ok(())
}

/// Fused GPU histogram + group-id assignment in a single launch.
///
/// Combines `launch_gpu_histogram` + `launch_keys_to_group_i32` into one
/// kernel, eliminating a ~500µs CUDA launch. Reads `keys[row]` once and
/// computes both the atomic bucket count and the dense group-id output.
pub fn launch_gpu_histogram_and_assign(
    client: &Client,
    keys: &Handle,
    min_val_handle: &Handle,
    counts_handle: &Handle,
    out: &Handle,
    height: usize,
    num_buckets: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`].
    unsafe {
        inner::gpu_histogram_and_assign_kernel::launch(
            client,
            count,
            dim(),
            wrap(keys, height),
            wrap(min_val_handle, 1),
            wrap(counts_handle, num_buckets),
            wrap(out, height),
        );
    }
    Ok(())
}

/// Invert the row-to-group mapping into contiguous per-group segments.
///
/// `seq` holds zeroed per-group counters; each row claims
/// `slots[seg_off[group] + seq[group]++] = row`. Slots form a set partition
/// of the rows (deterministic as a set; order within a segment is atomic
/// order, which only float-sum LSBs can observe, inside tolerance).
pub fn launch_group_invert(
    client: &Client,
    groups: &Handle,
    seg_off: &Handle,
    seq: &Handle,
    slots: &Handle,
    height: usize,
    num_groups: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::group_invert_kernel::launch(
            client,
            count,
            dim(),
            wrap(groups, height),
            wrap(seg_off, num_groups),
            wrap(seq, num_groups),
            wrap(slots, height),
        );
    }
    Ok(())
}

/// Segmented sum, one thread per group, serial within the segment.
///
/// Integer sums are exact regardless of slot order; float sums may vary in
/// Parameters for a segmented reduction kernel launch.
///
/// Bundles the eight handles/usizes shared by every segmented launch wrapper,
/// keeping each wrapper at three arguments (under the `too-many-arguments`
/// threshold).
#[derive(Clone, Copy, Debug)]
pub struct ReduceParams<'a> {
    /// Group slot indices.
    pub slots: &'a Handle,
    /// Group start offsets.
    pub seg_off: &'a Handle,
    /// Group lengths.
    pub seg_len: &'a Handle,
    /// Number of groups.
    pub num_groups: usize,
    /// Output value buffer.
    pub out_values: &'a Handle,
    /// Output validity buffer.
    pub out_valid: &'a Handle,
}

/// Sum directly from the per-row group-assignment vector, bypassing the
/// group_invert + slots indirection.
///
/// The host must zero-fill both `out_values` and `out_valid` before launch:
/// `out_values` holds `num_groups` elements of type T, `out_valid` holds
/// `num_groups` bytes. After launch they hold the per-group sum and validity.
///
/// This is ~500µs cheaper than the group_invert + reduce_sum combo and
/// eliminates the need for the intermediate slots buffer.
pub fn launch_reduce_sum_direct<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    assignment: &Handle,
    out_values: &Handle,
    out_valid: &Handle,
    height: usize,
    num_groups: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    unsafe {
        inner::reduce_sum_direct_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, height),
            wrap(valid, height),
            wrap(assignment, height),
            wrap(out_values, num_groups),
            wrap(out_valid, num_groups),
            num_groups,
        );
    }
    Ok(())
}

/// the last ulps with slot order, inside the comparator tolerance. Empty
/// segments yield the zero identity with valid output.
pub fn launch_reduce_sum<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_sum_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Parallel segmented sum using thread-scatter reduction — same semantics as
/// `launch_reduce_sum` but uses `reduce_threads_per_group` threads per group.
/// For low-cardinality groupBy (few groups × millions of rows), this can be
/// orders of magnitude faster because the GPU is no longer starved for warps.
///
/// The host must zero-fill `scratch` before launch: it has
/// `num_groups * reduce_threads_per_group` elements of type T.
pub fn launch_reduce_sum_parallel<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
    scratch: &Handle,
    reduce_threads_per_group: u32,
) -> Result<(), KernelError> {
    if params.num_groups == 0 {
        return Ok(());
    }
    // One block per group; each block has `reduce_threads_per_group` threads.
    // `ABSOLUTE_POS` ⇒ group = pos / threads_per_group, lane = pos % threads_per_group.
    let blocks = u32::try_from(params.num_groups).map_err(|_| KernelError::RowCount)?;
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_sum_parallel_kernel::launch::<T>(
            client,
            CubeCount::Static(blocks, 1, 1),
            CubeDim::new_1d(reduce_threads_per_group),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
            wrap(scratch, params.num_groups * reduce_threads_per_group as usize),
            reduce_threads_per_group as usize,
        );
    }
    Ok(())
}

/// Segmented minimum, one thread per group. Serves integer and boolean
/// types; float minimums go through the `is_nan()`-based
/// `launch_reduce_min_f32`/`f64` (see the fast-math note on
/// [`launch_cmp_f32`]). Empty segments yield null.
pub fn launch_reduce_min<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_min_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Fused segmented min+max, one thread per group. Single launch computes
/// both extrema simultaneously, halving CUDA launch overhead when both
/// min and max are needed (e.g. GPU histogram grouping).
///
/// Takes individual handles (not [`ReduceParams`]) because the fused kernel
/// needs three output buffers: `out_min`, `out_max`, and `out_valid`.
pub fn launch_reduce_min_max<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    slots: &Handle,
    seg_off: &Handle,
    seg_len: &Handle,
    num_groups: usize,
    out_min: &Handle,
    out_max: &Handle,
    out_valid: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_min_max_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(slots, values_len),
            wrap(seg_off, num_groups),
            wrap(seg_len, num_groups),
            wrap(out_min, num_groups),
            wrap(out_max, num_groups),
            wrap(out_valid, num_groups),
        );
    }
    Ok(())
}

/// Fused segmented min+max for floats with NaN handling.
///
/// `out_valid` may be a throwaway single-byte buffer when the caller does
/// not need validity information (e.g. GPU histogram path where nulls are
/// pre-checked).
macro_rules! launch_reduce_minmax_float {
    ($name:ident, $kernel:ident) => {
        /// Fused segmented min+max for floats with NaN handling.
        pub fn $name(
            client: &Client,
            values: &Handle,
            valid: &Handle,
            values_len: usize,
            slots: &Handle,
            seg_off: &Handle,
            seg_len: &Handle,
            num_groups: usize,
            out_min: &Handle,
            out_max: &Handle,
            out_valid: &Handle,
        ) -> Result<(), KernelError> {
            let Some(count) = count_for(num_groups)? else { return Ok(()) };
            // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
            unsafe {
                inner::$kernel::launch(
                    client,
                    count,
                    dim(),
                    wrap(values, values_len),
                    wrap(valid, values_len),
                    wrap(slots, values_len),
                    wrap(seg_off, num_groups),
                    wrap(seg_len, num_groups),
                    wrap(out_min, num_groups),
                    wrap(out_max, num_groups),
                    wrap(out_valid, num_groups),
                );
            }
            Ok(())
        }
    };
}

launch_reduce_minmax_float!(launch_reduce_min_max_f32, reduce_min_max_f32_kernel);
launch_reduce_minmax_float!(launch_reduce_min_max_f64, reduce_min_max_f64_kernel);

/// Fused global min+max **without slot indirection** — reads `values[row]`
/// directly in `[seg_off, seg_off+seg_len)`. Avoids the buffer-pool
/// uninit-read hazard and saves one global memory read per element.
///
/// For GPU histogram grouping, the entire column is a single contiguous
/// segment, so indirect `values[slots[base+k]]` indexing is unnecessary.
pub fn launch_reduce_min_max_direct<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    seg_off: u32,
    seg_len: u32,
    out_min: &Handle,
    out_max: &Handle,
    out_valid: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(1)? else { return Ok(()) };
    unsafe {
        inner::reduce_min_max_direct_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            seg_off,
            seg_len,
            wrap(out_min, 1),
            wrap(out_max, 1),
            wrap(out_valid, 1),
        );
    }
    Ok(())
}

/// Float direct min+max without slot indirection, NaN-safe.
macro_rules! launch_reduce_minmax_float_direct {
    ($name:ident, $kernel:ident) => {
        /// Launch a NaN-safe direct min+max reduction for float types without slot indirection.
        pub fn $name(
            client: &Client,
            values: &Handle,
            valid: &Handle,
            values_len: usize,
            seg_off: u32,
            seg_len: u32,
            out_min: &Handle,
            out_max: &Handle,
            out_valid: &Handle,
        ) -> Result<(), KernelError> {
            let Some(count) = count_for(1)? else { return Ok(()) };
            unsafe {
                inner::$kernel::launch(
                    client,
                    count,
                    dim(),
                    wrap(values, values_len),
                    wrap(valid, values_len),
                    seg_off,
                    seg_len,
                    wrap(out_min, 1),
                    wrap(out_max, 1),
                    wrap(out_valid, 1),
                );
            }
            Ok(())
        }
    };
}

launch_reduce_minmax_float_direct!(
    launch_reduce_min_max_direct_f32,
    reduce_min_max_direct_f32_kernel
);
launch_reduce_minmax_float_direct!(
    launch_reduce_min_max_direct_f64,
    reduce_min_max_direct_f64_kernel
);

/// Segmented maximum, one thread per group; integer/boolean types only.
/// Float maximums go through `launch_reduce_max_f32`/`f64`.
pub fn launch_reduce_max<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_max_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Segmented float minimum/maximum, one thread per group.
///
/// `NaN` values are skipped via `.is_nan()` (fast-math-safe); an all-`NaN`
/// segment yields a copied input `NaN` like the CPU engine, an empty one
/// yields null.
macro_rules! launch_reduce_minmax_float {
    ($name:ident, $kernel:ident) => {
        /// Segmented float minimum/maximum, one thread per group.
        ///
        /// `NaN` values are skipped via `.is_nan()` (fast-math-safe); an
        /// all-`NaN` segment yields a copied input `NaN` like the CPU engine,
        /// an empty one yields null.
        pub fn $name(
            client: &Client,
            values: &Handle,
            valid: &Handle,
            values_len: usize,
            params: ReduceParams<'_>,
        ) -> Result<(), KernelError> {
            let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
            // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
            unsafe {
                inner::$kernel::launch(
                    client,
                    count,
                    dim(),
                    wrap(values, values_len),
                    wrap(valid, values_len),
                    wrap(params.slots, values_len),
                    wrap(params.seg_off, params.num_groups),
                    wrap(params.seg_len, params.num_groups),
                    wrap(params.out_values, params.num_groups),
                    wrap(params.out_valid, params.num_groups),
                );
            }
            Ok(())
        }
    };
}

launch_reduce_minmax_float!(launch_reduce_min_f32, reduce_min_f32_kernel);
launch_reduce_minmax_float!(launch_reduce_max_f32, reduce_max_f32_kernel);
launch_reduce_minmax_float!(launch_reduce_min_f64, reduce_min_f64_kernel);
launch_reduce_minmax_float!(launch_reduce_max_f64, reduce_max_f64_kernel);

/// Segmented mean as `sum / count_valid` in `f64`, one thread per group.
/// Segments without valid values yield null.
pub fn launch_reduce_mean<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_mean_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Segmented count of valid values, one thread per group. Output is always
/// valid (an empty segment counts `0`, never null).
pub fn launch_reduce_count(
    client: &Client,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    // SAFETY: see [`wrap`]; shapes are exactly the launched ones.
    unsafe {
        inner::reduce_count_kernel::launch(
            client,
            count,
            dim(),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Segmented product, one thread per group. Empty segments yield identity `1`
/// with valid output (matches the CPU engine: product of nothing is `1`).
pub fn launch_reduce_product<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    unsafe {
        inner::reduce_product_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Segmented standard deviation as `sqrt( sum((x - mean)^2) / (n - ddof))` in
/// `f64`, one thread per group. Two-pass: mean first, then variance.
/// Segments with fewer than `ddof + 1` valid values yield null.
pub fn launch_reduce_std<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    ddof: u8,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    unsafe {
        inner::reduce_std_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            ddof,
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

/// Segmented variance as `sum((x - mean)^2) / (n - ddof)` in `f64`, one thread
/// per group. Segments with fewer than `ddof + 1` valid values yield null.
pub fn launch_reduce_var<T: Numeric>(
    client: &Client,
    values: &Handle,
    valid: &Handle,
    values_len: usize,
    ddof: u8,
    params: ReduceParams<'_>,
) -> Result<(), KernelError> {
    let Some(count) = count_for(params.num_groups)? else { return Ok(()) };
    unsafe {
        inner::reduce_var_kernel::launch::<T>(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(valid, values_len),
            wrap(params.slots, values_len),
            wrap(params.seg_off, params.num_groups),
            wrap(params.seg_len, params.num_groups),
            ddof,
            wrap(params.out_values, params.num_groups),
            wrap(params.out_valid, params.num_groups),
        );
    }
    Ok(())
}

// --- Binary arithmetic launch wrappers ---------------------------------

/// Elementwise binary arithmetic op on two (integer-typed) columns.
///
/// Null in either operand yields a null output row. Division-by-zero yields
/// a null row. Serves `i32`, `i64`, `u32`, `u64`; floats go through the
/// concrete `launch_binop_f32`/`launch_binop_f64` below.
pub fn launch_binop<T: Numeric>(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: BinOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    // SAFETY: see [`wrap`]; all handles are freshly allocated for the
    // launched shapes.
    unsafe {
        match op {
            BinOp::Add => inner::binop_add_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Sub => inner::binop_sub_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mul => inner::binop_mul_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Div => inner::binop_div_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::FloorDiv => inner::binop_floordiv_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mod => inner::binop_mod_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f32` binary arithmetic op.
pub fn launch_binop_f32(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: BinOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            BinOp::Add => inner::binop_f32_add_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Sub => inner::binop_f32_sub_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mul => inner::binop_f32_mul_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Div => inner::binop_f32_div_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::FloorDiv => inner::binop_f32_floordiv_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mod => inner::binop_f32_mod_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f64` binary arithmetic op.
pub fn launch_binop_f64(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: BinOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            BinOp::Add => inner::binop_f64_add_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Sub => inner::binop_f64_sub_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mul => inner::binop_f64_mul_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Div => inner::binop_f64_div_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::FloorDiv => inner::binop_f64_floordiv_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            BinOp::Mod => inner::binop_f64_mod_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

// --- Unary arithmetic launch wrappers ----------------------------------

/// Elementwise unary op on an integer-typed column.
pub fn launch_unaryop<T: Numeric>(
    client: &Client,
    input: &Handle,
    input_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: UnaryOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            UnaryOp::Abs => inner::unary_abs_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sign => inner::unary_sign_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Negate => inner::unary_negate_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Ceil |
            UnaryOp::Floor |
            UnaryOp::Exp |
            UnaryOp::Sqrt |
            UnaryOp::Log1p |
            UnaryOp::Sin |
            UnaryOp::Cos => {
                return Err(KernelError::RowCount); // Should be dispatched as float.
            }
        }
    }
    Ok(())
}

/// Elementwise `f32` unary op.
pub fn launch_unaryop_f32(
    client: &Client,
    input: &Handle,
    input_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: UnaryOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            UnaryOp::Abs => inner::unary_f32_abs_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Ceil => inner::unary_f32_ceil_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Floor => inner::unary_f32_floor_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Exp => inner::unary_f32_exp_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sqrt => inner::unary_f32_sqrt_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Log1p => inner::unary_f32_log1p_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sin => inner::unary_f32_sin_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Cos => inner::unary_f32_cos_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sign => inner::unary_f32_sign_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Negate => inner::unary_f32_negate_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f64` unary op.
pub fn launch_unaryop_f64(
    client: &Client,
    input: &Handle,
    input_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: UnaryOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            UnaryOp::Abs => inner::unary_f64_abs_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Ceil => inner::unary_f64_ceil_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Floor => inner::unary_f64_floor_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Exp => inner::unary_f64_exp_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sqrt => inner::unary_f64_sqrt_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Log1p => inner::unary_f64_log1p_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sin => inner::unary_f64_sin_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Cos => inner::unary_f64_cos_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Sign => inner::unary_f64_sign_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            UnaryOp::Negate => inner::unary_f64_negate_kernel::launch(
                client,
                count,
                dim,
                wrap(input, height),
                wrap(input_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

// --- Column-vs-column comparison launch wrappers ----------------------

/// Elementwise column-vs-column comparison for integer/boolean types.
pub fn launch_col_cmp<T: Numeric>(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            CmpOp::Eq => inner::col_cmp_eq_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::NotEq => inner::col_cmp_ne_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Lt => inner::col_cmp_lt_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::LtEq => inner::col_cmp_le_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Gt => inner::col_cmp_gt_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::GtEq => inner::col_cmp_ge_kernel::launch::<T>(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f32` column-vs-column comparison under the Polars total order.
pub fn launch_col_cmp_f32(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            CmpOp::Eq => inner::col_cmp_f32_eq_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::NotEq => inner::col_cmp_f32_ne_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Lt => inner::col_cmp_f32_lt_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::LtEq => inner::col_cmp_f32_le_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Gt => inner::col_cmp_f32_gt_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::GtEq => inner::col_cmp_f32_ge_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

/// Elementwise `f64` column-vs-column comparison under the Polars total order.
pub fn launch_col_cmp_f64(
    client: &Client,
    lhs: &Handle,
    lhs_valid: &Handle,
    rhs: &Handle,
    rhs_valid: &Handle,
    out: &Handle,
    out_valid: &Handle,
    height: usize,
    op: CmpOp,
) -> Result<(), KernelError> {
    let Some(count) = count_for(height)? else { return Ok(()) };
    let dim = dim();
    unsafe {
        match op {
            CmpOp::Eq => inner::col_cmp_f64_eq_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::NotEq => inner::col_cmp_f64_ne_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Lt => inner::col_cmp_f64_lt_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::LtEq => inner::col_cmp_f64_le_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::Gt => inner::col_cmp_f64_gt_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
            CmpOp::GtEq => inner::col_cmp_f64_ge_kernel::launch(
                client,
                count,
                dim,
                wrap(lhs, height),
                wrap(lhs_valid, height),
                wrap(rhs, height),
                wrap(rhs_valid, height),
                wrap(out, height),
                wrap(out_valid, height),
            ),
        }
    }
    Ok(())
}

// --- GPU LSD Radix Sort launch wrappers ---------------------------------

/// Number of 8-bit digit passes for sorting a `u32` key: 4.
pub const RADIX_PASSES_U32: u32 = 4;
/// Number of 8-bit digit passes for sorting a `u64` key: 8.
pub const RADIX_PASSES_U64: u32 = 8;
/// Radix sort digit width in bits.
const RADIX_BITS: u32 = 8;
/// Radix sort digit bins (2^RADIX_BITS).
pub const RADIX_BINS: u32 = 1 << RADIX_BITS;

/// Compute the 256-bin per-block histograms of one digit of a `u32` key
/// column. Output `histograms` must hold `blocks * 256` u32 words.
///
/// `blocks` is the number of parallel histogram blocks (one block thread per
/// output histogram), and `grid` is the number of elements each block scans.
pub fn launch_radix_histogram_u32(
    client: &Client,
    keys: &Handle,
    keys_len: usize,
    histograms: &Handle,
    blocks: usize,
    grid: usize,
    shift: u32,
) -> Result<(), KernelError> {
    let count = CubeCount::Static(u32::try_from(blocks).map_err(|_| KernelError::RowCount)?, 1, 1);
    unsafe {
        inner::radix_histogram_u32::launch(
            client,
            count,
            CubeDim::new_1d(1),
            wrap(keys, keys_len),
            wrap(histograms, blocks * RADIX_BINS as usize),
            blocks,
            grid,
            shift,
        );
    }
    Ok(())
}

/// Stable scatter a `u32` key column and its index buffer by one digit,
/// using the prefix-summed offsets in `scatter_offsets` (same layout as the
/// histogram: `blocks × 256`).
pub fn launch_radix_scatter_u32(
    client: &Client,
    keys_in: &Handle,
    keys_len: usize,
    indices_in: &Handle,
    scatter_offsets: &Handle,
    out_keys: &Handle,
    out_indices: &Handle,
    blocks: usize,
    grid: usize,
    shift: u32,
) -> Result<(), KernelError> {
    let count = CubeCount::Static(u32::try_from(blocks).map_err(|_| KernelError::RowCount)?, 1, 1);
    unsafe {
        inner::radix_scatter_u32::launch(
            client,
            count,
            CubeDim::new_1d(1),
            wrap(keys_in, keys_len),
            wrap(indices_in, keys_len),
            wrap(scatter_offsets, blocks * RADIX_BINS as usize),
            wrap(out_keys, keys_len),
            wrap(out_indices, keys_len),
            blocks,
            grid,
            shift,
        );
    }
    Ok(())
}

/// Same as [`launch_radix_histogram_u32`] but for `u64` keys.
pub fn launch_radix_histogram_u64(
    client: &Client,
    keys: &Handle,
    keys_len: usize,
    histograms: &Handle,
    blocks: usize,
    grid: usize,
    shift: u64,
) -> Result<(), KernelError> {
    let count = CubeCount::Static(u32::try_from(blocks).map_err(|_| KernelError::RowCount)?, 1, 1);
    unsafe {
        inner::radix_histogram_u64::launch(
            client,
            count,
            CubeDim::new_1d(1),
            wrap(keys, keys_len),
            wrap(histograms, blocks * RADIX_BINS as usize),
            blocks,
            grid,
            shift,
        );
    }
    Ok(())
}

/// Same as [`launch_radix_scatter_u32`] but for `u64` keys.
pub fn launch_radix_scatter_u64(
    client: &Client,
    keys_in: &Handle,
    keys_len: usize,
    indices_in: &Handle,
    scatter_offsets: &Handle,
    out_keys: &Handle,
    out_indices: &Handle,
    blocks: usize,
    grid: usize,
    shift: u64,
) -> Result<(), KernelError> {
    let count = CubeCount::Static(u32::try_from(blocks).map_err(|_| KernelError::RowCount)?, 1, 1);
    unsafe {
        inner::radix_scatter_u64::launch(
            client,
            count,
            CubeDim::new_1d(1),
            wrap(keys_in, keys_len),
            wrap(indices_in, keys_len),
            wrap(scatter_offsets, blocks * RADIX_BINS as usize),
            wrap(out_keys, keys_len),
            wrap(out_indices, keys_len),
            blocks,
            grid,
            shift,
        );
    }
    Ok(())
}

/// Fill an index buffer with `0..values_len` on the device.
pub fn launch_fill_index(
    client: &Client,
    out: &Handle,
    values_len: usize,
) -> Result<(), KernelError> {
    let Some(count) = count_for(values_len)? else { return Ok(()) };
    unsafe {
        inner::fill_index_kernel::launch(client, count, dim(), wrap(out, values_len));
    }
    Ok(())
}

/// Sign-flip each u32 so signed i32 values sort correctly as unsigned.
pub fn launch_sign_flip_u32(
    client: &Client,
    values: &Handle,
    values_len: usize,
    out: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(values_len)? else { return Ok(()) };
    unsafe {
        inner::sign_flip_u32_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(out, values_len),
        );
    }
    Ok(())
}

/// Device-side prefix-sum of a per-block histogram into scatter offsets.
///
/// Replaces the CPU `read_one` + host `prefix_sum_scatter_offsets` +
/// `create_from_slice` round-trip per sort pass. Single-thread: runs in
/// microseconds even for large `blocks`, and eliminates the per-pass CUDA
/// sync that made the old path 3× slower than host sort.
///
/// `histograms` and `scatter_offsets` must both have capacity for
/// `blocks * 256` `u32` words.
pub fn launch_radix_offsets(
    client: &Client,
    histograms: &Handle,
    scatter_offsets: &Handle,
    blocks: usize,
) -> Result<(), KernelError> {
    let count = CubeCount::Static(1, 1, 1);
    unsafe {
        inner::radix_offsets_kernel::launch(
            client,
            count,
            CubeDim::new_1d(1),
            wrap(histograms, blocks * RADIX_BINS as usize),
            wrap(scatter_offsets, blocks * RADIX_BINS as usize),
            blocks,
        );
    }
    Ok(())
}

/// Sign-flip each u64 so signed i64 values sort correctly as unsigned.
pub fn launch_sign_flip_u64(
    client: &Client,
    values: &Handle,
    values_len: usize,
    out: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(values_len)? else { return Ok(()) };
    unsafe {
        inner::sign_flip_u64_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(out, values_len),
        );
    }
    Ok(())
}

/// Bitwise NOT each u32 element. Reverses unsigned sort order: used after
/// sign-flip (signed) or on raw unsigned keys to turn GPU radix sort's
/// ascending output into descending order.
pub fn launch_not_u32(
    client: &Client,
    values: &Handle,
    values_len: usize,
    out: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(values_len)? else { return Ok(()) };
    unsafe {
        inner::not_u32_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(out, values_len),
        );
    }
    Ok(())
}

/// Same as [`launch_not_u32`] but for u64.
pub fn launch_not_u64(
    client: &Client,
    values: &Handle,
    values_len: usize,
    out: &Handle,
) -> Result<(), KernelError> {
    let Some(count) = count_for(values_len)? else { return Ok(()) };
    unsafe {
        inner::not_u64_kernel::launch(
            client,
            count,
            dim(),
            wrap(values, values_len),
            wrap(out, values_len),
        );
    }
    Ok(())
}

mod inner {
    use cubecl::prelude::*;

    /// Comparison operators for [`launch_cmp`](crate::launch_cmp).
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

    /// Boolean combination operators for [`launch_logic`](crate::launch_logic).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum LogicOp {
        /// Logical and.
        And,
        /// Logical or.
        Or,
    }

    /// Binary arithmetic operators for [`launch_binop`](crate::launch_binop).
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
        /// `left %% right` (modulus; remainder has the sign of the dividend).
        Mod,
    }

    /// Unary (single-argument) elementwise operators on a column.
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

    /// Compare one row against the literal; null rows never pass.
    ///
    /// These generic kernels serve integer and boolean types, where
    /// comparisons are native and exact. Float comparisons use the concrete
    /// kernels below: the `x != x` NaN idiom is unsound under fast-math
    /// folding (observed: wgpu compiles it to `false`), so floats detect
    /// `NaN` with the proper `.is_nan()` op.
    #[cube(launch)]
    pub fn cmp_eq_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] == lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    #[cube(launch)]
    pub fn cmp_ne_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] != lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    #[cube(launch)]
    pub fn cmp_lt_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] < lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    #[cube(launch)]
    pub fn cmp_le_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] <= lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    #[cube(launch)]
    pub fn cmp_gt_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] > lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    #[cube(launch)]
    pub fn cmp_ge_kernel<T: Numeric>(values: &[T], valid: &[u8], lit: &[T], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            let passed = if valid[row] != 0u8 { values[row] >= lit[0] } else { false };
            out[row] = passed as u8;
        }
    }

    /// Float comparison under the total order Polars uses (`NaN` sorts above
    /// `+inf`, `-0.0 == +0.0`). `NaN` is detected with `.is_nan()`, which
    /// survives fast-math folding that would break the `x != x` idiom. Each
    /// arm spells out its selection inline so macro hygiene cannot break it.
    macro_rules! cmp_float_kernel {
        ($name:ident, $ty:ty, Eq) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        let lt = (lhs < rhs) || (!lhs.is_nan() && rhs.is_nan());
                        let gt = (rhs < lhs) || (!rhs.is_nan() && lhs.is_nan());
                        (!(lt || gt))
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
        ($name:ident, $ty:ty, NotEq) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        let lt = (lhs < rhs) || (!lhs.is_nan() && rhs.is_nan());
                        let gt = (rhs < lhs) || (!rhs.is_nan() && lhs.is_nan());
                        ((lt) || (gt))
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
        ($name:ident, $ty:ty, Lt) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        (lhs < rhs) || (!lhs.is_nan() && rhs.is_nan())
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
        ($name:ident, $ty:ty, LtEq) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        let gt = (rhs < lhs) || (!rhs.is_nan() && lhs.is_nan());
                        (!(gt))
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
        ($name:ident, $ty:ty, Gt) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        (rhs < lhs) || (!rhs.is_nan() && lhs.is_nan())
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
        ($name:ident, $ty:ty, GtEq) => {
            #[cube(launch)]
            pub fn $name(values: &[$ty], valid: &[u8], lit: &[$ty], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    let passed = if valid[row] != 0u8 {
                        let lhs = values[row];
                        let rhs = lit[0];
                        let lt = (lhs < rhs) || (!lhs.is_nan() && rhs.is_nan());
                        (!(lt))
                    } else {
                        false
                    };
                    out[row] = passed as u8;
                }
            }
        };
    }

    cmp_float_kernel!(cmp_f32_eq_kernel, f32, Eq);
    cmp_float_kernel!(cmp_f32_ne_kernel, f32, NotEq);
    cmp_float_kernel!(cmp_f32_lt_kernel, f32, Lt);
    cmp_float_kernel!(cmp_f32_le_kernel, f32, LtEq);
    cmp_float_kernel!(cmp_f32_gt_kernel, f32, Gt);
    cmp_float_kernel!(cmp_f32_ge_kernel, f32, GtEq);
    cmp_float_kernel!(cmp_f64_eq_kernel, f64, Eq);
    cmp_float_kernel!(cmp_f64_ne_kernel, f64, NotEq);
    cmp_float_kernel!(cmp_f64_lt_kernel, f64, Lt);
    cmp_float_kernel!(cmp_f64_le_kernel, f64, LtEq);
    cmp_float_kernel!(cmp_f64_gt_kernel, f64, Gt);
    cmp_float_kernel!(cmp_f64_ge_kernel, f64, GtEq);

    /// Combine two masks elementwise: a null input row never passes.
    macro_rules! logic_kernel {
        ($name:ident, $op:tt) => {
            #[cube(launch)]
            pub fn $name(lhs: &[u8], rhs: &[u8], out: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    out[row] = ((lhs[row] != 0u8) $op (rhs[row] != 0u8)) as u8;
                }
            }
        };
    }

    logic_kernel!(logic_and_kernel, &&);
    logic_kernel!(logic_or_kernel, ||);

    #[cube(launch)]
    pub fn not_kernel(input: &[u8], out: &mut [u8]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = (input[row] == 0u8) as u8;
        }
    }

    #[cube(launch)]
    pub fn count_kernel(mask: &[u8], counts: &mut [u32], #[comptime] rows_per_thread: usize) {
        let thread = ABSOLUTE_POS;
        if thread < counts.len() {
            let mut local = 0u32;
            let mut j = 0usize;
            while j < rows_per_thread {
                let row = thread * rows_per_thread + j;
                if row < mask.len() && mask[row] != 0u8 {
                    local += 1u32;
                }
                j += 1usize;
            }
            counts[thread] = local;
        }
    }

    #[cube(launch)]
    pub fn scatter_kernel(
        mask: &[u8],
        bases: &[u32],
        indices: &mut [u32],
        #[comptime] rows_per_thread: usize,
    ) {
        let thread = ABSOLUTE_POS;
        let mut local = 0u32;
        let mut j = 0usize;
        while j < rows_per_thread {
            let row = thread * rows_per_thread + j;
            if row < mask.len() && mask[row] != 0u8 {
                indices[(bases[thread] + local) as usize] = row as u32;
                local += 1u32;
            }
            j += 1usize;
        }
    }

    #[cube(launch)]
    pub fn gather_kernel<T: Numeric>(values: &[T], indices: &[u32], out: &mut [T]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = values[indices[row] as usize];
        }
    }

    /// Fused gather of both values and validity through `indices` — one
    /// launch instead of two.
    #[cube(launch)]
    pub fn gather_values_mask_kernel<T: Numeric>(
        values: &[T],
        validity: &[u8],
        indices: &[u32],
        out_values: &mut [T],
        out_validity: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out_values.len() {
            let src = indices[row] as usize;
            out_values[row] = values[src];
            out_validity[row] = validity[src];
        }
    }

    #[cube(launch)]
    pub fn slice_kernel<T: Numeric>(values: &[T], out: &mut [T], #[comptime] offset: usize) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = values[row + offset];
        }
    }

    #[cube(launch)]
    pub fn cast_i32_to_f64_kernel(values: &[i32], out: &mut [f64]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = f64::cast_from(values[row]);
        }
    }

    #[cube(launch)]
    pub fn cast_u32_to_f64_kernel(values: &[u32], out: &mut [f64]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = f64::cast_from(values[row]);
        }
    }

    #[cube(launch)]
    pub fn group_count_kernel(groups: &[u32], counts: &[Atomic<u32>]) {
        let row = ABSOLUTE_POS;
        if row < groups.len() {
            counts[groups[row] as usize].fetch_add(1u32);
        }
    }

    /// GPU histogram: count occurrences of raw `i32` keys directly into buckets.
    ///
    /// `keys[row] - min_val` gives the bucket index; `Atomic::fetch_add`
    /// accumulates the per-bucket counts without host-side grouping. Small-range
    /// integer keys (e.g. `index % 64` → 64 buckets) avoid the key-column download
    /// entirely: the host only reads back `num_buckets * 4` bytes of counts.
    #[cube(launch)]
    pub fn gpu_histogram_kernel(keys: &[i32], min_val: &[i32], counts: &[Atomic<u32>]) {
        let row = ABSOLUTE_POS;
        if row < keys.len() {
            let bucket = keys[row] - min_val[0];
            counts[bucket as usize].fetch_add(1u32);
        }
    }

    /// Fused histogram + group-id assignment: computes both per-row group IDs
    /// and bucket counts in a single launch.
    ///
    /// CubeK-style kernel fusion: `gpu_histogram_kernel` + `keys_to_group_i32_kernel`
    /// read the same `keys[row]` and subtract the same `min_val[0]`. Fusing them
    /// saves one entire CUDA launch (~500µs) per groupBy call by eliminating
    /// the redundant key read and launch overhead.
    #[cube(launch)]
    pub fn gpu_histogram_and_assign_kernel(
        keys: &[i32],
        min_val: &[i32],
        counts: &[Atomic<u32>],
        out: &mut [u32],
    ) {
        let row = ABSOLUTE_POS;
        if row < keys.len() {
            let bucket = keys[row] - min_val[0];
            counts[bucket as usize].fetch_add(1u32);
            out[row] = bucket as u32;
        }
    }

    /// Compute per-row group IDs from raw `i32` keys: `out[row] = (keys[row] - min) as u32`.
    ///
    /// This runs entirely on the GPU and avoids downloading the key column for
    /// groupBy with small-range integer keys. The output is the dense group-id
    /// vector used by [`launch_group_invert`](crate::launch_group_invert).
    #[cube(launch)]
    pub fn keys_to_group_i32_kernel(keys: &[i32], min_val: &[i32], out: &mut [u32]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = (keys[row] - min_val[0]) as u32;
        }
    }

    #[cube(launch)]
    pub fn group_invert_kernel(
        groups: &[u32],
        seg_off: &[u32],
        seq: &[Atomic<u32>],
        slots: &mut [u32],
    ) {
        let row = ABSOLUTE_POS;
        if row < groups.len() {
            let group = groups[row] as usize;
            let position = seq[group].fetch_add(1u32);
            slots[(seg_off[group] + position) as usize] = row as u32;
        }
    }

    /// Parallel segmented sum via scratch-aggregate.
    ///
    /// Unlike `reduce_sum_kernel` (one thread per group — catastrophically
    /// slow for low-cardinality groupBy like 64 groups over 1M rows), this
    /// kernel launches `num_threads_per_group` threads per group. All threads
    /// each accumulate a strided chunk of their group's values, then write
    /// their partial sums into a per-group scratch buffer. Thread 0 then
    /// sums the scratch slots and writes the final output.
    ///
    /// Scratch layout: `scratch[group * num_threads_per_group + lane]` is the
    /// partial sum written by lane L of group G.  Host must zero-fill the
    /// scratch before launch.
    #[cube(launch)]
    pub fn reduce_sum_parallel_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [T],
        out_valid: &mut [u8],
        scratch: &mut [T],
        #[comptime] num_threads_per_group: usize,
    ) {
        let abs_pos = ABSOLUTE_POS;
        let group = abs_pos / num_threads_per_group;
        let lane = abs_pos % num_threads_per_group;
        let base = seg_off[group] as usize;
        let len = seg_len[group] as usize;
        let zero = T::from_int(0);

        // Each thread accumulates a strided chunk of its group's values.
        let mut acc = zero;
        let mut k = lane;
        while k < len {
            let row = slots[base + k] as usize;
            if valid[row] != 0u8 {
                acc += values[row];
            }
            k += num_threads_per_group;
        }

        // Write partial sum to this lane's exclusive slot.
        scratch[group * num_threads_per_group + lane] = acc;

        // Thread 0 sums all lanes' partial sums.
        // Within a single CubeCL launch, global-memory writes from any thread
        // propagate through L2 and are visible to all other threads because
        // a CubeCL cube is guaranteed to share a single L2 domain.
        if lane == 0usize {
            let group_base = group * num_threads_per_group;
            let mut total = zero;
            let mut t = 0usize;
            while t < num_threads_per_group {
                total += scratch[group_base + t];
                t += 1usize;
            }
            if len > 0usize {
                out_values[group] = total;
                out_valid[group] = 1u8;
            } else {
                out_values[group] = zero;
                out_valid[group] = 0u8;
            }
        }
    }

    /// Sum directly from the per-row group-assignment vector, bypassing
    /// the group_invert + slots indirection. 4× unrolled to reduce block count.
    ///
    /// `out_values` must be zero-filled by the host before launch.
    ///
    /// Polars semantics: the sum of an all-null group is `0` (not null), so
    /// every group's output is valid.
    #[cube(launch)]
    pub fn reduce_sum_direct_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        assignment: &[u32],
        out_values: &[Atomic<T>],
        out_valid: &mut [u8],
        num_groups: usize,
    ) {
        // Initialize all groups as valid (Polars: sum of all-null = 0).
        let g = ABSOLUTE_POS;
        if g < num_groups {
            out_valid[g] = 1u8;
        }
        // Accumulate values using the deterministic per-row assignment.
        // 4 rows per thread (stride-4 unrolling for reduced block count).
        let row_base = ABSOLUTE_POS * 4usize;
        let len = assignment.len();
        for offset in 0..4usize {
            let row = row_base + offset;
            if row < len {
                let group = assignment[row] as usize;
                if group < num_groups && valid[row] != 0u8 {
                    out_values[group].fetch_add(values[row]);
                }
            }
        }
    }

    /// Serial segmented sum for the group owned by this unit.
    ///
    /// Polars semantics: the sum of an all-null group is `0` (not null), so
    /// `out_valid` is always 1. This matches the CPU engine on Polars 0.55.2.
    #[cube(launch)]
    pub fn reduce_sum_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut acc = T::from_int(0);
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    acc += values[row];
                }
                k += 1usize;
            }
            out_values[group] = acc;
            out_valid[group] = 1u8;
        }
    }

    /// Serial segmented minimum for the group owned by this unit.
    ///
    /// Serves integer and boolean types, where every valid value
    /// participates. Float minimums use the concrete kernels below (`NaN`
    /// detection via `.is_nan()`, which survives fast-math folding).
    #[cube(launch)]
    pub fn reduce_min_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut found = false;
            let mut best = T::from_int(0);
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    let value = values[row];
                    if (!found) || (value < best) {
                        best = value;
                        found = true;
                    }
                }
                k += 1usize;
            }
            out_values[group] = best;
            out_valid[group] = found as u8;
        }
    }

    /// Fused global min+max without slot indirection — reads `values[k]`
    /// directly within the contiguous segment `[base, base+len)`.
    ///
    /// CubeK-style optimization: avoid the indirect `values[slots[base+k]]`
    /// read when the kernel owns a contiguous range (the common case for
    /// GPU histogram grouping, where the entire column is one segment).
    /// Saves one global memory read per element.
    #[cube(launch)]
    pub fn reduce_min_max_direct_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        seg_off: u32,
        seg_len: u32,
        out_min: &mut [T],
        out_max: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < out_min.len() {
            let mut found = false;
            let mut best_min = T::from_int(0);
            let mut best_max = T::from_int(0);
            let base = seg_off as usize;
            let len = seg_len as usize;
            let mut k = 0usize;
            while k < len {
                let row = base + k;
                if valid[row] != 0u8 {
                    let value = values[row];
                    if found {
                        if value < best_min {
                            best_min = value;
                        }
                        if best_max < value {
                            best_max = value;
                        }
                    } else {
                        best_min = value;
                        best_max = value;
                        found = true;
                    }
                }
                k += 1usize;
            }
            out_min[group] = best_min;
            out_max[group] = best_max;
            out_valid[group] = found as u8;
        }
    }

    /// Fused float global min+max without slot indirection — NaN-safe macro.
    macro_rules! reduce_minmax_float_direct_kernel {
        ($name:ident, $ty:ty) => {
            #[cube(launch)]
            pub fn $name(
                values: &[$ty],
                valid: &[u8],
                seg_off: u32,
                seg_len: u32,
                out_min: &mut [$ty],
                out_max: &mut [$ty],
                out_valid: &mut [u8],
            ) {
                let group = ABSOLUTE_POS;
                if group < out_min.len() {
                    let mut found_num = false;
                    let mut found_nan = false;
                    let zero = <$ty as Numeric>::from_int(0);
                    let mut best_min = zero;
                    let mut best_max = zero;
                    let mut nan_row = 0usize;
                    let base = seg_off as usize;
                    let len = seg_len as usize;
                    let mut k = 0usize;
                    while k < len {
                        let row = base + k;
                        if valid[row] != 0u8 {
                            let value = values[row];
                            if !value.is_nan() {
                                if !found_num {
                                    best_min = value;
                                    best_max = value;
                                    found_num = true;
                                } else {
                                    if value < best_min {
                                        best_min = value;
                                    }
                                    if best_max < value {
                                        best_max = value;
                                    }
                                }
                            } else {
                                nan_row = row;
                                found_nan = true;
                            }
                        }
                        k += 1usize;
                    }
                    if found_num {
                        out_min[group] = best_min;
                        out_max[group] = best_max;
                        out_valid[group] = 1u8;
                    } else if found_nan {
                        out_min[group] = values[nan_row];
                        out_max[group] = values[nan_row];
                        out_valid[group] = 1u8;
                    } else {
                        out_min[group] = zero;
                        out_max[group] = zero;
                        out_valid[group] = 0u8;
                    }
                }
            }
        };
    }

    reduce_minmax_float_direct_kernel!(reduce_min_max_direct_f32_kernel, f32);
    reduce_minmax_float_direct_kernel!(reduce_min_max_direct_f64_kernel, f64);

    /// Segmented min+max for the group owned by this unit, with slot indirection.
    ///
    /// Computes both min and max in a single pass over the segment's slots,
    /// halving the kernel launches when both are needed. Used when the segment
    /// is not contiguous in the value array (post group-invert permutation).
    #[cube(launch)]
    pub fn reduce_min_max_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_min: &mut [T],
        out_max: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut found = false;
            let mut best_min = T::from_int(0);
            let mut best_max = T::from_int(0);
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    let value = values[row];
                    if found {
                        if value < best_min {
                            best_min = value;
                        }
                        if best_max < value {
                            best_max = value;
                        }
                    } else {
                        best_min = value;
                        best_max = value;
                        found = true;
                    }
                }
                k += 1usize;
            }
            out_min[group] = best_min;
            out_max[group] = best_max;
            out_valid[group] = found as u8;
        }
    }

    /// Fused float segmented min+max with NaN handling.
    ///
    /// `NaN` values are skipped via `.is_nan()` (fast-math-safe); an all-`NaN`
    /// segment yields copied input `NaN` like the CPU engine.
    macro_rules! reduce_minmax_float_fused_kernel {
        ($name:ident, $ty:ty) => {
            #[cube(launch)]
            pub fn $name(
                values: &[$ty],
                valid: &[u8],
                slots: &[u32],
                seg_off: &[u32],
                seg_len: &[u32],
                out_min: &mut [$ty],
                out_max: &mut [$ty],
                out_valid: &mut [u8],
            ) {
                let group = ABSOLUTE_POS;
                if group < seg_len.len() {
                    let mut found_num = false;
                    let mut found_nan = false;
                    let zero = <$ty as Numeric>::from_int(0);
                    let mut best_min = zero;
                    let mut best_max = zero;
                    let mut nan_row = 0usize;
                    let base = seg_off[group] as usize;
                    let len = seg_len[group] as usize;
                    let mut k = 0usize;
                    while k < len {
                        let row = slots[base + k] as usize;
                        if valid[row] != 0u8 {
                            let value = values[row];
                            if !value.is_nan() {
                                if !found_num {
                                    best_min = value;
                                    best_max = value;
                                    found_num = true;
                                } else {
                                    if value < best_min {
                                        best_min = value;
                                    }
                                    if best_max < value {
                                        best_max = value;
                                    }
                                }
                            } else {
                                nan_row = row;
                                found_nan = true;
                            }
                        }
                        k += 1usize;
                    }
                    if found_num {
                        out_min[group] = best_min;
                        out_max[group] = best_max;
                        out_valid[group] = 1u8;
                    } else if found_nan {
                        out_min[group] = values[nan_row];
                        out_max[group] = values[nan_row];
                        out_valid[group] = 1u8;
                    } else {
                        out_min[group] = zero;
                        out_max[group] = zero;
                        out_valid[group] = 0u8;
                    }
                }
            }
        };
    }

    reduce_minmax_float_fused_kernel!(reduce_min_max_f32_kernel, f32);
    reduce_minmax_float_fused_kernel!(reduce_min_max_f64_kernel, f64);

    /// Serial segmented maximum for the group owned by this unit.
    ///
    /// Serves integer and boolean types; float maximums use the concrete
    /// kernels below.
    #[cube(launch)]
    pub fn reduce_max_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut found = false;
            let mut best = T::from_int(0);
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    let value = values[row];
                    if (!found) || (best < value) {
                        best = value;
                        found = true;
                    }
                }
                k += 1usize;
            }
            out_values[group] = best;
            out_valid[group] = found as u8;
        }
    }

    /// Float segmented minimum/maximum for the group owned by this unit.
    ///
    /// `NaN` values are skipped via `.is_nan()` (fast-math-safe); an
    /// all-`NaN` segment yields a copied input `NaN` like the CPU engine, an
    /// empty one yields null.
    macro_rules! reduce_minmax_float_kernel {
        ($name:ident, $ty:ty, Min) => {
            #[cube(launch)]
            pub fn $name(
                values: &[$ty],
                valid: &[u8],
                slots: &[u32],
                seg_off: &[u32],
                seg_len: &[u32],
                out_values: &mut [$ty],
                out_valid: &mut [u8],
            ) {
                let group = ABSOLUTE_POS;
                if group < seg_len.len() {
                    let mut found_num = false;
                    let mut found_nan = false;
                    let zero = <$ty as Numeric>::from_int(0);
                    let mut best = zero;
                    let mut nan_row = 0usize;
                    let base = seg_off[group] as usize;
                    let len = seg_len[group] as usize;
                    let mut k = 0usize;
                    while k < len {
                        let row = slots[base + k] as usize;
                        if valid[row] != 0u8 {
                            let value = values[row];
                            if !value.is_nan() {
                                if (!found_num) || (value < best) {
                                    best = value;
                                    found_num = true;
                                }
                            } else {
                                nan_row = row;
                                found_nan = true;
                            }
                        }
                        k += 1usize;
                    }
                    if found_num {
                        out_values[group] = best;
                        out_valid[group] = 1u8;
                    } else if found_nan {
                        out_values[group] = values[nan_row];
                        out_valid[group] = 1u8;
                    } else {
                        out_values[group] = zero;
                        out_valid[group] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, Max) => {
            #[cube(launch)]
            pub fn $name(
                values: &[$ty],
                valid: &[u8],
                slots: &[u32],
                seg_off: &[u32],
                seg_len: &[u32],
                out_values: &mut [$ty],
                out_valid: &mut [u8],
            ) {
                let group = ABSOLUTE_POS;
                if group < seg_len.len() {
                    let mut found_num = false;
                    let mut found_nan = false;
                    let zero = <$ty as Numeric>::from_int(0);
                    let mut best = zero;
                    let mut nan_row = 0usize;
                    let base = seg_off[group] as usize;
                    let len = seg_len[group] as usize;
                    let mut k = 0usize;
                    while k < len {
                        let row = slots[base + k] as usize;
                        if valid[row] != 0u8 {
                            let value = values[row];
                            if !value.is_nan() {
                                if (!found_num) || (best < value) {
                                    best = value;
                                    found_num = true;
                                }
                            } else {
                                nan_row = row;
                                found_nan = true;
                            }
                        }
                        k += 1usize;
                    }
                    if found_num {
                        out_values[group] = best;
                        out_valid[group] = 1u8;
                    } else if found_nan {
                        out_values[group] = values[nan_row];
                        out_valid[group] = 1u8;
                    } else {
                        out_values[group] = zero;
                        out_valid[group] = 0u8;
                    }
                }
            }
        };
    }

    reduce_minmax_float_kernel!(reduce_min_f32_kernel, f32, Min);
    reduce_minmax_float_kernel!(reduce_max_f32_kernel, f32, Max);
    reduce_minmax_float_kernel!(reduce_min_f64_kernel, f64, Min);
    reduce_minmax_float_kernel!(reduce_max_f64_kernel, f64, Max);

    /// Serial segmented mean for the group owned by this unit.
    #[cube(launch)]
    pub fn reduce_mean_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut acc = 0.0f64;
            let mut count = 0u32;
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    acc += f64::cast_from(values[row]);
                    count += 1u32;
                }
                k += 1usize;
            }
            if count > 0u32 {
                out_values[group] = acc / f64::cast_from(count);
                out_valid[group] = 1u8;
            } else {
                out_values[group] = 0.0f64;
                out_valid[group] = 0u8;
            }
        }
    }

    /// Serial segmented valid-count for the group owned by this unit.
    #[cube(launch)]
    pub fn reduce_count_kernel(
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [u32],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut count = 0u32;
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    count += 1u32;
                }
                k += 1usize;
            }
            out_values[group] = count;
            out_valid[group] = 1u8;
        }
    }

    /// Serial segmented product for the group owned by this unit.
    ///
    /// Empty segments yield identity `1` with valid output (matches the CPU
    /// engine: the product of no elements is `1`).
    #[cube(launch)]
    pub fn reduce_product_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        out_values: &mut [T],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let mut acc = T::from_int(1);
            let mut count = 0u32;
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    acc *= values[row];
                    count += 1u32;
                }
                k += 1usize;
            }
            out_values[group] = acc;
            // Identity `1` for empty groups (matches CPU polars).
            out_valid[group] = if count > 0u32 || len == 0 { 1u8 } else { 0u8 };
        }
    }

    /// Serial segmented standard deviation for the group owned by this unit.
    ///
    /// Two-pass: compute mean, then sum of squared deltas, finally
    /// `sqrt(ss / (n - ddof))`. Output is `f64`; segments with fewer than
    /// `ddof + 1` valid values yield null.
    #[cube(launch)]
    pub fn reduce_std_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        ddof: u8,
        out_values: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            // Pass 1: compute mean in f64.
            let mut sum = 0.0f64;
            let mut count = 0u32;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    sum += f64::cast_from(values[row]);
                    count += 1u32;
                }
                k += 1usize;
            }
            if count > ddof as u32 {
                let mean = sum / f64::cast_from(count);
                // Pass 2: compute sum of squared deltas.
                let mut ss = 0.0f64;
                k = 0usize;
                while k < len {
                    let row = slots[base + k] as usize;
                    if valid[row] != 0u8 {
                        let delta = f64::cast_from(values[row]) - mean;
                        ss += delta * delta;
                    }
                    k += 1usize;
                }
                let denom = f64::cast_from(count - ddof as u32);
                out_values[group] = f64::sqrt(ss / denom);
                out_valid[group] = 1u8;
            } else {
                out_values[group] = 0.0f64;
                out_valid[group] = 0u8;
            }
        }
    }

    /// Serial segmented variance for the group owned by this unit.
    ///
    /// Two-pass: compute mean, then sum of squared deltas, finally
    /// `ss / (n - ddof)`. Output is `f64`; segments with fewer than
    /// `ddof + 1` valid values yield null.
    #[cube(launch)]
    pub fn reduce_var_kernel<T: Numeric>(
        values: &[T],
        valid: &[u8],
        slots: &[u32],
        seg_off: &[u32],
        seg_len: &[u32],
        ddof: u8,
        out_values: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let group = ABSOLUTE_POS;
        if group < seg_len.len() {
            let base = seg_off[group] as usize;
            let len = seg_len[group] as usize;
            // Pass 1: compute mean in f64.
            let mut sum = 0.0f64;
            let mut count = 0u32;
            let mut k = 0usize;
            while k < len {
                let row = slots[base + k] as usize;
                if valid[row] != 0u8 {
                    sum += f64::cast_from(values[row]);
                    count += 1u32;
                }
                k += 1usize;
            }
            if count > ddof as u32 {
                let mean = sum / f64::cast_from(count);
                // Pass 2: compute sum of squared deltas.
                let mut ss = 0.0f64;
                k = 0usize;
                while k < len {
                    let row = slots[base + k] as usize;
                    if valid[row] != 0u8 {
                        let delta = f64::cast_from(values[row]) - mean;
                        ss += delta * delta;
                    }
                    k += 1usize;
                }
                let denom = f64::cast_from(count - ddof as u32);
                out_values[group] = ss / denom;
                out_valid[group] = 1u8;
            } else {
                out_values[group] = 0.0f64;
                out_valid[group] = 0u8;
            }
        }
    }

    // --- Binary arithmetic (column vs column) ----------------------------

    /// Generic elementwise binary op for integer types.
    ///
    /// Null in either operand yields a null output row. Division-by-zero on
    /// integer types yields `0` with a null output row, matching Polars.
    macro_rules! binop_int_kernel {
        ($name:ident, $op:tt) => {
            #[cube(launch)]
            pub fn $name<T: Numeric>(
                lhs: &[T],
                lhs_valid: &[u8],
                rhs: &[T],
                rhs_valid: &[u8],
                out: &mut [T],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        out[row] = lhs[row] $op rhs[row];
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = T::from_int(0);
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    // Division/modulus need zero-check on integers (no SIGFPE trap).
    #[cube(launch)]
    pub fn binop_div_kernel<T: Numeric>(
        lhs: &[T],
        lhs_valid: &[u8],
        rhs: &[T],
        rhs_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 && rhs[row] != T::from_int(0) {
                out[row] = lhs[row] / rhs[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn binop_floordiv_kernel<T: Numeric>(
        lhs: &[T],
        lhs_valid: &[u8],
        rhs: &[T],
        rhs_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 && rhs[row] != T::from_int(0) {
                let q = lhs[row] / rhs[row];
                let r = lhs[row] % rhs[row];
                // Floor correction: when signs differ and remainder != 0.
                if r != T::from_int(0) &&
                    ((lhs[row] < T::from_int(0)) != (rhs[row] < T::from_int(0)))
                {
                    out[row] = q - T::from_int(1);
                } else {
                    out[row] = q;
                }
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn binop_mod_kernel<T: Numeric>(
        lhs: &[T],
        lhs_valid: &[u8],
        rhs: &[T],
        rhs_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 && rhs[row] != T::from_int(0) {
                out[row] = lhs[row] % rhs[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    binop_int_kernel!(binop_add_kernel, +);
    binop_int_kernel!(binop_sub_kernel, -);
    binop_int_kernel!(binop_mul_kernel, *);

    /// Concrete `f32` binary op kernels. `Div`/`Mod` are guarded for Floats
    /// (`NaN` propagation is the natural op; no SIGFPE, but still skip zero).
    macro_rules! binop_f32_kernel {
        ($name:ident, $op:tt) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[f32],
                lhs_valid: &[u8],
                rhs: &[f32],
                rhs_valid: &[u8],
                out: &mut [f32],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        out[row] = lhs[row] $op rhs[row];
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0.0f32;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    binop_f32_kernel!(binop_f32_add_kernel, +);
    binop_f32_kernel!(binop_f32_sub_kernel, -);
    binop_f32_kernel!(binop_f32_mul_kernel, *);
    binop_f32_kernel!(binop_f32_div_kernel, /);

    #[cube(launch)]
    pub fn binop_f32_floordiv_kernel(
        lhs: &[f32],
        lhs_valid: &[u8],
        rhs: &[f32],
        rhs_valid: &[u8],
        out: &mut [f32],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                let q = lhs[row] / rhs[row];
                out[row] = q.floor();
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f32;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn binop_f32_mod_kernel(
        lhs: &[f32],
        lhs_valid: &[u8],
        rhs: &[f32],
        rhs_valid: &[u8],
        out: &mut [f32],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                out[row] = lhs[row] % rhs[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f32;
                out_valid[row] = 0u8;
            }
        }
    }

    /// Concrete `f64` binary op kernels.
    macro_rules! binop_f64_kernel {
        ($name:ident, $op:tt) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[f64],
                lhs_valid: &[u8],
                rhs: &[f64],
                rhs_valid: &[u8],
                out: &mut [f64],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        out[row] = lhs[row] $op rhs[row];
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0.0f64;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    binop_f64_kernel!(binop_f64_add_kernel, +);
    binop_f64_kernel!(binop_f64_sub_kernel, -);
    binop_f64_kernel!(binop_f64_mul_kernel, *);
    binop_f64_kernel!(binop_f64_div_kernel, /);

    #[cube(launch)]
    pub fn binop_f64_floordiv_kernel(
        lhs: &[f64],
        lhs_valid: &[u8],
        rhs: &[f64],
        rhs_valid: &[u8],
        out: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                let q = lhs[row] / rhs[row];
                out[row] = q.floor();
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f64;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn binop_f64_mod_kernel(
        lhs: &[f64],
        lhs_valid: &[u8],
        rhs: &[f64],
        rhs_valid: &[u8],
        out: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                out[row] = lhs[row] % rhs[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f64;
                out_valid[row] = 0u8;
            }
        }
    }

    // --- Unary elementwise -------------------------------------------------

    /// Generic integer abs and sign.
    #[cube(launch)]
    pub fn unary_abs_kernel<T: Numeric>(
        input: &[T],
        input_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                let v = input[row];
                out[row] = if v < T::from_int(0) { T::from_int(0) - v } else { v };
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_negate_kernel<T: Numeric>(
        input: &[T],
        input_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                out[row] = T::from_int(0) - input[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_sign_kernel<T: Numeric>(
        input: &[T],
        input_valid: &[u8],
        out: &mut [T],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                let v = input[row];
                out[row] = if v > T::from_int(0) {
                    T::from_int(1)
                } else if v < T::from_int(0) {
                    T::from_int(0) - T::from_int(1)
                } else {
                    T::from_int(0)
                };
                out_valid[row] = 1u8;
            } else {
                out[row] = T::from_int(0);
                out_valid[row] = 0u8;
            }
        }
    }

    /// Float unary kernels (Ceil, Floor, Exp, Sqrt, Log1p, Sin, Cos, Abs,
    /// Sign, Negate) - per function to keep each kernel a straight line.
    macro_rules! unary_f32_unary_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            pub fn $name(input: &[f32], input_valid: &[u8], out: &mut [f32], out_valid: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if input_valid[row] != 0u8 {
                        out[row] = input[row].$method();
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0.0f32;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    unary_f32_unary_kernel!(unary_f32_ceil_kernel, ceil);
    unary_f32_unary_kernel!(unary_f32_floor_kernel, floor);
    unary_f32_unary_kernel!(unary_f32_exp_kernel, exp);
    unary_f32_unary_kernel!(unary_f32_sqrt_kernel, sqrt);
    unary_f32_unary_kernel!(unary_f32_log1p_kernel, log1p);
    unary_f32_unary_kernel!(unary_f32_sin_kernel, sin);
    unary_f32_unary_kernel!(unary_f32_cos_kernel, cos);

    #[cube(launch)]
    pub fn unary_f32_abs_kernel(
        input: &[f32],
        input_valid: &[u8],
        out: &mut [f32],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                out[row] = input[row].abs();
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f32;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_f32_sign_kernel(
        input: &[f32],
        input_valid: &[u8],
        out: &mut [f32],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                let v = input[row];
                out[row] = if v.is_nan() {
                    f32::NAN
                } else if v > 0.0f32 {
                    1.0f32
                } else if v < 0.0f32 {
                    0.0f32 - 1.0f32
                } else {
                    0.0f32
                };
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f32;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_f32_negate_kernel(
        input: &[f32],
        input_valid: &[u8],
        out: &mut [f32],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                out[row] = 0.0f32 - input[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f32;
                out_valid[row] = 0u8;
            }
        }
    }

    macro_rules! unary_f64_unary_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            pub fn $name(input: &[f64], input_valid: &[u8], out: &mut [f64], out_valid: &mut [u8]) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if input_valid[row] != 0u8 {
                        out[row] = input[row].$method();
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0.0f64;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    unary_f64_unary_kernel!(unary_f64_ceil_kernel, ceil);
    unary_f64_unary_kernel!(unary_f64_floor_kernel, floor);
    unary_f64_unary_kernel!(unary_f64_exp_kernel, exp);
    unary_f64_unary_kernel!(unary_f64_sqrt_kernel, sqrt);
    unary_f64_unary_kernel!(unary_f64_log1p_kernel, log1p);
    unary_f64_unary_kernel!(unary_f64_sin_kernel, sin);
    unary_f64_unary_kernel!(unary_f64_cos_kernel, cos);

    #[cube(launch)]
    pub fn unary_f64_abs_kernel(
        input: &[f64],
        input_valid: &[u8],
        out: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                out[row] = input[row].abs();
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f64;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_f64_sign_kernel(
        input: &[f64],
        input_valid: &[u8],
        out: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                let v = input[row];
                out[row] = if v.is_nan() {
                    f64::NAN
                } else if v > 0.0f64 {
                    1.0f64
                } else if v < 0.0f64 {
                    0.0f64 - 1.0f64
                } else {
                    0.0f64
                };
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f64;
                out_valid[row] = 0u8;
            }
        }
    }

    #[cube(launch)]
    pub fn unary_f64_negate_kernel(
        input: &[f64],
        input_valid: &[u8],
        out: &mut [f64],
        out_valid: &mut [u8],
    ) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            if input_valid[row] != 0u8 {
                out[row] = 0.0f64 - input[row];
                out_valid[row] = 1u8;
            } else {
                out[row] = 0.0f64;
                out_valid[row] = 0u8;
            }
        }
    }

    // --- Column-vs-column comparisons ------------------------------------

    /// Integer col_cmp kernels — each spelled out inline so macro hygiene
    /// keeps `lhs`/`rhs`/`row` accessible.
    macro_rules! col_cmp_int_kernel {
        ($name:ident, $op:tt) => {
            #[cube(launch)]
            pub fn $name<T: Numeric>(
                lhs: &[T],
                lhs_valid: &[u8],
                rhs: &[T],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let passed = lhs[row] $op rhs[row];
                        out[row] = if passed { 1u8 } else { 0u8 };
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    col_cmp_int_kernel!(col_cmp_eq_kernel, ==);
    col_cmp_int_kernel!(col_cmp_ne_kernel, !=);
    col_cmp_int_kernel!(col_cmp_lt_kernel, <);
    col_cmp_int_kernel!(col_cmp_le_kernel, <=);
    col_cmp_int_kernel!(col_cmp_gt_kernel, >);
    col_cmp_int_kernel!(col_cmp_ge_kernel, >=);

    /// Float col_cmp under the Polars total order (`NaN` sorts above
    /// `+inf`). Same `is_nan()`-based ordering as the literal comparison
    /// kernels above.
    macro_rules! col_cmp_float_kernel {
        ($name:ident, $ty:ty, Eq) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        let lt = (a < b) || (!a.is_nan() && b.is_nan());
                        let gt = (b < a) || (!b.is_nan() && a.is_nan());
                        out[row] = (!(lt || gt)) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, NotEq) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        let lt = (a < b) || (!a.is_nan() && b.is_nan());
                        let gt = (b < a) || (!b.is_nan() && a.is_nan());
                        out[row] = ((lt) || (gt)) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, Lt) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        out[row] = ((a < b) || (!a.is_nan() && b.is_nan())) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, LtEq) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        let gt = (b < a) || (!b.is_nan() && a.is_nan());
                        out[row] = (!(gt)) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, Gt) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        out[row] = ((b < a) || (!b.is_nan() && a.is_nan())) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
        ($name:ident, $ty:ty, GtEq) => {
            #[cube(launch)]
            pub fn $name(
                lhs: &[$ty],
                lhs_valid: &[u8],
                rhs: &[$ty],
                rhs_valid: &[u8],
                out: &mut [u8],
                out_valid: &mut [u8],
            ) {
                let row = ABSOLUTE_POS;
                if row < out.len() {
                    if lhs_valid[row] != 0u8 && rhs_valid[row] != 0u8 {
                        let a = lhs[row];
                        let b = rhs[row];
                        let lt = (a < b) || (!a.is_nan() && b.is_nan());
                        out[row] = (!(lt)) as u8;
                        out_valid[row] = 1u8;
                    } else {
                        out[row] = 0u8;
                        out_valid[row] = 0u8;
                    }
                }
            }
        };
    }

    col_cmp_float_kernel!(col_cmp_f32_eq_kernel, f32, Eq);
    col_cmp_float_kernel!(col_cmp_f32_ne_kernel, f32, NotEq);
    col_cmp_float_kernel!(col_cmp_f32_lt_kernel, f32, Lt);
    col_cmp_float_kernel!(col_cmp_f32_le_kernel, f32, LtEq);
    col_cmp_float_kernel!(col_cmp_f32_gt_kernel, f32, Gt);
    col_cmp_float_kernel!(col_cmp_f32_ge_kernel, f32, GtEq);
    col_cmp_float_kernel!(col_cmp_f64_eq_kernel, f64, Eq);
    col_cmp_float_kernel!(col_cmp_f64_ne_kernel, f64, NotEq);
    col_cmp_float_kernel!(col_cmp_f64_lt_kernel, f64, Lt);
    col_cmp_float_kernel!(col_cmp_f64_le_kernel, f64, LtEq);
    col_cmp_float_kernel!(col_cmp_f64_gt_kernel, f64, Gt);
    col_cmp_float_kernel!(col_cmp_f64_ge_kernel, f64, GtEq);

    // --- GPU LSD Radix Sort -------------------------------------------------

    /// Compute a per-block 256-bin histogram of the `shift`-th 8-bit digit of
    /// `keys`. Each of the `blocks` blocks produces one 256-word histogram;
    /// output is `blocks × 256` u32 words laid out `[block][bin]`.
    ///
    /// Per-thread local accumulation keeps the inner loop conflict-free:
    /// each thread reads a discontiguous slice of rows (stride = `grid`); the
    /// 256-bin counter lives in a `Array<u32>` (CubeCL's stack array).
    #[cube(launch)]
    pub fn radix_histogram_u32(
        keys: &[u32],
        histograms: &mut [u32],
        #[comptime] blocks: usize,
        #[comptime] grid: usize,
        #[comptime] shift: u32,
    ) {
        let block = ABSOLUTE_POS;
        if block < blocks {
            let base = grid * block;
            let end = base + grid;
            let mut hist = Array::<u32>::new(256usize);
            let mut row = base;
            while row < end {
                if row < keys.len() {
                    let digit = ((keys[row] >> shift) & 0xFF) as usize;
                    hist[digit] += 1u32;
                }
                row += 1usize;
            }
            let mut b = 0usize;
            while b < 256usize {
                histograms[block * 256 + b] = hist[b];
                b += 1usize;
            }
        }
    }

    /// Stable scatter of `(key, index)` pairs by the `shift`-th 8-bit digit.
    ///
    /// `scatter_offsets[block * 256 + digit]` is the global output index where
    /// this block's first element with the given digit goes. Each thread
    /// walks its rows in order, increments a per-digit local counter, and
    /// writes to `out_keys[out_pos]` / `out_indices[out_pos]`.
    ///
    /// The host must prefix-sum `scatter_offsets` before launching — it starts
    /// as a straight count prefix sum, then is patched per-block so that each
    /// block's counters start after the previous block's identical-digit range.
    #[cube(launch)]
    pub fn radix_scatter_u32(
        keys_in: &[u32],
        indices_in: &[u32],
        scatter_offsets: &[u32],
        out_keys: &mut [u32],
        out_indices: &mut [u32],
        #[comptime] blocks: usize,
        #[comptime] grid: usize,
        #[comptime] shift: u32,
    ) {
        let block = ABSOLUTE_POS;
        if block < blocks {
            let base = grid * block;
            let end = base + grid;
            let mut local = Array::<u32>::new(256usize);
            let mut row = base;
            while row < end {
                if row < keys_in.len() {
                    let digit = ((keys_in[row] >> shift) & 0xFF) as usize;
                    let off = scatter_offsets[block * 256 + digit];
                    let pos = off + local[digit];
                    out_keys[pos as usize] = keys_in[row];
                    out_indices[pos as usize] = indices_in[row];
                    local[digit] += 1u32;
                }
                row += 1usize;
            }
        }
    }

    /// Same as [`radix_histogram_u32`] but for 64-bit keys.
    #[cube(launch)]
    pub fn radix_histogram_u64(
        keys: &[u64],
        histograms: &mut [u32],
        #[comptime] blocks: usize,
        #[comptime] grid: usize,
        #[comptime] shift: u64,
    ) {
        let block = ABSOLUTE_POS;
        if block < blocks {
            let base = grid * block;
            let end = base + grid;
            let mut hist = Array::<u32>::new(256usize);
            let mut row = base;
            while row < end {
                if row < keys.len() {
                    let digit = ((keys[row] >> shift) & 0xFF) as usize;
                    hist[digit] += 1u32;
                }
                row += 1usize;
            }
            let mut b = 0usize;
            while b < 256usize {
                histograms[block * 256 + b] = hist[b];
                b += 1usize;
            }
        }
    }

    /// Same as [`radix_scatter_u32`] but for 64-bit keys.
    #[cube(launch)]
    pub fn radix_scatter_u64(
        keys_in: &[u64],
        indices_in: &[u32],
        scatter_offsets: &[u32],
        out_keys: &mut [u64],
        out_indices: &mut [u32],
        #[comptime] blocks: usize,
        #[comptime] grid: usize,
        #[comptime] shift: u64,
    ) {
        let block = ABSOLUTE_POS;
        if block < blocks {
            let base = grid * block;
            let end = base + grid;
            let mut local = Array::<u32>::new(256usize);
            let mut row = base;
            while row < end {
                if row < keys_in.len() {
                    let digit = ((keys_in[row] >> shift) & 0xFF) as usize;
                    let off = scatter_offsets[block * 256 + digit];
                    let pos = off + local[digit];
                    out_keys[pos as usize] = keys_in[row];
                    out_indices[pos as usize] = indices_in[row];
                    local[digit] += 1u32;
                }
                row += 1usize;
            }
        }
    }

    /// Fill an index buffer with `0, 1, 2, …, len-1`, used to initialise the
    /// permutation that the radix sort reorders.
    #[cube(launch)]
    pub fn fill_index_kernel(out: &mut [u32]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = row as u32;
        }
    }

    /// Flip the sign bit of each u32 so that signed i32 sort order is
    /// preserved when sorting as unsigned: negative i32 values become smaller
    /// u32, positive become larger, matching `i32::cmp` semantics.
    #[cube(launch)]
    pub fn sign_flip_u32_kernel(values: &[u32], out: &mut [u32]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = values[row] ^ (1u32 << 31);
        }
    }

    /// Same as [`sign_flip_u32_kernel`] but for u64.
    #[cube(launch)]
    pub fn sign_flip_u64_kernel(values: &[u64], out: &mut [u64]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = values[row] ^ (1u64 << 63);
        }
    }

    /// Bitwise NOT a u32 buffer into `out`. Reverses unsigned sort order:
    /// applied to sign-flipped signed keys (or raw unsigned keys) it turns
    /// ascending-order radix-sort output into descending order.
    #[cube(launch)]
    pub fn not_u32_kernel(values: &[u32], out: &mut [u32]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = !values[row];
        }
    }

    /// Same as [`not_u32_kernel`] but for u64.
    #[cube(launch)]
    pub fn not_u64_kernel(values: &[u64], out: &mut [u64]) {
        let row = ABSOLUTE_POS;
        if row < out.len() {
            out[row] = !values[row];
        }
    }

    /// Device-side prefix sum: convert per-block per-digit histogram counts
    /// into per-block per-digit scatter offsets.
    ///
    /// `histogram[block * 256 + digit]` holds the count of rows in `block`
    /// whose current digit equals `digit`. Output `scatter_offsets` receives
    /// the global output position where each block's first matching row is
    /// written during scatter.
    ///
    /// Single-thread: iterates `blocks * 256` elements sequentially — fast
    /// even at 100K blocks (~25M ops in <10 µs on GPU) and avoids forcing a
    /// full CUDA sync per pass. The topology matches `launch_radix_histogram_*`
    /// for uniform kernel indexing.
    #[cube(launch)]
    pub fn radix_offsets_kernel(
        histogram: &[u32],
        scatter_offsets: &mut [u32],
        #[comptime] blocks: usize,
    ) {
        let thread = ABSOLUTE_POS;
        if thread == 0 && blocks > 0 {
            let mut digit_total = Array::<u32>::new(256usize);
            let mut b = 0usize;
            while b < blocks {
                let mut digit = 0usize;
                while digit < 256usize {
                    digit_total[digit] += histogram[b * 256 + digit];
                    digit += 1usize;
                }
                b += 1usize;
            }
            // Exclusive prefix sum over digit_total → starting offset per digit.
            let mut digit_prefix = Array::<u32>::new(256usize);
            digit_prefix[0] = 0u32;
            let mut digit = 1usize;
            while digit < 256usize {
                digit_prefix[digit] = digit_prefix[digit - 1] + digit_total[digit - 1];
                digit += 1usize;
            }
            // Scatter offsets: per-block start = digit_prefix[digit] + sum of
            // previous blocks' counts for that digit.
            let mut running = Array::<u32>::new(256usize);
            let mut d = 0usize;
            while d < 256usize {
                running[d] = digit_prefix[d];
                d += 1usize;
            }
            let mut bb = 0usize;
            while bb < blocks {
                let mut digit = 0usize;
                while digit < 256usize {
                    scatter_offsets[bb * 256 + digit] = running[digit];
                    running[digit] += histogram[bb * 256 + digit];
                    digit += 1usize;
                }
                bb += 1usize;
            }
        }
    }
}
