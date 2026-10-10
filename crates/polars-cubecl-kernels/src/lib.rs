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
pub use inner::{CmpOp, LogicOp};
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

    /// Serial segmented sum for the group owned by this unit.
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
}
