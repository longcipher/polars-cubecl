//! Per-runtime allocation front-end for kernel scratch buffers.
//!
//! [`acquire`] and [`recycle`] are the two ends of the scratch-buffer
//! lifecycle. They front the backend's own allocation pool (`client.empty`
//! / `Handle::drop`) which is always safe and amortised. Cross-launch pooling
//! is disabled (see [`recycle`]) due to correctness issues on cubecl-cpu.

use cubecl_runtime::server::Handle;

/// Acquire a device buffer of `bytes` bytes.
///
/// Always goes through `client.empty` for a fresh allocation. Cross-launch
/// pooling used to live here but was disabled after a correctness bug on
/// the CPU backend (see [`recycle`]).
pub(crate) fn acquire(client: &cubecl::prelude::Client, bytes: usize) -> Handle {
    client.empty(bytes.max(1))
}

/// Rounds `bytes` up to the next power of two ≥ 64.
///
/// Test-only helper kept for any future pool re‑enable.
#[cfg(test)]
#[inline]
fn bucket_key(bytes: usize) -> usize {
    bytes.max(1).next_power_of_two().max(64)
}

/// Return a device buffer to the pool for future reuse. Callers should call
/// this on handles they would otherwise drop after an op finishes.
///
/// # Pool disabled
///
/// Device buffer pooling across kernel launches is disabled: cubecl-cpu
/// does not guarantee that reused buffers preserve correct contents for
/// subsequent reads, producing value mismatches in filter/gather pipelines
/// after a handful of plan invocations. Dropping the handle here (instead
/// of pooling) lets the backend reclaim the allocation via its own pool,
/// which keeps allocation amortised without the cross-launch correctness
/// risk.
pub(crate) fn recycle(handle: Handle, _bytes: usize) {
    drop(handle);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_key_rounds_up_and_clamps() {
        // 0 → max(1) → 1 → next_power_of_two = 1 → max(64) = 64
        assert_eq!(bucket_key(0), 64);
        // Small sizes clamp to the floor of 64.
        assert_eq!(bucket_key(1), 64);
        assert_eq!(bucket_key(63), 64);
        assert_eq!(bucket_key(64), 64);
        // Sizes above 64 round to next power of two.
        assert_eq!(bucket_key(65), 128);
        assert_eq!(bucket_key(1024), 1024);
        assert_eq!(bucket_key(1025), 2048);
        assert_eq!(bucket_key(1_000_000), 1_048_576);
    }
}
