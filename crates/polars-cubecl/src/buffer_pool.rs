//! Per-runtime buffer pool for kernel scratch buffers.
//!
//! Each `alloc()` in the executor goes through `client.empty()` (or
//! `client.create_from_slice()`) → a fresh device allocation. On CUDA, those
//! translate to `cudaMalloc`/`cudaFree` (~10-100 µs each under the CUDA
//! context lock). An operator pipeline allocates and frees between every step,
//! so we pool output buffers by rounded size and reuse them across ops and
//! across plans. Releasing a handle returns it to the pool; acquiring with a
//! matching size finds a warm buffer (only the `Handle` clone is cheap — no
//! GPU side work).
//!
//! # Safety invariant
//!
//! Each pool entry stores the *exact* byte size alongside the handle. On
//! acquire we pop entries until we find one whose stored size ≥ the request;
//! undersized entries are dropped (CUDA frees them) rather than reused, which
//! prevents silent buffer-overflow when `bucket_key` maps distinct request
//! sizes to the same bucket.

use std::{cell::RefCell, collections::HashMap};

use cubecl_runtime::server::Handle;

/// Rounds `bytes` up to the next power of two ≥ 64. The pool is keyed on
/// these rounded sizes, so a request for 1 000 004 bytes and one for 1 000 010
/// bytes share the same bucket (1 048 576 B). This bounds distinct buckets to
/// `log2(max_allocation)` and keeps the pool small.
#[expect(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn bucket_key(bytes: usize) -> usize {
    let rounded = bytes.max(1).next_power_of_two();
    rounded.max(64)
}

/// A pooled handle tagged with its exact allocated byte size. The pool key
/// is `bucket_key(byte_size)`, so multiple entries in the same bucket may
/// have different actual sizes — we only reuse one that fits.
type PoolEntry = (Handle, usize);

thread_local! {
    /// Thread-local pool: the executor is single-threaded per call, so this
    /// avoids any synchronisation cost. Cross-thread sharing is unnecessary
    /// because each `CubeclEngine::collect` call runs on the caller's thread.
    static POOL: RefCell<HashMap<usize, Vec<PoolEntry>>> = RefCell::new(HashMap::new());
}

/// Acquire a device buffer of at least `bytes` bytes, preferring a pooled
/// handle that's large enough. Falls back to fresh allocation from `client`
/// when no pooled handle in the bucket is large enough.
pub(crate) fn acquire(client: &cubecl::prelude::Client, bytes: usize) -> Handle {
    let key = bucket_key(bytes);
    POOL.with_borrow_mut(|pool| {
        if let Some(bucket) = pool.get_mut(&key) {
            // Find an entry whose stored byte size covers the request.
            while let Some((handle, stored_size)) = bucket.pop() {
                if stored_size >= bytes {
                    return handle;
                }
                // Undersized: drop it (CUDA frees the allocation).
            }
        }
        // No pooled handle available — allocate fresh at exact size.
        client.empty(bytes.max(1))
    })
}

/// Return a device buffer to the pool for future reuse. Callers should call
/// this on handles they would otherwise drop after an op finishes.
pub(crate) fn recycle(handle: Handle, bytes: usize) {
    let key = bucket_key(bytes);
    POOL.with_borrow_mut(|pool| {
        let bucket = pool.entry(key).or_default();
        // Bound total pooled memory: keep at most 32 buffers per bucket
        // (≈ 32 MiB for MB-scale buffers, more for smaller ones).
        if bucket.len() < 32 {
            bucket.push((handle, bytes));
        }
        // Otherwise let it drop (CUDA frees the allocation).
    });
}

/// Reset the thread-local pool, releasing every pooled handle. Used in tests
/// to keep memory accounting predictable; otherwise the pool lives for the
/// process lifetime, which is fine because pooled buffers are bounded and
/// eagerly reclaimable by the CUDA driver.
#[cfg(test)]
pub(crate) fn reset_pool() {
    POOL.with_borrow_mut(|pool| pool.clear());
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
