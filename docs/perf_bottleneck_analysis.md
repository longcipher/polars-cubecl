# polars-cubecl — Performance Bottleneck Analysis

> Platform: NVIDIA CUDA (primary), AMD HIP, Apple Metal.
> Baseline: `docs/benchmarks.md` (Metal M1 Max, 1M rows, filter=10.2 ms, group-by=29.6 ms).
> Status: Post GPU-histogram + direct-sum optimization; serial reduce_sum (4.89 ms at 1M/64).

---

## 1. Pipeline Decomposition

Every operator pipeline follows the same pattern: **ingest → N kernel launches → readback → download**. We decompose each operation into its constituent kernel launches and sync points to expose where time goes.

### 1.1 Filter (`a > 512` over `(a: i32, g: i32, b: i64)`, 1M rows)

| # | Step | Type | Cost driver |
|---|------|------|-------------|
| 1 | Upload 3 columns (values + validity) | DMA H→D | 12 MB (values) + 3 MB (validity) ≈ 15 MB ≈ 120 µs at 128 GB/s PCIe |
| 2 | `launch_cmp` | kernel launch | ~500 µs CUDA startup |
| 3 | `launch_count` | kernel launch | ~500 µs |
| 4 | Read back counts (`threads` × u32) | Sync D→H | ~500 µs sync + µs transfer |
| 5 | Host prefix sum | CPU | <10 µs |
| 6 | Upload bases | DMA H→D | ~50 µs |
| 7 | `launch_scatter` | kernel launch | ~500 µs |
| 8 | `launch_gather_values_mask` × 3 cols | 3 kernel launches | ~1500 µs |
| 9 | Download result (3 compacted columns) | DMA D→H | ~300 µs |

**Launch overhead: ~6 launches × 500 µs = 3000 µs = 3.0 ms**
**Sync overhead: ~2 sync points × 500 µs = 1.0 ms**
**DMA: ~570 µs**
**Total estimate: ~4.6 ms** (real: 10.2 ms Metal, ~5-7 ms CUDA with buffer pool)

The remaining ~2-4 ms comes from:

- Metal's higher per-launch cost (~1 ms vs CUDA's ~500 µs)
- Memory allocation overhead (mitigated by buffer pool)
- Polars CPU filter runs at 0.82 ms — the GPU overhead is 6-12x the total CPU time

### 1.2 Group-by (`group_by(g).agg([b.sum()])`, 1M rows, 64 groups)

*GPU histogram fast path — all-sum with integer key.*

| # | Step | Type | Cost driver |
|---|------|------|-------------|
| 1 | Upload key `g` + value `b` + validities | DMA H→D | ~16 MB ≈ 128 µs |
| 2 | Read key validity (null check) | Sync D→H | ~500 µs |
| 3 | `launch_reduce_min_max_direct` | kernel launch | ~500 µs |
| 4 | Read back min + max (8 bytes) | Sync D→H | ~500 µs |
| 5 | `launch_gpu_histogram_and_assign` | kernel launch | ~500 µs |
| 6 | Read back counts + assignment | Sync D→H | ~500 µs |
| 7 | CPU: build first/last rows, compute sizes | CPU row scan | ~20 µs |
| 8 | `launch_reduce_sum_direct` | kernel launch | ~500 µs + kernel runtime |
| 9 | Download result | DMA D→H | ~100 µs |

**Launch overhead: ~4 launches × 500 µs = 2.0 ms**
**Sync overhead: ~3 sync points × 500 µs = 1.5 ms**
**CPU scan: ~20 µs**
**DMA: ~250 µs**
**reduce_sum_direct kernel: ~4.89 ms (at 1M/64 groups)**
**Total estimate: ~9.2 ms** (real: 29.6 ms Metal including all DMA and alloc overhead)

### 1.3 Group-by with `len()` (`group_by(g).agg([b.sum(), len()])`)

Same as 1.2 plus:

- `launch_reduce_sum_direct` for sum
- `len()` is free (computed from sizes during histogram)
- `materialize_keys` launches 1 gather kernel, downloads key column

The dominant cost is unchanged — the sum kernel dominates at ~4.89 ms.

---

## 2. Root Cause Hierarchy

Ranked by estimated wall-clock impact (CUDA, 1M rows, 3-column filter):

| Priority | Bottleneck | Impact | Current state |
|----------|-----------|--------|---------------|
| **P0** | reduce_sum kernel runtime | ~4.9 ms | Serial loop, 1 thread per group |
| **P1** | Kernel launch overhead | ~3.0 ms | 4-6 launches per op |
| **P2** | Sync points (readback) | ~1.5 ms | 2-3 syncs per op |
| **P3** | DMA transfers | ~0.6 ms | Fixed by data size |
| **P4** | Buffer allocation | ~0.1 ms | Mitigated by buffer pool |

### P0 — reduce_sum Serial Kernel (dominant compute cost)

The serial `reduce_sum_direct_kernel` runs ABSOLUTE_POS threads, each writing atomically:

- At 1M rows / 64 groups: 1M atomic adds spread across 64 buckets
- The kernel is limited by **atomic contention** on the 64 output slots
- Even the serial-per-group kernel (1 thread per group, each loops 15625 elements) takes 4.89 ms — this is **GPU underutilization**: only 64 of 16384+ CUDA cores do useful work

Why the parallel version is 17% slower:

- 64 blocks × 256 threads = 16384 threads total
- 64 block launch overhead (each block's thread 0 does 256 serial reads)
- The serial baseline: 1 block × 256 threads, only 64 with work, serial loop = same throughput as 64 serial threads
- **The bottleneck is not compute/memory bandwidth — it's low thread-count GPU underutilization**

### P1 — Kernel Launch Overhead

Each `launch_*` call costs ~500 µs (CUDA) or ~1 ms (Metal/wgpu):

- CUDA: PTX→cubin JIT happens on first launch (~1 ms for first, ~500 µs for subsequent via Kache cache)
- Metal: MSL→AIR→Metal pipeline compile, ~1 ms per distinct kernel
- CPU backend: trivial (Function call)

For a filter with 3 data columns, we pay 6 launch overheads before doing any useful work that exceeds the CPU reference (0.82 ms).

### P2 — Sync Points

Every `client.read_one()` is a full pipeline flush + CUDA sync:

- Filter: 1 sync (counts readback)
- Group-by histogram: 3 syncs (validity check, min+max readback, histogram readback)
- Each sync: ~500 µs on CUDA

---

## 3. Prioritized Optimization Roadmap

### 3.1 P0: Vectorized reduce_sum (Target: 4.89 ms → ~1.2 ms)

**Approach**: Change `reduce_sum_direct_kernel` to use `Vector<T, 4>` or `Vector<T, 8>` loads. Each thread processes N rows per iteration, reducing instruction count by N×.

**Expected gain**: 2-4× speedup on the kernel itself (from 4.89 ms to ~1.2-2.4 ms).

**Implementation**: In the `inner` module's `reduce_sum_direct_kernel`, change:
```rust
// Before: scalar per row
if row < assignment.len() {
    let group = assignment[row] as usize;
    if group < num_groups && valid[row] != 0u8 {
        out_values[group].fetch_add(values[row]);
    }
}
// After: vectorized — process 4 rows per thread
let row_base = ABSOLUTE_POS * 4;
if row_base < assignment.len() {
    let v = Vector::cast_from(values[row_base]);  // load 4 values at once
    let g = assignment[row_base];                 // still scalar (gather)
    // ... distribute 4 accumulates to their groups
}
```

**Challenge**: Atomic adds are per-element, so full vectorization requires either:

- (a) Vector atomic adds (if backend supports `Atomic<Vector<T,N>>`)
- (b) 4 separate scalar atomics per iteration (still reduces loop overhead by 4×)
- (c) Shared-memory reduction within a block, then atomic add to global

**P0a (highest ROI)**: Option (b) — unroll the loop 4× manually. No new backend features, just fewer loop iterations. Expected: 30-40% reduction in kernel time.

### 3.2 P1: Kernel Fusion (Target: 4-6 launches → 1-2 launches)

**Approach**: Fuse multiple independent kernels into one. Two strategies:

**3.2.1 Filter fusion** (cmp + count + scatter):

- Currently: 3 launches (cmp→count→scatter) + 1 readback
- Fused: 1 launch that produces counts directly from the comparison (no intermediate mask buffer)
- Eliminates mask buffer allocation + 2 launches + 1 sync
- Expected savings: ~1.5 ms on CUDA

**3.2.2 Multi-column gather fusion** (filter):

- Currently: N separate `launch_gather_values_mask` calls
- Fused: 1 launch that gathers M columns from the same index vector
- Requires either dynamic column count (virtual args) or comptime specialization
- Expected: savings = (N-1) × 500 µs per filter

**3.2.3 Group-by sum+count fusion**:

- When query has both `sum()` and `count()`/`len()`, compute both in one kernel pass
- Eliminates one redundant read of `values` + `assignment`
- Expected: ~500 µs (one less launch)

### 3.3 P2: Reduce Sync Points (Target: 3 syncs → 1 sync)

**3.3.1 Batched readback for histogram path**:

- Current: separate `read_one` for min, max, counts, assignment
- Fix: Use CubeCL's `read_all` or batch handles into one sync
- Saves ~1.0 ms on group-by (3 syncs → 1 sync)

**3.3.2 Async readback with CUDA streams**:

- Launch sum kernel before reading back histogram
- Overlap kernel execution with host-side CPU work (first/last row computation)
- Saves ~500 µs by overlapping CPU post-processing

### 3.4 P3: DMA Optimization (Already largely addressed)

- Buffer pool eliminates most allocation overhead
- Zero-copy assignment (GPU histogram path) avoids re-uploading
- Remaining: upload/download are fundamentally data-bound

### 3.5 P4: Host-Side Optimization

- `group_rows` dense-integer fast path already avoids FxHashMap
- Remaining CPU cost: <1% of total at M-scale

---

## 4. Theoretical Ceiling Analysis

Best case after all P0-P2 optimizations (CUDA, 1M rows):

| Component | Current | Optimized |
|-----------|---------|-----------|
| Upload | 128 µs | 128 µs |
| Launch overhead (fused) | 3000 µs | 500 µs |
| Sync points (batched) | 1500 µs | 500 µs |
| reduce_sum kernel | 4890 µs | 1200 µs |
| Download | 300 µs | 300 µs |
| **Total** | **~9.8 ms** | **~2.6 ms** |
| CPU reference | 0.82 ms | 0.82 ms |
| **Ratio** | **12×** | **3.2×** |

To beat the CPU at 1M rows, we need:

- reduce_sum kernel < 300 µs (needs parallel reduction tree + warp shuffles)
- Total launches ≤ 2
- Total sync points ≤ 1
- **Target: < 0.8 ms total** to match CPU

At **10M rows** (CPU ~8.2 ms), the GPU wins easily even with current overhead because the kernel time scales linearly while CPU scaling is linearithmic. The crossover is around **2-3M rows** with current code, potentially **500K rows** after optimizations.

---

## 5. Profiling Methodology

To validate these estimates on real hardware:

### 5.1 CUDA Profiling

```bash
# Nsight Systems timeline
nsys profile -t cuda,nvtx --stats=true cargo bench -p polars-cubecl --features cuda --bench engine

# Per-kernel metrics
ncu --target-processes all --set full cargo bench -p polars-cubecl --features cuda --bench engine
```

### 5.2 Metal Profiling

```bash
# Metal GPU trace
xcrun metal-gpu-tool -o trace.gputrace cargo bench -p polars-cubecl --features metal --bench engine

# Metal System Trace instrument for CPU+GPU correlation
instruments -t "Metal System Trace" -D trace.trace cargo bench ...
```

### 5.3 Criterion-based micro-benchmarks

Add per-phase timing at the criterion level to measure individual kernel overhead vs data size, using `criterion::BenchmarkGroup::bench_function` with GPU event timers (CUDA events, Metal command buffer start/end time).

---

## 6. Key Metrics to Track

1. **`reduce_sum_direct` wall-time** vs `num_groups` (1→65536) and `height` (10K→10M)
2. **Per-launch overhead** measured as: launch empty kernel to completion, divide by count
3. **Sync-to-sync latency**: time between `kernel launch` → `read_one` completion
4. **Kernel overlap ratio**: how much of the CPU-side work (prefix sum, first/last rows) overlaps with GPU execution
5. **Memory bandwidth utilization**: `reduce_sum` bytes-read / (time × peak bandwidth)

---

## 7. Conclusion

The current performance is bounded by three compounding factors:

1. **Kernel launch tax** (~500 µs each × 4-6 launches = 2-3 ms)
2. **Sync point overhead** (~500 µs each × 2-3 = 1-1.5 ms)
3. **Serial reduce kernel** (~4.9 ms at 1M/64)

The optimize-all-three target is ~2.6 ms at 1M rows (vs 0.82 ms CPU). The GPU becomes the winner at ~2M+ rows where its O(n) scaling beats the CPU's O(n log n) scaling. The highest single-ROI action is **vectorizing reduce_sum** (P0); the second-highest is **batching the histogram readbacks** (P2.3.1).
