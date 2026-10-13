# polars-cubecl — Measured Baseline

> Apple M1 Max, 32 GB. Criterion, 8 s measurement + 2 s warm-up per bench.
> Engine build: `cargo bench -p polars-cubecl [--features metal]`.
> `cubecl` = Metal GPU via wgpu (`Backend::Auto`); `polars_cpu` = `LazyFrame::collect()`.

## Numbers

Filter (`a > 512` over `(a: i32, g: i32, b: i64)`):

| rows | Metal | Polars CPU | ratio |
| --- | --- | --- | --- |
| 100 000 | 4.09 ms (24 Melem/s) | 0.12 ms (827 Melem/s) | 34x slower |
| 1 000 000 | 10.2 ms (98 Melem/s) | 0.82 ms (1220 Melem/s) | 12x slower |

Group-by (`group_by(g).agg([b.sum(), len()])`, 64 groups):

| rows | Metal | Polars CPU | ratio |
| --- | --- | --- | --- |
| 100 000 | 6.27 ms | ~0.3 ms | ~20x slower |
| 1 000 000 | 29.6 ms | 2.45 ms (408 Melem/s) | 12x slower |

CPU backend (CubeCL LLVM, correctness vehicle, not a speed path) is another
5–10x slower than Metal at these sizes.

## Reading the numbers honestly

The GPU loses everywhere at these sizes, by roughly 10–30x. The breakdown:

1. **Per-launch overhead dominates.** A filter runs ~9 launches
   (compare + count + scatter + gather per column + validity) and a
   group-by runs more (invert + one reduce per agg + gathers). At ~1 ms per
   wgpu/Metal launch, launches alone exceed the entire CPU runtime.
2. **Transfers are secondary but real.** ~30 MB per 1 M-row filter query
   (upload values + validity, download results + indices) plus two host
   round-trips (filter tallies, group layout).
3. **Group-by grouping is host-serial.** `FxHashMap` grouping is O(n) on one
   core (~20–40 ms/M rows); the device only accelerates the reductions.
4. **Throughput scales with size** (24 → 98 Melem/s from 100 k to 1 M rows):
   fixed costs amortize, but the crossover where the GPU wins outright is
   beyond practical single-frame sizes for these operators.

None of this is a correctness problem — every number above is a
differentially verified correct result — but it sets the optimization
agenda honestly.

## What would move the needle

- **Kernel fusion** (biggest lever): one kernel per pipeline stage instead
  of one per op would remove most of the ~1 ms/launch tax. The current
  one-kernel-per-op shape is a deliberate M0/M1 simplicity choice.
- **Device-side grouping**: replace host `FxHashMap` grouping with a device
  probe once float atomics (or deterministic alternatives) are available
  across backends.
- **Fewer round-trips**: device-side prefix sums for compaction (needs
  reliable cross-unit barriers — unavailable on the CPU backend, see
  `design.md` §6.2), persistent command buffers, async pipelining.
- **`min_rows_for_gpu`**: defaults to 10 000 rows; below that the transfer
  term dominates and the engine stays on the CPU. Evidence-based tuning of
  this threshold per backend is future work.

## Reproduce

```bash
cargo bench -p polars-cubecl --bench engine
cargo bench -p polars-cubecl --features metal --bench engine
```

Benchmarks are reported, never asserted: they exist to make performance
claims falsifiable, not to gate CI.

## Pipeline-Level Bottleneck Analysis

A detailed breakdown of per-operation kernel launches, sync points, DMA
timing, and ranked optimization opportunities lives in
[`perf_bottleneck_analysis.md`](perf_bottleneck_analysis.md). Key findings
(as of 2026-10-13, post GPU-histogram + direct-sum):

| Operation | Launch overhead | Sync overhead | Kernel compute | Total est. |
|-----------|----------------|---------------|----------------|------------|
| Filter (1M) | ~3.0 ms (6 launches) | ~1.0 ms (2 syncs) | ~0.1 ms | ~4–7 ms |
| Group-by sum (1M/64) | ~2.0 ms (4 launches) | ~1.5 ms (3 syncs) | ~4.9 ms | ~9–15 ms |

The three highest-ROI optimizations are:
1. **Vectorize reduce_sum** (kernel: 4.9 ms → 1.2 ms)
2. **Kernel fusion** (filter: 3 launches → 1 launch; saves ~1.5 ms)
3. **Batch histogram readbacks** (group-by: 3 syncs → 1 sync; saves ~1.0 ms)
