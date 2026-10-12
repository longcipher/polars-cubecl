# cudf-bench: polars-cubecl vs cuDF Performance Comparison

This crate provides Criterion benchmarks covering the same operations cudf's
nvbench suite targets. Run them to compare `polars-cubecl` (CubeCL backends)
against the Polars CPU engine as a first baseline, then against cuDF numbers
gathered separately.

## Operations benchmarked

| Group | Op | cuDF benchmark equivalent |
|---|---|---|
| `filter` | Column filter (~50% selectivity) | `STREAM_COMPACTION_NVBENCH` |
| `binary_op` | Column add (i64, f64) | `BINARYOP_NVBENCH` |
| `reduction` | Column sum, min | `REDUCTION_NVBENCH` |
| `group_by` | Group-by + sum/mean/first/count | `GROUPBY_NVBENCH` |
| `sort` | Single-column sort (asc/desc) | `SORT_NVBENCH` |
| `chained_filter_groupby_sort` | Filter → GroupBy → Sort pipeline | Composite |

## Data sizes

- 100K rows (small, transfer-dominated)
- 1M rows (medium)
- 10M rows (large, kernel-dominated)
- 50M rows (very large, memory-bandwidth bound)

## Running the benchmarks

```bash
# Default CPU backend
cargo bench -p cudf-bench

# NVIDIA GPU
cargo bench -p cudf-bench --features cuda -- --noplot

# Apple Silicon
cargo bench -p cudf-bench --features metal -- --noplot

# AMD ROCm
cargo bench -p cudf-bench --features hip -- --noplot

# Save results as baseline for comparison
cargo bench -p cudf-bench -- --save-baseline polars-cubecl
```

## Comparing against cuDF

1. Build cudf with benchmarks enabled:
   ```bash
   cd /path/to/cudf/cpp
   # Follow cudf build instructions with -DBUILD_BENCHMARKS=ON
   ```

2. Run the relevant cudf nvbench binaries:
   ```bash
   ./build/benchmarks/libcudf/STREAM_COMPACTION_NVBENCH
   ./build/benchmarks/libcudf/BINARYOP_NVBENCH
   ./build/benchmarks/libcudf/REDUCTION_NVBENCH
   ./build/benchmarks/libcudf/GROUPBY_NVBENCH
   ./build/benchmarks/libcudf/SORT_NVBENCH
   # ... etc.
   ```

3. Capture output and run the comparison script:
   ```bash
   python3 compare.py --cudf-results cudf_output.txt \
                      --cubecl-dir target/criterion \
                      --output comparison.md
   ```

## Target performance goals

| Metric | Goal |
|---|---|
| Small data (100K) | Within 2x of cuDF (transfer dominates) |
| Medium data (1M) | Within 1.5x of cuDF |
| Large data (10M+) | At or above cuDF performance |
| Very large (50M+) | Match or exceed cuDF bandwidth utilization |

## How to interpret results

- If CubeCL is **slower on small data**: expected — PCIe transfer overhead
  dominates. The `min_rows_for_gpu` config (default 10K) gates this.
- If CubeCL is **slower on large data**: investigate kernel efficiency,
  memory coalescing, or backend compilation quality.
- If CubeCL is **faster on large data**: excellent — the GPU's parallelism
  wins once transfer is amortized.

## Profiling workflow

When polars-cubecl is slower than expected:

1. Run `cargo bench` to identify which group regresses.
2. Check the execution report (`collect_strict_with_report`) to confirm the
   engine engaged and didn't fall back to CPU.
3. Use `perf`/` Instruments` / `nsys` to profile the bottleneck:
   - Transfer overhead → reduce host↔device round-trips
   - Kernel launch overhead → fuse kernels, batch work
   - Memory bandwidth → improve access patterns, reduce intermediate allocs
   - ALU-bound → check CubeCL codegen quality for the backend
