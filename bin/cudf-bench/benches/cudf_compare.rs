//! Head-to-head benchmark suite: `polars-cubecl` (CubeCL backend) vs
//! Polars CPU engine, covering the same operations cuDF nvbench targets.
//!
//! Mirrors the operations in cudf's `cpp/benchmarks/`:
//!   - filter/stream_compaction → bench_filter
//!   - binaryop                 → bench_binary_op
//!   - reduction                → bench_reduction
//!   - groupby                  → bench_group_by
//!   - sort                     → bench_sort
//!   - chained pipeline         → bench_chained
//!
//! Run with `cargo bench -p cudf-bench` (add `--features metal` on Apple
//! Silicon, `--features cuda` on NVIDIA, etc.)

use std::{hint::black_box, ops::Add, time::Duration};

use std::ops::Mul;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars_core::prelude::SortMultipleOptions;
use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::{CubeclEngine, EngineConfig};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::IntoLazy,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn engine() -> Option<CubeclEngine> {
    CubeclEngine::new(EngineConfig {
        backend: polars_cubecl::Backend::Cuda,
        raise_on_fail: true,
        min_rows_for_gpu: 0,
    })
    .ok()
}

/// Generate a frame with integer/f64 columns for benchmarking.
fn frame_i64_f64(rows: usize) -> DataFrame {
    let keys: Vec<i32> = (0..rows).map(|index| (index % 1024) as i32).collect();
    let groups: Vec<i32> = (0..rows).map(|index| (index % 64) as i32).collect();
    let vals_i64: Vec<i64> = (0..rows).map(|index| index as i64).collect();
    let vals_f64: Vec<f64> = (0..rows).map(|index| index as f64).collect();
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), keys).into(),
        Series::new("g".into(), groups).into(),
        Series::new("b".into(), vals_i64).into(),
        Series::new("c".into(), vals_f64).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

/// Sizes: 100K, 1M, and 10M row-scale tests.
const SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];

/// Sort options: descending, no nulls (data is non-null), single-thread for
/// deterministic comparison. cuDF benchmarks don't use maintain_order.
fn sort_desc() -> SortMultipleOptions {
    SortMultipleOptions {
        descending: vec![true],
        nulls_last: vec![false],
        multithreaded: false,
        maintain_order: false,
        limit: None,
    }
}

fn sort_asc() -> SortMultipleOptions {
    SortMultipleOptions {
        descending: vec![false],
        nulls_last: vec![false],
        multithreaded: false,
        maintain_order: false,
        limit: None,
    }
}

// ---------------------------------------------------------------------------
// Filter (mirrors cudf STREAM_COMPACTION_NVBENCH)
// ---------------------------------------------------------------------------

fn bench_filter(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("filter");
    for rows in SIZES {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        let id_cl = format!("cubecl_{rows}");
        let id_cpu = format!("polars_cpu_{rows}");

        group.bench_with_input(BenchmarkId::new(&id_cl, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(frame.clone().lazy().filter(col("a").gt(lit(512))))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new(&id_cpu, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .filter(col("a").gt(lit(512)))
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Binary op (mirrors cudf BINARYOP_NVBENCH)
// ---------------------------------------------------------------------------

fn bench_binary_op(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary_op");
    for rows in SIZES {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        // i64 add.
        let id = format!("cubecl_add_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame.clone().lazy().select([col("b").add(col("b")).alias("result")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_add_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .select([col("b").add(col("b")).alias("result")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // f64 add.
        let id = format!("cubecl_add_f64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame.clone().lazy().select([col("c").add(col("c")).alias("result")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_add_f64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .select([col("c").add(col("c")).alias("result")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // i64 multiply.
        let id = format!("cubecl_mul_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame.clone().lazy().select([col("b").mul(col("b")).alias("result")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_mul_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .select([col("b").mul(col("b")).alias("result")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Reduction (mirrors cudf REDUCTION_NVBENCH)
// ---------------------------------------------------------------------------

fn bench_reduction(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("reduction");
    for rows in SIZES {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        // Sum.
        let id = format!("cubecl_sum_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(frame.clone().lazy().select([col("b").sum()]))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_sum_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .select([col("b").sum()])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // Min.
        let id = format!("cubecl_min_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(frame.clone().lazy().select([col("b").min()]))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_min_i64_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .select([col("b").min()])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GroupBy (mirrors cudf GROUPBY_NVBENCH)
// ---------------------------------------------------------------------------

fn bench_group_by(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("group_by");
    for rows in SIZES {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        // Sum aggregation.
        let id = format!("cubecl_sum_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame
                            .clone()
                            .lazy()
                            .group_by([col("g")])
                            .agg([col("b").sum().alias("s"), len().alias("n")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_sum_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .group_by([col("g")])
                    .agg([col("b").sum().alias("s"), len().alias("n")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // Mean aggregation.
        let id = format!("cubecl_mean_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame
                            .clone()
                            .lazy()
                            .group_by([col("g")])
                            .agg([col("b").mean().alias("m")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_mean_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .group_by([col("g")])
                    .agg([col("b").mean().alias("m")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // First aggregation.
        let id = format!("cubecl_first_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame
                            .clone()
                            .lazy()
                            .group_by([col("g")])
                            .agg([col("b").first().alias("f")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_first_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .group_by([col("g")])
                    .agg([col("b").first().alias("f")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // Count.
        let id = format!("cubecl_count_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame
                            .clone()
                            .lazy()
                            .group_by([col("g")])
                            .agg([col("b").count().alias("c")]),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_count_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .group_by([col("g")])
                    .agg([col("b").count().alias("c")])
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Sort (mirrors cudf SORT_NVBENCH)
// ---------------------------------------------------------------------------

fn bench_sort(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sort");
    for rows in SIZES {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        // Ascending sort.
        let id = format!("cubecl_sort_asc_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(frame.clone().lazy().sort(["b"], sort_asc()))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_sort_asc_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .sort(["b"], sort_asc())
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });

        // Descending sort.
        let id = format!("cubecl_sort_desc_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(frame.clone().lazy().sort(["b"], sort_desc()))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_sort_desc_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .sort(["b"], sort_desc())
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Chained pipeline (filter → group_by → sort)
// ---------------------------------------------------------------------------

fn bench_chained(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("chained_filter_groupby_sort");
    for rows in [1_000_000_usize, 10_000_000_usize] {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64_f64(rows);
        let Some(eng) = engine() else { return };

        let id = format!("cubecl_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = eng
                    .collect_strict(
                        frame
                            .clone()
                            .lazy()
                            .filter(col("a").gt(lit(256)))
                            .group_by([col("g")])
                            .agg([col("b").sum().alias("s")])
                            .sort(["s"], sort_desc()),
                    )
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        let id = format!("polars_cpu_{rows}");
        group.bench_with_input(BenchmarkId::new(&id, rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = frame
                    .clone()
                    .lazy()
                    .filter(col("a").gt(lit(256)))
                    .group_by([col("g")])
                    .agg([col("b").sum().alias("s")])
                    .sort(["s"], sort_desc())
                    .collect()
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion group registration
// ---------------------------------------------------------------------------

criterion_group!(
    name = cudf_compare;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1))
        .sample_size(30);
    targets = bench_filter,
           bench_binary_op,
           bench_reduction,
           bench_group_by,
           bench_sort,
           bench_chained,
);

criterion_main!(cudf_compare);
