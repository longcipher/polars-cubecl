//! Performance baseline: transfer bandwidth, filter/group-by throughput,
//! and end-to-end latency vs the Polars CPU engine.
//!
//! Benchmarks are reported, never asserted. Run with
//! `cargo bench -p polars-cubecl` (add `--features metal` on Apple Silicon
//! to measure the Metal backend).

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::{CubeclEngine, EngineConfig};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::IntoLazy,
};

fn engine() -> Option<CubeclEngine> {
    CubeclEngine::new(EngineConfig {
        backend: polars_cubecl::Backend::Auto,
        raise_on_fail: true,
        min_rows_for_gpu: 0,
    })
    .ok()
}

fn frame_i64(rows: usize) -> DataFrame {
    let keys: Vec<i32> = (0..rows).map(|index| (index % 1024) as i32).collect();
    let groups: Vec<i32> = (0..rows).map(|index| (index % 64) as i32).collect();
    let values: Vec<i64> = (0..rows).map(|index| index as i64).collect();
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), keys).into(),
        Series::new("g".into(), groups).into(),
        Series::new("b".into(), values).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

fn bench_filter(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("filter");
    for rows in [100_000_usize, 1_000_000] {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64(rows);
        let Some(engine) = engine() else { return };
        group.bench_with_input(BenchmarkId::new("cubecl", rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = engine
                    .collect_strict(frame.clone().lazy().filter(col("a").gt(lit(512))))
                    .unwrap_or(DataFrame::empty_with_height(0));
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new("polars_cpu", rows), &rows, |bencher, _| {
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

fn bench_group_by(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("group_by");
    for rows in [100_000_usize, 1_000_000] {
        group.throughput(Throughput::Elements(rows as u64));
        let frame = frame_i64(rows);
        let Some(engine) = engine() else { return };
        group.bench_with_input(BenchmarkId::new("cubecl", rows), &rows, |bencher, _| {
            bencher.iter(|| {
                let out = engine
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
        group.bench_with_input(BenchmarkId::new("polars_cpu", rows), &rows, |bencher, _| {
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
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8)).warm_up_time(Duration::from_secs(2));
    targets = bench_filter, bench_group_by
);
criterion_main!(benches);
