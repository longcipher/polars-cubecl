//! End-to-end stress tests: multi-op pipelines and larger inputs.
//!
//! These tests exercise the full engine pipeline from lowering through
//! execution and splice-back on workloads that combine many operators,
//! stress the compaction / group-by kernels at scale, and chain device +
//! CPU fallback phases in a single query.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_backend, check_backend_unsorted,
};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::{IntoLazy, LazyFrame},
};

fn rules() -> CompareRules {
    CompareRules::default()
}

fn check(lf: LazyFrame) -> Result<(), HarnessError> {
    for backend in all_backends() {
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-op pipelines: filter → select → filter (three device phases).
// ---------------------------------------------------------------------------

#[test]
fn chained_filters_with_select_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into(),
        Series::new("b".into(), &[
            Some(10_i64),
            None,
            Some(30),
            Some(40),
            Some(50),
            None,
            None,
            Some(80),
            Some(90),
            Some(100),
        ])
        .into(),
        Series::new("c".into(), &(0..10).map(|i| i as f32).collect::<Vec<f32>>()).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(
        frame
            .lazy()
            .filter(col("a").gt(lit(2)))
            .select([col("a"), col("b")])
            .filter(col("a").lt(lit(8)))
            .select([col("b")]),
    )
}

// ---------------------------------------------------------------------------
// Multi-op pipeline ending in an unsupported tail (sort) stays GPU-correct
// through the device head.
// ---------------------------------------------------------------------------

#[test]
fn filter_group_sort_hybrid_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1].as_slice()).into(),
        Series::new("v".into(), &[
            Some(10_i64),
            Some(20),
            None,
            Some(40),
            Some(50),
            Some(60),
            None,
            Some(80),
            Some(90),
            Some(100),
            Some(110),
            None,
        ])
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .filter(col("v").gt(lit(15)))
            .group_by([col("g")])
            .agg([col("v").sum().alias("s"), len().alias("n")])
            .sort(["g"], Default::default());
        let Some(report) = check_backend_unsorted(&lf, backend, &keys, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
        assert!(report.ops_ran().contains(&Op::Filter));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Large-frame stress: 100K rows with moderate cardinality.
// ---------------------------------------------------------------------------

#[test]
fn group_large_frame_stress_matches_oracle() -> Result<(), HarnessError> {
    let rows: usize = 100_000;
    let keys: Vec<i32> = (0..rows).map(|i| (i % 500) as i32).collect();
    let values: Vec<Option<i64>> = (0..rows)
        .map(|i| if i % 23 == 0 { None } else { Some((i as i64).wrapping_mul(7919)) })
        .collect();
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), keys).into(),
        Series::new("v".into(), values).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([
        col("v").sum().alias("s"),
        col("v").mean().alias("m"),
        col("v").min().alias("lo"),
        col("v").max().alias("hi"),
        col("v").count().alias("c"),
        len().alias("n"),
    ]);
    let key_names = vec!["g".to_string()];
    for backend in all_backends() {
        let result = check_backend_unsorted(&lf, backend, &key_names, &rules());
        if let Err(error) = result {
            unreachable!("group_large_frame_stress: {error}");
        }
    }
    Ok(())
}

#[test]
fn filter_large_frame_stress_matches_oracle() -> Result<(), HarnessError> {
    let rows = 200_000_i32;
    let half = rows / 2;
    let values: Vec<Option<i32>> = (0..rows).map(Some).collect();
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), values).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(half))))
}

// ---------------------------------------------------------------------------
// Multiple aliases and column reuse in a single projection.
// ---------------------------------------------------------------------------

#[test]
fn wide_projection_with_many_aliases_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3]).into(),
        Series::new("b".into(), &[10_i64, 20, 30]).into(),
        Series::new("c".into(), &[true, false, true]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(
        frame
            .lazy()
            .select([
                col("a").alias("x"),
                col("a").alias("y"),
                col("b").alias("p"),
                col("b").alias("q"),
                col("c").alias("r"),
            ])
            .filter(col("x").gt(lit(1))),
    )
}

// ---------------------------------------------------------------------------
// Slice at exact boundary: slice(len, 0) is a no-op producing zero rows.
// ---------------------------------------------------------------------------

#[test]
fn slice_zero_len_then_filter_matches_oracle() -> Result<(), HarnessError> {
    let frame =
        DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().slice(0, 0).filter(col("a").gt(lit(0))))
}

#[test]
fn slice_past_end_matches_oracle() -> Result<(), HarnessError> {
    let frame =
        DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().slice(3, 100).filter(col("a").gt(lit(0))))
}

// ---------------------------------------------------------------------------
// Negative slice offsets: exercise the signed-offset arithmetic in the
// executor.
// ---------------------------------------------------------------------------

#[test]
fn negative_slice_offset_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for offset in [-1_i64, -3, -5, -10, -100] {
        for len in [0_u32, 1, 3, 100] {
            check(frame.clone().lazy().slice(offset, len).filter(col("a").gt(lit(0))))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// All-null groups with a very small cardinality (1 group): the degenerate
// where every row is in the same group.
// ---------------------------------------------------------------------------

#[test]
fn group_all_same_key_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[7_i32; 128]).into(),
        Series::new(
            "v".into(),
            &(0..128)
                .map(|i| if i % 5 == 0 { None } else { Some(i64::from(i)) })
                .collect::<Vec<Option<i64>>>(),
        )
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([
        col("v").sum().alias("s"),
        col("v").count().alias("c"),
        len().alias("n"),
    ]);
    let key_names = vec!["g".to_string()];
    for backend in all_backends() {
        let _ = check_backend_unsorted(&lf, backend, &key_names, &rules())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Filter that selects nothing (0% selectivity) on a large frame: the
// compaction kernel must correctly produce an empty result.
// ---------------------------------------------------------------------------

#[test]
fn filter_zero_selectivity_large_frame() -> Result<(), HarnessError> {
    let rows = 50_000;
    let values: Vec<i32> = (0..rows).collect();
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), values).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(rows + 1))))
}

// ---------------------------------------------------------------------------
// Filter that selects everything (100% selectivity) on a large frame: the
// compaction kernel must pass through all rows.
// ---------------------------------------------------------------------------

#[test]
fn filter_full_selectivity_large_frame() -> Result<(), HarnessError> {
    let rows = 50_000;
    let values: Vec<i32> = (0..rows).collect();
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), values).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(-1))))
}

// ---------------------------------------------------------------------------
// Power-of-two boundary heights around common thread-block counts.
// ---------------------------------------------------------------------------

#[test]
fn filter_at_power_of_two_boundaries() -> Result<(), HarnessError> {
    for rows in [255_i32, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049] {
        let half = rows / 2;
        let values: Vec<Option<i32>> = (0..rows).map(|i| Some(i - half)).collect();
        let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), values).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
        check(frame.lazy().filter(col("a").gt(lit(0))))?;
    }
    Ok(())
}
