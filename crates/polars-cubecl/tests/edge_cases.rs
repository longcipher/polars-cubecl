//! Edge-case differential tests: boundary conditions the main suites skip.
//!
//! Single-row frames, all-null frames, extreme-value combinations, and
//! multi-op pipelines that exercise the kernel boundaries less traveled by
//! the property and scenario suites.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_backend, check_backend_unsorted,
    check_or_fallback,
};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::{IntoLazy, LazyFrame},
};
use proptest::prelude::*;

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
// Single-row frame: every kernel path must handle height == 1 without
// panicking or producing a wrong result (thread-block math often assumes
// height >= 2).
// ---------------------------------------------------------------------------

#[test]
fn filter_single_row_match() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[42_i32]).into(),
        Series::new("b".into(), &[Some(7_i64)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().filter(col("a").gt(lit(0)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn filter_single_row_no_match() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[42_i32]).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().filter(col("a").gt(lit(100)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn slice_single_row() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[5_i32]).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().slice(0, 1);
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// All-null columns: every supported dtype must filter / group / slice a
// column where every value is `null`.
// ---------------------------------------------------------------------------

#[test]
fn filter_all_null_i32_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[None, None, None, None] as &[Option<i32>; 4]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().filter(col("a").gt(lit(0)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn group_all_null_values_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 1, 2, 2]).into(),
        Series::new("v".into(), &[None, None, None, None] as &[Option<i64>; 4]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .group_by([col("g")])
            .agg([col("v").sum().alias("s"), len().alias("n")]);
        let Some(report) = check_backend_unsorted(&lf, backend, &keys, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extreme magnitude combinations: filter on i64::MAX, f64 specials, etc.
// ---------------------------------------------------------------------------

#[test]
fn filter_i64_extremes_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[
            Some(i64::MIN),
            Some(-1),
            Some(0),
            Some(1),
            Some(i64::MAX),
            None,
        ])
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        for threshold in [i64::MIN, -1, 0, 1, i64::MAX] {
            let lf = frame.clone().lazy().filter(col("a").gt(lit(threshold)));
            let _ = check_backend(&lf, backend, &rules())?;
        }
    }
    Ok(())
}

#[test]
fn filter_u64_extremes_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[Some(0_u64), Some(u64::MAX), None, Some(1), Some(u64::MAX - 1)])
            .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().filter(col("a").gt(lit(u64::MAX / 2)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn filter_u32_extremes_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[Some(0_u32), Some(u32::MAX), None, Some(1)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(u32::MAX / 2))))
}

// ---------------------------------------------------------------------------
// Multi-op pipelines: filter→slice, filter→group, select→filter→slice
// ---------------------------------------------------------------------------

#[test]
fn filter_then_slice_then_select_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5, 6, 7, 8]).into(),
        Series::new("b".into(), &[
            Some(10_i64),
            None,
            Some(30),
            Some(40),
            Some(50),
            None,
            None,
            Some(80),
        ])
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(2))).slice(1, 3).select([col("b"), col("a")]))
}

#[test]
fn filter_select_drop_matches_oracle() -> Result<(), HarnessError> {
    // After selecting only `b`, filtering on `a` must fail to engage (the
    // column is gone) — but an earlier filter before select is fine.
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into(),
        Series::new("b".into(), &[10_i64, 20, 30, 40, 50]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("a").gt(lit(1))).select([col("b")]))
}

#[test]
fn multiple_aliases_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3]).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().select([col("a").alias("x"), col("a").alias("y"), col("a").alias("z")]))
}

// ---------------------------------------------------------------------------
// Boolean columns edge cases: all-true, all-false, all-null
// ---------------------------------------------------------------------------

#[test]
fn filter_all_true_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("flag".into(), &[true, true, true]).into(),
        Series::new("a".into(), &[1_i32, 2, 3]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("flag")))
}

#[test]
fn filter_all_false_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("flag".into(), &[false, false, false]).into(),
        Series::new("a".into(), &[1_i32, 2, 3]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("flag")))
}

#[test]
fn filter_all_null_bool_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("flag".into(), &[None, None, None] as &[Option<bool>; 3]).into(),
        Series::new("a".into(), &[1_i32, 2, 3]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.lazy().filter(col("flag")))
}

// ---------------------------------------------------------------------------
// Proptest: tiny randomized frames under filter + group, at modest case
// counts to keep CI fast.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Boolean column with nulls: verify the NOT variant (De Morgan) + And/Or
// ---------------------------------------------------------------------------

#[test]
fn filter_bool_with_nulls_and_combinators() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("flag".into(), &[Some(true), None, Some(false), None, Some(true)]).into(),
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.clone().lazy().filter(col("flag").and(col("a").gt(lit(1)))))?;
    check(frame.lazy().filter(col("flag").or(col("a").gt(lit(10)))))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Boolean as group-by key: bool dtype is in the supported subset.
// ---------------------------------------------------------------------------

#[test]
fn group_by_bool_key_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("k".into(), &[true, false, true, false, true]).into(),
        Series::new("v".into(), &[1_i32, 2, 3, 4, 5]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let keys = vec!["k".to_string()];
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .group_by([col("k")])
            .agg([col("v").sum().alias("s"), len().alias("n")]);
        let Some(report) = check_backend_unsorted(&lf, backend, &keys, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn filter_bool_column_property(
        values in proptest::collection::vec(proptest::option::of(any::<bool>()), 0..32),
    ) {
        let height = values.len();
        let frame = DataFrame::new_infer_height(vec![
            Series::new("flag".into(), values).into(),
            Series::new("a".into(), (0..height).map(|i| i as i32).collect::<Vec<i32>>()).into(),
        ]).unwrap_or(DataFrame::empty_with_height(0));
        let lf = frame.lazy().filter(col("flag"));
        for backend in all_backends() {
            prop_assert!(
                check_backend(&lf, backend, &rules()).is_ok(),
                "backend {backend:?} diverged on bool filter"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn filter_single_column_i32_matches_oracle(
        values in proptest::collection::vec(proptest::option::of(any::<i32>()), 0..32),
        op_idx in 0..6_u8,
        threshold in any::<i32>(),
    ) {
        let frame = DataFrame::new_infer_height(vec![
            Series::new("a".into(), values).into(),
        ]).unwrap_or(DataFrame::empty_with_height(0));
        let lf = match op_idx {
            0 => frame.lazy().filter(col("a").gt(lit(threshold))),
            1 => frame.lazy().filter(col("a").lt(lit(threshold))),
            2 => frame.lazy().filter(col("a").eq(lit(threshold))),
            3 => frame.lazy().filter(col("a").neq(lit(threshold))),
            4 => frame.lazy().filter(col("a").gt_eq(lit(threshold))),
            5 => frame.lazy().filter(col("a").lt_eq(lit(threshold))),
            _ => unreachable!(),
        };
        for backend in all_backends() {
            prop_assert!(
                check_backend(&lf, backend, &rules()).is_ok(),
                "backend {backend:?} diverged on op_idx={op_idx}, threshold={threshold}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn group_tiny_frame_matches_oracle(
        keys in proptest::collection::vec(0_i32..4, 0..16),
        values in proptest::collection::vec(proptest::option::of(-100_i64..100_i64), 0..16),
    ) {
        let height = keys.len().max(values.len());
        if height == 0 {
            return Ok(());
        }
        let mut aligned_keys = keys;
        aligned_keys.resize(height, 0);
        let mut aligned_values = values;
        aligned_values.resize(height, None);
        let frame = DataFrame::new_infer_height(vec![
            Series::new("g".into(), aligned_keys).into(),
            Series::new("v".into(), aligned_values).into(),
        ]).unwrap_or(DataFrame::empty_with_height(0));
        let lf = frame.lazy().group_by([col("g")]).agg([
            col("v").sum().alias("s"),
            col("v").count().alias("c"),
            len().alias("n"),
        ]);
        let key_names = vec!["g".to_string()];
        for backend in all_backends() {
            prop_assert!(
                check_backend_unsorted(&lf, backend, &key_names, &rules()).is_ok(),
                "backend {backend:?} diverged on tiny frame"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn f64_filter_matches_oracle_with_fallback(
        values in proptest::collection::vec(proptest::option::of(any::<f64>()), 0..32),
        threshold in any::<f64>(),
    ) {
        let frame = DataFrame::new_infer_height(vec![
            Series::new("f".into(), values).into(),
        ]).unwrap_or(DataFrame::empty_with_height(0));
        let lf = frame.lazy().filter(col("f").gt(lit(threshold)));
        for backend in all_backends() {
            prop_assert!(
                check_or_fallback(&lf, backend, &rules(), true).is_ok(),
                "backend {backend:?} diverged on f64 filter"
            );
        }
    }
}
