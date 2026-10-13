//! M0 differential tests: `filter` correctness against the Polars CPU oracle.
//!
//! Every test runs the same [`LazyFrame`] through the oracle and through the
//! engine in strict mode, asserts the engine engaged, and compares row for
//! row in original order.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_backend, check_fallback_backend,
};
use polars_lazy::{
    dsl::{col, lit},
    frame::{IntoLazy, LazyFrame},
};
use proptest::prelude::*;

fn rules() -> CompareRules {
    CompareRules::default()
}

fn int_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into(),
        Series::new("b".into(), &[Some(10_i64), None, Some(30), Some(40), None]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn filter_greater_than_matches_oracle_on_all_backends() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("a").gt(lit(2)));
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert_eq!(report.ops_ran(), &[Op::Filter]);
        }
    }
    Ok(())
}

#[test]
fn filter_with_nulls_excludes_null_rows() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("b").gt(lit(15)));
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert_eq!(report.ops_ran(), &[Op::Filter]);
        }
    }
    Ok(())
}

#[test]
fn filter_all_out_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("a").gt(lit(100)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn filter_all_in_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("a").gt(lit(0)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn empty_input_filter_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().clear().lazy().filter(col("a").gt(lit(0)));
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn unsupported_query_falls_back_cleanly() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let other = int_frame();
        let lf: LazyFrame =
            int_frame().lazy().join(other.lazy(), [col("a")], [col("a")], Default::default());
        let _ = check_fallback_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

proptest! {
#[test]
fn filter_i32_threshold_property(
        values in proptest::collection::vec(proptest::option::of(any::<i32>()), 0..256),
        threshold in any::<i32>(),
    ) {
        for backend in all_backends() {
            let frame = DataFrame::new_infer_height(vec![
                Series::new("a".into(), values.clone()).into(),
            ]).unwrap_or(DataFrame::empty_with_height(0));
            let lf = frame.lazy().filter(col("a").gt(lit(threshold)));
            let result = check_backend(&lf, backend, &rules());
            prop_assert!(result.is_ok(), "backend {backend:?} diverged: {result:?}");
        }
    }
}

proptest! {
#[test]
fn filter_u32_threshold_property(
        values in proptest::collection::vec(proptest::option::of(any::<u32>()), 0..256),
        threshold in any::<u32>(),
    ) {
        for backend in all_backends() {
            let frame = DataFrame::new_infer_height(vec![
                Series::new("a".into(), values.clone()).into(),
            ]).unwrap_or(DataFrame::empty_with_height(0));
            let lf = frame.lazy().filter(col("a").gt(lit(threshold)));
            let result = check_backend(&lf, backend, &rules());
            prop_assert!(result.is_ok(), "backend {backend:?} diverged on u32: {result:?}");
        }
    }
}

proptest! {
#[test]
fn filter_f32_threshold_property(
        values in proptest::collection::vec(proptest::option::of(any::<f32>()), 0..128),
        threshold in any::<f32>(),
    ) {
        let frame = DataFrame::new_infer_height(vec![
            Series::new("f".into(), values).into(),
        ]).unwrap_or(DataFrame::empty_with_height(0));
        let lf = frame.lazy().filter(col("f").gt(lit(threshold)));
        for backend in all_backends() {
            let result = check_backend(&lf, backend, &rules());
            prop_assert!(result.is_ok(), "backend {backend:?} diverged on f32: {result:?}");
        }
    }
}
