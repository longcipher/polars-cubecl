//! M0 differential tests: `select` (projection) and `slice` against the oracle.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{CompareRules, HarnessError, all_backends, check_backend};
use polars_lazy::{dsl::col, frame::IntoLazy};

fn rules() -> CompareRules {
    CompareRules::default()
}

fn int_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into(),
        Series::new("b".into(), &[Some(10_i64), None, Some(30), Some(40), None]).into(),
        Series::new("c".into(), &[true, false, true, false, true]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn unfiltered_passthrough_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy();
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().is_empty(), "passthrough query unexpectedly engaged");
        }
    }
    Ok(())
}

#[test]
fn select_subset_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().select([col("b"), col("a")]);
        // The optimizer may fold the whole projection into the scan, leaving
        // a bare `DataFrameScan`; engagement plus value equality is the check.
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn select_with_rename_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().select([col("a").alias("x"), col("a").alias("y")]);
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn filter_then_select_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("a").gt(2)).select([col("b"), col("c")]);
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}

#[test]
fn slice_middle_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().slice(1, 3);
        let _ = check_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn slice_boundaries_match_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        for (offset, len) in [(0_i64, 0_u32), (0, 5), (5, 0), (3, 100), (5, 5), (-2, 2), (-100, 3)]
        {
            let lf = int_frame().lazy().slice(offset, len);
            let _ = check_backend(&lf, backend, &rules())?;
        }
    }
    Ok(())
}

#[test]
fn filter_then_slice_matches_oracle() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = int_frame().lazy().filter(col("c").eq(true)).slice(0, 1);
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert_eq!(report.ops_ran(), &[Op::Filter, Op::Slice]);
        }
    }
    Ok(())
}
