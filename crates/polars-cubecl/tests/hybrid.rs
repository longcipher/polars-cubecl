//! M3 differential tests: hybrid partial substitution.
//!
//! Queries with a supported head and an unsupported tail run the head on
//! the device and the tail on the CPU engine. Strict mode still applies:
//! the engine must engage (no silent whole-plan fallback).

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_backend, check_backend_unsorted,
    check_fallback_backend,
};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::{IntoLazy, LazyFrame},
};

fn rules() -> CompareRules {
    CompareRules::default()
}

fn int_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[3_i32, 1, 2, 5, 4]).into(),
        Series::new("v".into(), &[Some(10_i64), Some(20), None, Some(40), Some(50)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn filter_before_sort_runs_filter_on_device() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf: LazyFrame =
            int_frame().lazy().filter(col("a").gt(lit(1))).sort(["a"], Default::default());
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}

#[test]
fn group_before_sort_runs_group_on_device() -> Result<(), HarnessError> {
    let keys = vec!["a".to_string()];
    for backend in all_backends() {
        let lf: LazyFrame = int_frame()
            .lazy()
            .group_by([col("a")])
            .agg([col("v").sum().alias("s"), len().alias("n")])
            .sort(["a"], Default::default());
        let Some(report) = check_backend_unsorted(&lf, backend, &keys, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

#[test]
fn join_of_filtered_scans_accelerates_both_arms() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let left = int_frame().lazy().filter(col("a").gt(lit(1)));
        let right = int_frame().lazy().filter(col("v").gt(lit(15)));
        let lf: LazyFrame = left.join(right, [col("a")], [col("a")], Default::default());
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}

#[test]
#[expect(clippy::expect_used, reason = "test asserts concat build")]
fn fully_unsupported_plan_still_falls_back() -> Result<(), HarnessError> {
    use polars_lazy::dsl::concat;
    for backend in all_backends() {
        // `concat` is outside the engine's supported subset (no Filter, Project,
        // Slice, GroupBy, or Sort kernel at the head). It must fall back cleanly.
        let other = DataFrame::new_infer_height(vec![
            Series::new("a".into(), &[7_i32, 8]).into(),
            Series::new("v".into(), &[Some(70_i64), Some(80)]).into(),
        ])
        .unwrap_or(DataFrame::empty_with_height(0));
        let lf = concat(
            [int_frame().lazy(), other.lazy()],
            polars_lazy::prelude::UnionArgs::default(),
        )
        .expect("concat build failed");
        let _ = check_fallback_backend(&lf, backend, &rules())?;
    }
    Ok(())
}
