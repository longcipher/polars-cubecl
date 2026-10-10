//! M0 differential tests: predicate shapes and dtype coverage.
//!
//! Every comparison operator, boolean combination, and supported dtype is
//! driven against the oracle, including `NaN`/null edge cases and row counts
//! around the kernel threading boundaries.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{CompareRules, HarnessError, all_backends, check_backend};
use polars_lazy::{
    dsl::{col, lit},
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

fn sized_i32(rows: usize) -> DataFrame {
    let values: Vec<Option<i32>> = (0..rows)
        .map(|index| {
            if index % 7 == 3 {
                None
            } else {
                Some((index as i32).wrapping_mul(1_234_567).wrapping_add(index as i32))
            }
        })
        .collect();
    DataFrame::new_infer_height(vec![Series::new("a".into(), values).into()])
        .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn all_comparison_operators_match_oracle() -> Result<(), HarnessError> {
    let frame = sized_i32(200);
    for threshold in [i32::MIN, -1_000_000, 0, 42, 1_000_000, i32::MAX] {
        check(frame.clone().lazy().filter(col("a").eq(lit(threshold))))?;
        check(frame.clone().lazy().filter(col("a").neq(lit(threshold))))?;
        check(frame.clone().lazy().filter(col("a").lt(lit(threshold))))?;
        check(frame.clone().lazy().filter(col("a").lt_eq(lit(threshold))))?;
        check(frame.clone().lazy().filter(col("a").gt(lit(threshold))))?;
        check(frame.clone().lazy().filter(col("a").gt_eq(lit(threshold))))?;
    }
    Ok(())
}

#[test]
fn boolean_combinations_match_oracle() -> Result<(), HarnessError> {
    let frame = sized_i32(200);
    check(frame.clone().lazy().filter(col("a").gt(lit(0)).and(col("a").lt(lit(1_000_000)))))?;
    check(
        frame.clone().lazy().filter(col("a").lt(lit(-1_000_000)).or(col("a").gt(lit(1_000_000)))),
    )?;
    check(frame.clone().lazy().filter(col("a").gt(lit(0)).not()))?;
    check(
        frame
            .lazy()
            .filter(col("a").gt(lit(0)).and(col("a").lt(lit(10)).or(col("a").gt(lit(100))))),
    )?;
    Ok(())
}

#[test]
fn row_counts_around_threading_boundaries_match() -> Result<(), HarnessError> {
    for rows in [0, 1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4096, 10_000] {
        let frame = sized_i32(rows);
        check(frame.clone().lazy().filter(col("a").gt(lit(0))))?;
        check(frame.lazy().filter(col("a").lt(lit(0))))?;
    }
    Ok(())
}

#[test]
fn integer_dtypes_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("i64c".into(), &[Some(i64::MIN), None, Some(0), Some(i64::MAX)]).into(),
        Series::new("u32c".into(), &[Some(0_u32), Some(1), None, Some(u32::MAX)]).into(),
        Series::new("u64c".into(), &[Some(u64::MAX), Some(0), None, Some(7)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.clone().lazy().filter(col("i64c").gt(lit(0_i64))))?;
    check(frame.clone().lazy().filter(col("i64c").lt_eq(lit(i64::MAX))))?;
    check(frame.clone().lazy().filter(col("u32c").neq(lit(1_u32))))?;
    check(frame.clone().lazy().filter(col("u64c").gt(lit(6_u64))))?;
    check(frame.lazy().filter(col("u32c").eq(lit(u32::MAX))))?;
    Ok(())
}

#[test]
fn float_predicates_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("f".into(), &[
            Some(0.0_f64),
            Some(-0.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            None,
            Some(0.1),
            Some(1e300),
        ])
        .into(),
        Series::new("g".into(), &[
            Some(0.5_f32),
            None,
            Some(f32::NAN),
            Some(1.5_f32),
            Some(f32::NEG_INFINITY),
            Some(2.5_f32),
            None,
            Some(0.0_f32),
        ])
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.clone().lazy().filter(col("g").gt(lit(0.0_f32))))?;
    check(frame.lazy().filter(col("g").eq(lit(f32::NAN))))?;
    Ok(())
}

#[test]
fn float64_predicates_match_oracle() -> Result<(), HarnessError> {
    // `f64` kernels need a backend with `double` support; elsewhere these
    // pin graceful CPU fallback (see `supports_f64`).
    let frame = float_frame();
    for backend in all_backends() {
        for query in [
            frame.clone().lazy().filter(col("f").gt(lit(0.0))),
            frame.clone().lazy().filter(col("f").lt(lit(0.0))),
            frame.clone().lazy().filter(col("f").eq(lit(f64::NAN))),
            frame.clone().lazy().filter(col("f").neq(lit(f64::NAN))),
        ] {
            let Some(_) =
                polars_cubecl_testing::check_or_fallback(&query, backend, &rules(), true)?
            else {
                continue;
            };
        }
    }
    Ok(())
}

fn float_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("f".into(), &[
            Some(0.0_f64),
            Some(-0.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            None,
            Some(0.1),
            Some(1e300),
        ])
        .into(),
        Series::new("g".into(), &[
            Some(0.5_f32),
            None,
            Some(f32::NAN),
            Some(1.5_f32),
            Some(f32::NEG_INFINITY),
            Some(2.5_f32),
            None,
            Some(0.0_f32),
        ])
        .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn int_column_against_float_literal_matches_oracle() -> Result<(), HarnessError> {
    let frame = sized_i32(500);
    for backend in all_backends() {
        for query in [
            frame.clone().lazy().filter(col("a").gt(lit(0.5))),
            frame.clone().lazy().filter(col("a").lt(lit(-0.5))),
            frame.clone().lazy().filter(col("a").eq(lit(42.0))),
        ] {
            let Some(_) =
                polars_cubecl_testing::check_or_fallback(&query, backend, &rules(), true)?
            else {
                continue;
            };
        }
    }
    Ok(())
}

#[test]
fn boolean_mask_and_comparison_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("flag".into(), &[Some(true), Some(false), None, Some(true)]).into(),
        Series::new("a".into(), &[1_i32, 2, 3, 4]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.clone().lazy().filter(col("flag")))?;
    check(frame.clone().lazy().filter(col("flag").eq(lit(true))))?;
    check(frame.clone().lazy().filter(col("flag").not()))?;
    check(frame.lazy().filter(col("flag").and(col("a").gt(lit(2)))))?;
    Ok(())
}

#[test]
fn filter_reports_filter_op() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = sized_i32(100).lazy().filter(col("a").gt(lit(0)));
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}

#[test]
fn chained_filters_match_oracle() -> Result<(), HarnessError> {
    let frame = sized_i32(500);
    check(frame.lazy().filter(col("a").gt(lit(-100))).filter(col("a").lt(lit(100))))
}

#[test]
fn filter_with_f64_sibling_column_matches_oracle() -> Result<(), HarnessError> {
    // The `f64` column is only moved (gathered bitwise via `u64` kernels),
    // never computed, so this engages on every backend including Apple GPUs.
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1_i32, 2, 3, 4]).into(),
        Series::new("w".into(), &[Some(1.5_f64), None, Some(f64::NAN), Some(-0.0)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    check(frame.clone().lazy().filter(col("a").gt(lit(1))))?;
    check(frame.clone().lazy().select([col("w")]))?;
    check(frame.lazy().slice(1, 2))?;
    Ok(())
}

#[test]
fn select_then_filter_matches_oracle() -> Result<(), HarnessError> {
    let frame = sized_i32(300);
    check(frame.lazy().select([col("a").alias("x")]).filter(col("x").gt(lit(0))))
}
