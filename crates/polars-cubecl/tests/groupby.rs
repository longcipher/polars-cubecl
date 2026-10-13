//! M1 differential tests: grouped aggregation against the oracle.
//!
//! Group order is out of contract without `maintain_order`, so both sides
//! are sorted by key before comparison. Strict mode plus engagement
//! assertions still apply.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_fallback_backend, check_unsorted_or_fallback,
};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::{IntoLazy, LazyFrame},
};
use proptest::prelude::*;

fn rules() -> CompareRules {
    CompareRules::default()
}

fn check_unsorted(lf: LazyFrame, keys: &[&str], uses_f64_math: bool) -> Result<(), HarnessError> {
    let key_names: Vec<String> = keys.iter().map(|key| (*key).to_string()).collect();
    for backend in all_backends() {
        let Some(report) =
            check_unsorted_or_fallback(&lf, backend, &key_names, &rules(), uses_f64_math)?
        else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

fn group_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 2, 1, 2, 1, 3]).into(),
        Series::new("v".into(), &[Some(10_i64), Some(20), None, Some(40), Some(50), None]).into(),
        Series::new("w".into(), &[1.5_f64, f64::NAN, 2.5, 3.5, -0.0, 4.5]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn group_sum_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame().lazy().group_by([col("g")]).agg([col("v").sum().alias("s")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_min_max_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame()
        .lazy()
        .group_by([col("g")])
        .agg([col("v").min().alias("lo"), col("v").max().alias("hi")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_mean_count_len_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame().lazy().group_by([col("g")]).agg([
        col("v").mean().alias("m"),
        col("v").count().alias("c"),
        len().alias("n"),
    ]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_first_last_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame()
        .lazy()
        .group_by([col("g")])
        .agg([col("v").first().alias("f"), col("v").last().alias("l")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_float_min_max_sum_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame().lazy().group_by([col("g")]).agg([
        col("w").min().alias("lo"),
        col("w").max().alias("hi"),
        col("w").sum().alias("s"),
    ]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_null_keys_form_own_group() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[Some(1_i32), None, Some(1), None]).into(),
        Series::new("v".into(), &[10_i64, 20, 30, 40]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([col("v").sum().alias("s")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_all_null_values_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 1, 2]).into(),
        Series::new("v".into(), &[None, None, Some(5_i64)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([
        col("v").sum().alias("s"),
        col("v").min().alias("lo"),
        col("v").max().alias("hi"),
        col("v").mean().alias("m"),
        col("v").count().alias("c"),
        len().alias("n"),
    ]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_empty_input_matches_oracle() -> Result<(), HarnessError> {
    // `mean` needs `f64` arithmetic, so backends without `double` fall back;
    // the empty comparison is order-free either way.
    let lf = group_frame().clear().lazy().group_by([col("g")]).agg([
        col("v").sum().alias("s"),
        col("v").mean().alias("m"),
        len().alias("n"),
    ]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_f64_movement_matches_oracle() -> Result<(), HarnessError> {
    // `first`/`last`/`count`/`len` over `f64` need no arithmetic: values move
    // through bitwise `u64` kernels, so this engages on every backend
    // including Apple GPUs. `f64` keys group on the host exactly.
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[Some(0.0_f64), Some(-0.0), Some(f64::NAN), None, Some(0.0)])
            .into(),
        Series::new("w".into(), &[Some(1.5_f64), None, Some(f64::NAN), Some(2.5), Some(-0.0)])
            .into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([
        col("w").first().alias("f"),
        col("w").last().alias("l"),
        col("w").count().alias("c"),
        len().alias("n"),
    ]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_multi_key_matches_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 1, 2, 2, 1]).into(),
        Series::new("h".into(), &[Some(true), None, Some(true), Some(false), Some(true)]).into(),
        Series::new("v".into(), &[10_i64, 20, 30, 40, 50]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame
        .lazy()
        .group_by([col("g"), col("h")])
        .agg([col("v").sum().alias("s"), len().alias("n")]);
    check_unsorted(lf, &["g", "h"], false)
}

#[test]
fn group_filter_combo_matches_oracle() -> Result<(), HarnessError> {
    let lf = group_frame()
        .lazy()
        .filter(col("v").gt(lit(15)))
        .group_by([col("g")])
        .agg([col("v").sum().alias("s"), col("v").mean().alias("m")]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_then_filter_matches_oracle() -> Result<(), HarnessError> {
    // Group order is out of contract; sort by key before comparing.
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = group_frame()
            .lazy()
            .group_by([col("g")])
            .agg([col("v").sum().alias("s")])
            .filter(col("s").gt(lit(20)));
        let Some(report) = check_unsorted_or_fallback(&lf, backend, &keys, &rules(), false)? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
        assert!(report.ops_ran().contains(&Op::Filter));
    }
    Ok(())
}

#[test]
fn group_u64_keys_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[u64::MAX, 0_u64, u64::MAX, 7, 0]).into(),
        Series::new("v".into(), &[1_i32, 2, 3, 4, 5]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([col("v").sum().alias("s"), len().alias("n")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_f32_keys_match_oracle() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[Some(0.0_f32), Some(-0.0), Some(f32::NAN), None, Some(0.0)])
            .into(),
        Series::new("v".into(), &[1_i64, 2, 3, 4, 5]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([col("v").sum().alias("s"), len().alias("n")]);
    check_unsorted(lf, &["g"], false)
}

#[test]
fn group_large_int_mean_matches_oracle() -> Result<(), HarnessError> {
    let base = 4_503_599_627_370_496_i64;
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 1, 2]).into(),
        Series::new("v".into(), &[base, base + 512, base + 1]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame
        .lazy()
        .group_by([col("g")])
        .agg([col("v").mean().alias("m"), col("v").sum().alias("s")]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_all_nan_min_max_yield_null() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 1, 2]).into(),
        Series::new("w".into(), &[Some(f64::NAN), Some(f64::NAN), Some(1.5)]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    let lf = frame.lazy().group_by([col("g")]).agg([
        col("w").min().alias("lo"),
        col("w").max().alias("hi"),
        col("w").sum().alias("s"),
        col("w").mean().alias("m"),
    ]);
    check_unsorted(lf, &["g"], true)
}

#[test]
fn group_stable_order_falls_back() -> Result<(), HarnessError> {
    for backend in all_backends() {
        let lf = group_frame().lazy().group_by_stable([col("g")]).agg([col("v").sum().alias("s")]);
        let _ = check_fallback_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
fn group_with_trailing_sort_runs_group_on_device() -> Result<(), HarnessError> {
    // The sort tail stays on the CPU engine; the group-by head engages.
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = group_frame()
            .lazy()
            .group_by([col("g")])
            .agg([col("v").sum().alias("s")])
            .sort(["g"], Default::default());
        let Some(report) = check_unsorted_or_fallback(&lf, backend, &keys, &rules(), false)? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

fn largish_frame(rows: usize, cardinality: usize) -> DataFrame {
    let keys: Vec<Option<i32>> = (0..rows)
        .map(|index| if index % 29 == 7 { None } else { Some((index % cardinality.max(1)) as i32) })
        .collect();
    let values: Vec<Option<i64>> = (0..rows)
        .map(|index| {
            if index % 11 == 5 {
                None
            } else {
                Some((index as i64).wrapping_mul(7919).wrapping_add(index as i64))
            }
        })
        .collect();
    DataFrame::new_infer_height(vec![
        Series::new("g".into(), keys).into(),
        Series::new("v".into(), values).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

#[test]
fn group_large_cardinality_scales_match_oracle() -> Result<(), HarnessError> {
    for (rows, cardinality) in [(10_000, 1), (10_000, 64), (10_000, 9_999), (100_000, 128)] {
        let frame = largish_frame(rows, cardinality);
        let lf = frame.lazy().group_by([col("g")]).agg([
            col("v").sum().alias("s"),
            col("v").mean().alias("m"),
            col("v").min().alias("lo"),
            col("v").max().alias("hi"),
            col("v").count().alias("c"),
            len().alias("n"),
        ]);
        check_unsorted(lf, &["g"], true)?;
    }
    Ok(())
}

proptest! {
    #[test]
    fn groupby_agg_property(
        rows in 0..512_usize,
        cardinality in any::<u8>(),
        // Small magnitudes: the debug-mode oracle panics on `i64` overflow,
        // so overflow equivalence is out of scope for this property.
        values in proptest::collection::vec(
            proptest::option::of(-1_000_i64..1_000_i64),
            0..512,
        ),
    ) {
        let cardinality = (usize::from(cardinality) % 17).max(1);
        let keys: Vec<Option<i32>> = (0..rows)
            .map(|index| {
                if index % 13 == 4 {
                    None
                } else {
                    Some((index % cardinality) as i32)
                }
            })
            .collect();
        let mut aligned: Vec<Option<i64>> = values;
        aligned.resize(rows, None);
        let frame = DataFrame::new_infer_height(vec![
            Series::new("g".into(), keys).into(),
            Series::new("v".into(), aligned).into(),
        ])
        .unwrap_or(DataFrame::empty_with_height(0));
        let lf = frame.lazy().group_by([col("g")]).agg([
            col("v").sum().alias("s"),
            col("v").mean().alias("m"),
            col("v").min().alias("lo"),
            col("v").max().alias("hi"),
            col("v").count().alias("c"),
            col("v").first().alias("f"),
            col("v").last().alias("l"),
            len().alias("n"),
        ]);
        let result = check_unsorted(lf, &["g"], true);
        prop_assert!(result.is_ok(), "diverged: {result:?}");
    }
}
