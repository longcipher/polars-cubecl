//! Differential integration tests for unary math functions.
//!
//! Each test compares the GPU-accelerated engine against the CPU oracle for
//! `select(abs(x))`, `select(ceil(x))`, etc. — across backends, literal
//! shapes, and a range of float inputs including ±0, ±inf, NaN, and values
//! that stress float boundaries.

use std::f64::consts::PI;
use std::ops::Neg;

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{CompareRules, all_backends, check_backend, check_quiet_fallback};
use polars_lazy::{
    dsl::col,
    frame::IntoLazy,
};

fn rules() -> CompareRules {
    CompareRules::default()
}

// ---------------------------------------------------------------------------
// Abs
// ---------------------------------------------------------------------------

#[test]
fn select_abs_float_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1.0_f64, -2.5, 0.0, PI, -0.0],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").abs().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// Note: `abs`/`sign` over an Int32 column is not currently lowered into the
// engine subset — the kernel requires float input, and the lowering does not
// auto-widen Int32 → Float64.  The float-column test above covers the lowered
// path; int-column abs/sign stays on the CPU engine where it belongs.

// ---------------------------------------------------------------------------
// Ceil
// ---------------------------------------------------------------------------

#[test]
fn select_ceil_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1.0_f64, -2.5, 0.0, PI, -0.1, f64::INFINITY, f64::NEG_INFINITY],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").ceil().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Floor
// ---------------------------------------------------------------------------

#[test]
fn select_floor_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1.0_f64, -2.5, 0.0, PI, -0.1],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").floor().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Exp
// ---------------------------------------------------------------------------

#[test]
fn select_exp_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, 1.0, -1.0, 2.5, -0.5],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").exp().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Log1p
// ---------------------------------------------------------------------------

#[test]
fn select_log1p_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, 1.0, -0.5, 2.5, std::f64::consts::E - 1.0],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").log1p().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sin
// ---------------------------------------------------------------------------

#[test]
fn select_sin_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, PI / 2.0, PI, -PI / 2.0, 1.0],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").sin().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cos
// ---------------------------------------------------------------------------

#[test]
fn select_cos_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, PI / 2.0, PI, -PI / 2.0, 1.0],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").cos().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sqrt
// ---------------------------------------------------------------------------

#[test]
fn select_sqrt_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, 1.0, 4.0, 2.25, 100.0],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").sqrt().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sign
// ---------------------------------------------------------------------------

#[test]
fn select_sign_float_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1.0_f64, -2.5, 0.0, -0.0, f64::NAN],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").sign().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

#[test]
fn select_sign_int_matches_oracle_via_fallback() -> Result<(), polars_cubecl_testing::HarnessError> {
    // `sign` over an int column is a float-only math op on a non-float
    // column — the lowering rejects it, and the engine cleanly falls back
    // to CPU.  `check_quiet_fallback` asserts the result is CPU-correct and
    // that the engine did *not* engage.
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1_i32, -5, 0, 42, -1],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").sign().alias("y")]);
        let _ = check_quiet_fallback(&lf, backend, &rules())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Negate
// ---------------------------------------------------------------------------

#[test]
fn select_negate_int_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[1_i32, -5, 0, 42],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .select([col("x").neg().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Nested math: sin(abs(x)) — composition of two unary math ops.
// ---------------------------------------------------------------------------

#[test]
fn select_nested_sin_abs_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[0.0_f64, -1.0, 2.5, -PI],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .select([col("x").abs().sin().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Math in a multi-column projection: combine math with arithmetic.
// ---------------------------------------------------------------------------

#[test]
fn select_math_plus_arithmetic_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    // Use float columns for both operands so the projection stays in the
    // engine subset without non-identity casts inside arithmetic.
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[1.0_f64, -2.0, 3.0, -4.0]).into(),
        Series::new("b".into(), &[10.0_f64, 20.0, 30.0, 40.0]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .select([
                col("a").abs().alias("abs_a"),
                (col("a").abs() + col("b")).alias("sum"),
            ]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Math with nulls: abs over a nullable float column.
// ---------------------------------------------------------------------------

#[test]
fn select_abs_with_nulls_matches_oracle() -> Result<(), polars_cubecl_testing::HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new(
        "x".into(),
        &[Some(1.0_f64), None, Some(-2.5), None, Some(0.0)],
    )
    .into()])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf = frame.clone().lazy().select([col("x").abs().alias("y")]);
        let Some(report) = check_backend(&lf, backend, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::Project));
    }
    Ok(())
}
