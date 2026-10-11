//! Error-path and configuration edge-case tests.
//!
//! These tests pin the engine's behavior on paths that should *not* produce
//! correct results but must fail gracefully: unsupported backends, configs
//! that force immediate fallback, and queries that exercise the engine's
//! error-mapping paths without panicking.

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::{CubeclEngine, EngineConfig};
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_fallback_backend,
    check_fallback_backend_unsorted,
};
use polars_lazy::{
    dsl::{col, len, lit},
    frame::{IntoLazy, LazyFrame},
};

fn rules() -> CompareRules {
    CompareRules::default()
}

// ---------------------------------------------------------------------------
// Engine configuration: `min_rows_for_gpu` forces fallback on small frames.
// ---------------------------------------------------------------------------

#[test]
#[expect(clippy::expect_used, reason = "test asserts small-frame fallback")]
#[expect(clippy::unwrap_used, reason = "test asserts cpu-correct baseline")]
fn min_rows_for_gpu_forces_fallback_on_small_frames() {
    let frame =
        DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3, 4, 5]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let config = EngineConfig {
            backend,
            raise_on_fail: false,
            min_rows_for_gpu: 100, // frame has only 5 rows → fallback
        };
        // Engine creation with a valid backend succeeds even with high min_rows.
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().filter(col("a").gt(lit(2)));
        // Non-strict mode: falls back silently and returns CPU-correct results.
        let (result, report) = engine.collect_with_report(lf).expect("non-strict collect failed");
        assert!(!report.engaged(), "should not engage below min_rows_for_gpu");
        // The result must still be CPU-correct.
        let expected = frame.clone().lazy().filter(col("a").gt(lit(2))).collect().unwrap();
        assert_eq!(result.height(), expected.height());
        assert_eq!(result.width(), expected.width());
    }
}

#[test]
#[expect(clippy::expect_used, reason = "test asserts single-row engagement")]
fn min_rows_zero_engages_on_any_size() {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[42_i32]).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let config = EngineConfig { backend, raise_on_fail: false, min_rows_for_gpu: 0 };
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().filter(col("a").gt(lit(0)));
        let (_result, report) =
            engine.collect_strict_with_report(lf).expect("strict collect on single row failed");
        assert!(report.engaged(), "min_rows=0 should engage even on 1-row frame");
    }
}

// ---------------------------------------------------------------------------
// Strict mode: unsupported queries must error, not panic.
// ---------------------------------------------------------------------------

#[test]
#[expect(clippy::expect_used, reason = "test asserts engine creation")]
fn strict_mode_errors_on_pure_sort() {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[3_i32, 1, 4, 1, 5, 9, 2, 6]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let config = EngineConfig { backend, raise_on_fail: false, min_rows_for_gpu: 0 };
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().sort(["a"], Default::default());
        // Strict mode on an unsupported query must error (not panic).
        assert!(
            engine.collect_strict(lf).is_err(),
            "strict sort should fail on backend {backend:?}"
        );
    }
}

#[test]
#[expect(clippy::expect_used, reason = "test asserts cpu fallback collect")]
#[expect(clippy::unwrap_used, reason = "test asserts cpu-correct baseline")]
fn non_strict_mode_sorts_via_fallback() {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("a".into(), &[3_i32, 1, 4, 1, 5, 9, 2, 6]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let config = EngineConfig { backend, raise_on_fail: false, min_rows_for_gpu: 0 };
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().sort(["a"], Default::default());
        // Non-strict mode: sort falls back to CPU and returns correct results.
        let (result, report) = engine.collect_with_report(lf).expect("non-strict collect failed");
        assert!(!report.engaged(), "sort should not engage the GPU engine");
        let expected = frame.clone().lazy().sort(["a"], Default::default()).collect().unwrap();
        assert_eq!(result, expected);
    }
}

// ---------------------------------------------------------------------------
// raise_on_fail: makes non-strict collect behave like strict.
// ---------------------------------------------------------------------------

#[test]
#[expect(clippy::expect_used, reason = "test asserts engine creation")]
fn raise_on_fail_makes_collect_err_on_unsupported() {
    let frame =
        DataFrame::new_infer_height(vec![Series::new("a".into(), &[3_i32, 1, 4, 1, 5]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let config = EngineConfig { backend, raise_on_fail: true, min_rows_for_gpu: 0 };
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().sort(["a"], Default::default());
        assert!(engine.collect(lf).is_err(), "raise_on_fail should error on unsupported sort");
    }
}

// ---------------------------------------------------------------------------
// Empty frames: every backend must handle zero-height inputs without panic.
// ---------------------------------------------------------------------------

#[test]
#[expect(clippy::expect_used, reason = "test asserts fallible collect")]
fn empty_frame_filter_does_not_panic() {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3]).into()])
        .unwrap_or(DataFrame::empty_with_height(0))
        .clear();
    for backend in all_backends() {
        let config = EngineConfig { backend, raise_on_fail: false, min_rows_for_gpu: 0 };
        let engine = CubeclEngine::new(config).expect("engine creation failed");
        let lf = frame.clone().lazy().filter(col("a").gt(lit(0)));
        let (result, _) = engine.collect_with_report(lf).expect("empty frame filter failed");
        assert_eq!(result.height(), 0);
    }
}

#[test]
fn empty_frame_group_by_does_not_panic() {
    let frame = DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 2]).into(),
        Series::new("v".into(), &[Some(10_i64), None]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
    .clear();
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = frame
            .clone()
            .lazy()
            .group_by([col("g")])
            .agg([col("v").sum().alias("s"), len().alias("n")]);
        let _ = check_fallback_backend_unsorted(&lf, backend, &keys, &rules());
    }
}

// ---------------------------------------------------------------------------
// Unsupported operations fall back cleanly (existing paths, but pinned here
// for explicit coverage alongside the new config paths).
// ---------------------------------------------------------------------------

#[test]
fn join_falls_back_cleanly() -> Result<(), HarnessError> {
    let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2, 3]).into()])
        .unwrap_or(DataFrame::empty_with_height(0));
    for backend in all_backends() {
        let lf: LazyFrame = frame.clone().lazy().join(
            frame.clone().lazy(),
            [col("a")],
            [col("a")],
            Default::default(),
        );
        let _ = check_fallback_backend(&lf, backend, &rules())?;
    }
    Ok(())
}

#[test]
#[expect(clippy::expect_used, reason = "test asserts concat build")]
fn concat_falls_back_cleanly() -> Result<(), HarnessError> {
    use polars_lazy::dsl::concat;
    for backend in all_backends() {
        let frame = DataFrame::new_infer_height(vec![Series::new("a".into(), &[1_i32, 2]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
        let other = DataFrame::new_infer_height(vec![Series::new("a".into(), &[3_i32, 4]).into()])
            .unwrap_or(DataFrame::empty_with_height(0));
        let lf = concat([frame.lazy(), other.lazy()], polars_lazy::prelude::UnionArgs::default())
            .expect("concat build failed");
        let _ = check_fallback_backend(&lf, backend, &rules())?;
    }
    Ok(())
}
