//! Differential testing harness: Polars CPU is the oracle.
//!
//! Every result produced by the engine is compared against the same
//! [`LazyFrame`] executed by the Polars CPU engine. The comparison rules live
//! here ([`assert_frame_equal`]), together with the engagement guards that
//! keep a silently-falling-back engine from passing vacuously, the input
//! generators, and the backend matrix runner.

use polars_core::{
    datatypes::{CompatLevel, DataType},
    frame::DataFrame,
    prelude::Series,
};
pub use polars_cubecl::{Backend, BackendKind, CubeclEngine, EngineConfig, ExecReport, Op};
use polars_lazy::{frame::LazyFrame, prelude::IntoLazy};
use proptest::prelude::*;
use thiserror::Error;

/// Tolerance rules for floating-point comparison.
///
/// Tolerance is scoped to floats only: integers and booleans always compare
/// exactly, so it cannot mask an integer bug.
#[derive(Clone, Copy, Debug)]
pub struct CompareRules {
    /// Relative tolerance for `f32`.
    pub rtol_f32: f32,
    /// Absolute tolerance for `f32`.
    pub atol_f32: f32,
    /// Relative tolerance for `f64`.
    pub rtol_f64: f64,
    /// Absolute tolerance for `f64`.
    pub atol_f64: f64,
}

impl Default for CompareRules {
    fn default() -> Self {
        Self { rtol_f32: 1e-6, atol_f32: 1e-5, rtol_f64: 1e-12, atol_f64: 1e-12 }
    }
}

/// Error returned when two frames differ.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CompareError {
    /// Different number of columns.
    #[error("width mismatch: actual {actual}, expected {expected}")]
    Width {
        /// Actual column count.
        actual: usize,
        /// Expected column count.
        expected: usize,
    },
    /// Column names differ (order matters).
    #[error("column name mismatch at {index}: actual {actual}, expected {expected}")]
    Name {
        /// Column position.
        index: usize,
        /// Actual name.
        actual: String,
        /// Expected name.
        expected: String,
    },
    /// Column dtypes differ exactly.
    #[error("dtype mismatch for column {column}: actual {actual:?}, expected {expected:?}")]
    DType {
        /// Column name.
        column: String,
        /// Actual dtype.
        actual: DataType,
        /// Expected dtype.
        expected: DataType,
    },
    /// Different number of rows.
    #[error("height mismatch: actual {actual}, expected {expected}")]
    Height {
        /// Actual row count.
        actual: usize,
        /// Expected row count.
        expected: usize,
    },
    /// Null positions differ.
    #[error("null mismatch for column {column} at row {row}")]
    Null {
        /// Column name.
        column: String,
        /// Row index.
        row: usize,
    },
    /// Values differ.
    #[error(
        "value mismatch for column {column} at row {row}: actual {actual}, expected {expected}"
    )]
    Value {
        /// Column name.
        column: String,
        /// Row index.
        row: usize,
        /// Actual value rendered as a string.
        actual: String,
        /// Expected value rendered as a string.
        expected: String,
    },
    /// A column dtype the comparator does not cover.
    #[error("comparator does not cover dtype {0:?}")]
    UnsupportedColumn(DataType),
}

/// Compare two frames: schema strictly, values under [`CompareRules`].
///
/// Order-preserving operators must match row-for-row in original order; call
/// [`sort_by_keys`] on both sides first when order is out of contract
/// (unordered `group_by`).
pub fn assert_frame_equal(
    actual: &DataFrame,
    expected: &DataFrame,
    rules: &CompareRules,
) -> Result<(), CompareError> {
    if actual.width() != expected.width() {
        return Err(CompareError::Width { actual: actual.width(), expected: expected.width() });
    }
    if actual.height() != expected.height() {
        return Err(CompareError::Height { actual: actual.height(), expected: expected.height() });
    }
    let actual_columns = actual.columns();
    let expected_columns = expected.columns();
    for (index, (left, right)) in actual_columns.iter().zip(expected_columns.iter()).enumerate() {
        if left.name() != right.name() {
            return Err(CompareError::Name {
                index,
                actual: left.name().to_string(),
                expected: right.name().to_string(),
            });
        }
        if left.dtype() != right.dtype() {
            return Err(CompareError::DType {
                column: left.name().to_string(),
                actual: left.dtype().clone(),
                expected: right.dtype().clone(),
            });
        }
        compare_series(left.as_materialized_series(), right.as_materialized_series(), rules)?;
    }
    Ok(())
}

/// Sort a frame by key columns (ascending), for order-free comparison.
pub fn sort_by_keys(frame: &DataFrame, keys: &[String]) -> Result<DataFrame, String> {
    frame
        .clone()
        .lazy()
        .sort_by_exprs(
            keys.iter().map(polars_lazy::dsl::col).collect::<Vec<_>>(),
            polars_core::prelude::SortMultipleOptions::default(),
        )
        .collect()
        .map_err(|error| error.to_string())
}

fn compare_series(left: &Series, right: &Series, rules: &CompareRules) -> Result<(), CompareError> {
    debug_assert_eq!(left.len(), right.len());
    let column = left.name().to_string();
    match left.dtype() {
        DataType::Int32 => compare_ints::<i32>(left, right, &column),
        DataType::Int64 => compare_ints::<i64>(left, right, &column),
        DataType::UInt32 => compare_ints::<u32>(left, right, &column),
        DataType::UInt64 => compare_ints::<u64>(left, right, &column),
        DataType::Boolean => compare_bools(left, right, &column),
        DataType::Float32 => {
            let (rtol, atol) = (f64::from(rules.rtol_f32), f64::from(rules.atol_f32));
            compare_mapped_floats::<f32>(left, right, rtol, atol, &column)
        }
        DataType::Float64 => {
            compare_mapped_floats::<f64>(left, right, rules.rtol_f64, rules.atol_f64, &column)
        }
        other => Err(CompareError::UnsupportedColumn(other.clone())),
    }
}

fn null_mismatch(column: &str, row: usize, left_ok: bool, right_ok: bool) -> Option<CompareError> {
    (left_ok != right_ok).then(|| CompareError::Null { column: column.to_string(), row })
}

fn compare_ints<T>(left: &Series, right: &Series, column: &str) -> Result<(), CompareError>
where
    T: polars_arrow::types::NativeType + PartialEq + std::fmt::Display,
{
    let (left_values, left_valid) = primitive_parts::<T>(left);
    let (right_values, right_valid) = primitive_parts::<T>(right);
    for (row, ((left_value, left_ok), (right_value, right_ok))) in left_values
        .iter()
        .zip(left_valid.iter())
        .zip(right_values.iter().zip(right_valid.iter()))
        .enumerate()
    {
        if let Some(error) = null_mismatch(column, row, *left_ok, *right_ok) {
            return Err(error);
        }
        if *left_ok && left_value != right_value {
            return Err(CompareError::Value {
                column: column.to_string(),
                row,
                actual: format!("{left_value}"),
                expected: format!("{right_value}"),
            });
        }
    }
    Ok(())
}

fn compare_bools(left: &Series, right: &Series, column: &str) -> Result<(), CompareError> {
    let (left_values, left_valid) = boolean_parts(left);
    let (right_values, right_valid) = boolean_parts(right);
    for (row, ((left_value, left_ok), (right_value, right_ok))) in left_values
        .iter()
        .zip(left_valid.iter())
        .zip(right_values.iter().zip(right_valid.iter()))
        .enumerate()
    {
        if let Some(error) = null_mismatch(column, row, *left_ok, *right_ok) {
            return Err(error);
        }
        if *left_ok && left_value != right_value {
            return Err(CompareError::Value {
                column: column.to_string(),
                row,
                actual: format!("{left_value}"),
                expected: format!("{right_value}"),
            });
        }
    }
    Ok(())
}

fn compare_mapped_floats<T>(
    left: &Series,
    right: &Series,
    rtol: f64,
    atol: f64,
    column: &str,
) -> Result<(), CompareError>
where
    T: polars_arrow::types::NativeType + Copy + Into<f64>,
{
    let (left_values, left_valid) = primitive_parts::<T>(left);
    let (right_values, right_valid) = primitive_parts::<T>(right);
    for (row, ((left_value, left_ok), (right_value, right_ok))) in left_values
        .iter()
        .zip(left_valid.iter())
        .zip(right_values.iter().zip(right_valid.iter()))
        .enumerate()
    {
        if let Some(error) = null_mismatch(column, row, *left_ok, *right_ok) {
            return Err(error);
        }
        if *left_ok && !floats_equal((*left_value).into(), (*right_value).into(), rtol, atol) {
            return Err(CompareError::Value {
                column: column.to_string(),
                row,
                actual: format!("{}", (*left_value).into()),
                expected: format!("{}", (*right_value).into()),
            });
        }
    }
    Ok(())
}

/// `NaN` matches `NaN`; infinities match exactly; otherwise `|a-b| <= atol +
/// rtol*|b|`.
fn floats_equal(left: f64, right: f64, rtol: f64, atol: f64) -> bool {
    if left.is_nan() && right.is_nan() {
        return true;
    }
    if left.is_infinite() || right.is_infinite() {
        // Infinities need exact equality: a tolerance would make a finite
        // value match `+inf` at a huge `atol`.
        #[expect(clippy::float_cmp, reason = "infinities need exact equality")]
        return left == right;
    }
    (left - right).abs() <= rtol.mul_add(right.abs(), atol)
}

fn primitive_parts<T>(series: &Series) -> (Vec<T>, Vec<bool>)
where
    T: polars_arrow::types::NativeType + Clone,
{
    use polars_arrow::array::PrimitiveArray;
    let mut values = Vec::with_capacity(series.len());
    let mut valid = Vec::with_capacity(series.len());
    for chunk_idx in 0..series.n_chunks() {
        let array = series.to_arrow(chunk_idx, CompatLevel::newest());
        if let Some(source) = array.as_any().downcast_ref::<PrimitiveArray<T>>() {
            values.extend_from_slice(source.values().as_slice());
            match source.validity() {
                None => valid.extend(std::iter::repeat_n(true, source.len())),
                Some(bits) => {
                    valid.reserve(source.len());
                    for index in 0..source.len() {
                        valid.push(bits.get_bit(index));
                    }
                }
            }
        }
    }
    (values, valid)
}

fn boolean_parts(series: &Series) -> (Vec<bool>, Vec<bool>) {
    use polars_arrow::array::BooleanArray;
    let mut values = Vec::with_capacity(series.len());
    let mut valid = Vec::with_capacity(series.len());
    for chunk_idx in 0..series.n_chunks() {
        let array = series.to_arrow(chunk_idx, CompatLevel::newest());
        if let Some(source) = array.as_any().downcast_ref::<BooleanArray>() {
            values.extend(source.values().iter());
            match source.validity() {
                None => valid.extend(std::iter::repeat_n(true, source.len())),
                Some(bits) => {
                    valid.reserve(source.len());
                    for index in 0..source.len() {
                        valid.push(bits.get_bit(index));
                    }
                }
            }
        }
    }
    (values, valid)
}

/// Error returned by the differential harness.
#[derive(Debug, Error)]
pub enum HarnessError {
    /// The CPU oracle itself failed (usually a bad test query).
    #[error("oracle query failed: {0}")]
    Oracle(String),
    /// The engine failed in strict mode.
    #[error("engine query failed: {0}")]
    Engine(String),
    /// The engine did not engage: the test would pass vacuously.
    #[error("engine fell back silently on a query it must run")]
    NotEngaged,
    /// Results differ; see the inner comparison error.
    #[error("engine result differs from CPU oracle: {0}")]
    Mismatch(#[from] CompareError),
}

/// Run `lf` on the CPU oracle and on `engine` (strict), and compare.
///
/// Fails when the engine does not engage, so a silently-falling-back engine
/// can never pass.
pub fn assert_engine_matches_cpu(
    lf: LazyFrame,
    engine: &CubeclEngine,
    rules: &CompareRules,
) -> Result<ExecReport, HarnessError> {
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    let (actual, report) = engine
        .collect_strict_with_report(lf)
        .map_err(|error| HarnessError::Engine(error.to_string()))?;
    if !report.engaged() {
        return Err(HarnessError::NotEngaged);
    }
    assert_frame_equal(&actual, &expected, rules)?;
    Ok(report)
}

/// Run `lf` on the CPU oracle and on `engine` (strict), sort both sides by
/// `keys`, and compare.
///
/// For `group_by` without `maintain_order`, where row order is out of
/// contract. Fails when the engine does not engage.
pub fn assert_engine_matches_cpu_unsorted(
    lf: LazyFrame,
    engine: &CubeclEngine,
    keys: &[String],
    rules: &CompareRules,
) -> Result<ExecReport, HarnessError> {
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    let (actual, report) = engine
        .collect_strict_with_report(lf)
        .map_err(|error| HarnessError::Engine(error.to_string()))?;
    if !report.engaged() {
        return Err(HarnessError::NotEngaged);
    }
    let actual = sort_by_keys(&actual, keys).map_err(HarnessError::Engine)?;
    let expected = sort_by_keys(&expected, keys).map_err(HarnessError::Oracle)?;
    assert_frame_equal(&actual, &expected, rules)?;
    Ok(report)
}

/// Assert that an unsupported query falls back cleanly: strict mode errors,
/// non-strict mode returns the CPU-correct result without engaging.
pub fn assert_fallback_for_unsupported(
    lf: LazyFrame,
    engine: &CubeclEngine,
    rules: &CompareRules,
) -> Result<(), HarnessError> {
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    if engine.collect_strict(lf.clone()).is_ok() {
        return Err(HarnessError::Engine(
            "strict mode unexpectedly accepted an unsupported query".to_string(),
        ));
    }
    let (actual, report) =
        engine.collect_with_report(lf).map_err(|error| HarnessError::Engine(error.to_string()))?;
    if report.engaged() {
        return Err(HarnessError::Engine(
            "non-strict mode unexpectedly engaged on an unsupported query".to_string(),
        ));
    }
    assert_frame_equal(&actual, &expected, rules)?;
    Ok(())
}

/// Whether `backend` reliably executes `f64` kernels.
///
/// Delegates to the runtime capability query: Apple Silicon has no `double`
/// in MSL, so both Metal stacks (and Metal-backed Vulkan) reject `f64`
/// shaders at compile time. `f64` queries must therefore expect graceful CPU
/// fallback there (pinned by the fallback half of the `f64` tests). Every
/// other backend runs `f64` differentially verified.
///
/// `Backend::Auto` resolves to a concrete backend at engine creation; the
/// differential suite never passes `Auto` here, so it maps to true.
#[must_use]
pub const fn supports_f64(backend: Backend) -> bool {
    match backend {
        Backend::Auto => true,
        Backend::Cpu => polars_cubecl::supports_f64(polars_cubecl::BackendKind::Cpu),
        Backend::Cuda => polars_cubecl::supports_f64(polars_cubecl::BackendKind::Cuda),
        Backend::Hip => polars_cubecl::supports_f64(polars_cubecl::BackendKind::Hip),
        Backend::Metal | Backend::Vulkan => {
            polars_cubecl::supports_f64(polars_cubecl::BackendKind::Wgpu)
        }
        Backend::MetalNative => {
            polars_cubecl::supports_f64(polars_cubecl::BackendKind::MetalNative)
        }
    }
}

/// Backends compiled into this test build, most capable first.
///
/// Honored filter: `POLARS_CUBECL_BACKENDS` (comma-separated subset) restricts
/// the matrix to named backends. Run one backend per process in CI
/// (`POLARS_CUBECL_BACKENDS=metal ...`) — several GPU stacks in a single
/// process can interfere (observed: concurrent Metal compilation across
/// `metal`/`metal-native`/`vulkan` flakes `f64` shader builds).
#[must_use]
pub fn all_backends() -> Vec<Backend> {
    let mut backends = Vec::with_capacity(5);
    #[cfg(feature = "cuda")]
    backends.push(Backend::Cuda);
    #[cfg(feature = "hip")]
    backends.push(Backend::Hip);
    #[cfg(all(feature = "metal-native", target_vendor = "apple"))]
    backends.push(Backend::MetalNative);
    #[cfg(feature = "metal")]
    backends.push(Backend::Metal);
    #[cfg(feature = "vulkan")]
    backends.push(Backend::Vulkan);
    #[cfg(feature = "cpu")]
    backends.push(Backend::Cpu);
    filter_backends(backends)
}

/// Restrict `backends` to `POLARS_CUBECL_BACKENDS` when set.
fn filter_backends(backends: Vec<Backend>) -> Vec<Backend> {
    let Ok(list) = std::env::var("POLARS_CUBECL_BACKENDS") else {
        return backends;
    };
    select_backends(backends, &list)
}

fn select_backends(backends: Vec<Backend>, list: &str) -> Vec<Backend> {
    let wanted: Vec<&str> = list.split(',').map(str::trim).collect();
    backends
        .into_iter()
        .filter(|backend| {
            wanted.iter().any(|name| match *name {
                "cpu" => *backend == Backend::Cpu,
                "cuda" => *backend == Backend::Cuda,
                "hip" => *backend == Backend::Hip,
                "metal" => *backend == Backend::Metal,
                "metal-native" => *backend == Backend::MetalNative,
                "vulkan" => *backend == Backend::Vulkan,
                _ => false,
            })
        })
        .collect()
}

/// Strict test configuration for one backend: never fall back on small inputs.
///
/// Strictness comes from calling `collect_strict*`; `raise_on_fail` stays
/// false so the same engine can also exercise the silent-fallback path.
#[must_use]
pub const fn engine_config_for(backend: Backend) -> EngineConfig {
    EngineConfig { backend, raise_on_fail: false, min_rows_for_gpu: 0 }
}

/// Backends that must be present, from `POLARS_CUBECL_REQUIRE_BACKENDS`
/// (comma-separated: `cpu,cuda,hip,metal,metal-native,vulkan`).
///
/// Tests skip unavailable backends silently by default; listing a backend
/// here turns a skip into a failure, so CI matrices can never pass
/// vacuously on machines that should have the hardware.
#[must_use]
pub fn required_backends() -> Vec<Backend> {
    let mut required = Vec::new();
    let Ok(list) = std::env::var("POLARS_CUBECL_REQUIRE_BACKENDS") else {
        return required;
    };
    for name in list.split(',') {
        match name.trim() {
            "cpu" => required.push(Backend::Cpu),
            "cuda" => required.push(Backend::Cuda),
            "hip" => required.push(Backend::Hip),
            "metal" => required.push(Backend::Metal),
            "metal-native" => required.push(Backend::MetalNative),
            "vulkan" => required.push(Backend::Vulkan),
            _ => {}
        }
    }
    required
}

/// Whether `backend` is required to be present (see [`required_backends`]).
#[must_use]
pub fn is_required(backend: Backend) -> bool {
    required_backends().contains(&backend)
}

/// Build an engine for `backend`: `Ok(None)` when the backend has no usable
/// device and is not required, `Err` when it is required but missing.
pub fn engine_for_backend(backend: Backend) -> Result<Option<CubeclEngine>, HarnessError> {
    match CubeclEngine::new(engine_config_for(backend)) {
        Ok(engine) => Ok(Some(engine)),
        Err(error) => {
            if is_required(backend) {
                Err(HarnessError::Engine(format!("required backend {backend:?} missing: {error}")))
            } else {
                Ok(None)
            }
        }
    }
}

/// Run one differential comparison on one backend.
///
/// Returns `None` when the backend is unavailable and not required (skip),
/// otherwise the report. Errors name the backend.
pub fn check_backend(
    lf: &LazyFrame,
    backend: Backend,
    rules: &CompareRules,
) -> Result<Option<ExecReport>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    assert_engine_matches_cpu(lf.clone(), &engine, rules)
        .map(Some)
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Assert non-strict CPU fallback without attempting strict mode.
///
/// Strict mode on an unsupported query *launches and fails* (e.g. an `f64`
/// shader the Metal compiler rejects), and that failure can poison the
/// backend's shared client for concurrently running tests. The
/// "unsupported implies strict errors" property is pinned by the non-`f64`
/// fallback tests on every backend; here only the production behavior is
/// pinned: non-strict never engages and stays CPU-correct.
pub fn check_quiet_fallback(
    lf: &LazyFrame,
    backend: Backend,
    rules: &CompareRules,
) -> Result<Option<()>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    let (actual, report) = engine
        .collect_with_report(lf.clone())
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))?;
    if report.engaged() {
        return Err(HarnessError::Engine(
            "non-strict mode unexpectedly engaged on an unsupported query".to_string(),
        ));
    }
    assert_frame_equal(&actual, &expected, rules)
        .map(|()| Some(()))
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Assert non-strict CPU fallback without attempting strict mode, sorting
/// both sides by `keys` (see [`check_quiet_fallback`] for why strict mode is
/// not attempted here, and [`check_fallback_backend_unsorted`] for why the
/// comparison is order-free).
pub fn check_quiet_fallback_unsorted(
    lf: &LazyFrame,
    backend: Backend,
    keys: &[String],
    rules: &CompareRules,
) -> Result<Option<()>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    let (actual, report) = engine
        .collect_with_report(lf.clone())
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))?;
    if report.engaged() {
        return Err(HarnessError::Engine(
            "non-strict mode unexpectedly engaged on an unsupported query".to_string(),
        ));
    }
    let actual = sort_by_keys(&actual, keys).map_err(HarnessError::Engine)?;
    let expected = sort_by_keys(&expected, keys).map_err(HarnessError::Oracle)?;
    assert_frame_equal(&actual, &expected, rules)
        .map(|()| Some(()))
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Run one unordered differential comparison on one backend, degrading to an
/// (order-free, strict-free) fallback check when the query uses `f64` on a
/// backend without `f64` support (see [`supports_f64`]).
///
/// Returns the engagement report, or `None` for fallback/skip. Errors name
/// the backend.
pub fn check_unsorted_or_fallback(
    lf: &LazyFrame,
    backend: Backend,
    keys: &[String],
    rules: &CompareRules,
    uses_f64_math: bool,
) -> Result<Option<ExecReport>, HarnessError> {
    if uses_f64_math && !supports_f64(backend) {
        check_quiet_fallback_unsorted(lf, backend, keys, rules)?;
        return Ok(None);
    }
    check_backend_unsorted(lf, backend, keys, rules)
}

/// Run one differential comparison on one backend, degrading to a
/// (strict-free) fallback check when the query uses `f64` arithmetic on a backend
/// without `f64` support.
///
/// Returns the engagement report, or `None` for fallback/skip. Errors name
/// the backend.
pub fn check_or_fallback(
    lf: &LazyFrame,
    backend: Backend,
    rules: &CompareRules,
    uses_f64_math: bool,
) -> Result<Option<ExecReport>, HarnessError> {
    if uses_f64_math && !supports_f64(backend) {
        check_quiet_fallback(lf, backend, rules)?;
        return Ok(None);
    }
    check_backend(lf, backend, rules)
}

/// Run one unordered differential comparison on one backend.
///
/// Returns `None` when the backend is unavailable and not required (skip),
/// otherwise the report. Errors name the backend.
pub fn check_backend_unsorted(
    lf: &LazyFrame,
    backend: Backend,
    keys: &[String],
    rules: &CompareRules,
) -> Result<Option<ExecReport>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    assert_engine_matches_cpu_unsorted(lf.clone(), &engine, keys, rules)
        .map(Some)
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Run one fallback check on one backend, sorting both sides by `keys`.
///
/// For `group_by` without `maintain_order`, where row order is out of
/// contract: the CPU oracle itself returns groups in nondeterministic order
/// across collects, so an unsorted fallback comparison would flake. Returns
/// `None` when the backend is unavailable and not required (skip). Errors
/// name the backend.
pub fn check_fallback_backend_unsorted(
    lf: &LazyFrame,
    backend: Backend,
    keys: &[String],
    rules: &CompareRules,
) -> Result<Option<()>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    let expected = lf.clone().collect().map_err(|error| HarnessError::Oracle(error.to_string()))?;
    if engine.collect_strict(lf.clone()).is_ok() {
        return Err(HarnessError::Engine(
            "strict mode unexpectedly accepted an unsupported query".to_string(),
        ));
    }
    let (actual, report) = engine
        .collect_with_report(lf.clone())
        .map_err(|error| HarnessError::Engine(error.to_string()))?;
    if report.engaged() {
        return Err(HarnessError::Engine(
            "non-strict mode unexpectedly engaged on an unsupported query".to_string(),
        ));
    }
    let actual = sort_by_keys(&actual, keys).map_err(HarnessError::Engine)?;
    let expected = sort_by_keys(&expected, keys).map_err(HarnessError::Oracle)?;
    assert_frame_equal(&actual, &expected, rules)
        .map(|()| Some(()))
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Run one fallback check on one backend.
///
/// Returns `None` when the backend is unavailable and not required (skip).
/// Errors name the backend.
pub fn check_fallback_backend(
    lf: &LazyFrame,
    backend: Backend,
    rules: &CompareRules,
) -> Result<Option<()>, HarnessError> {
    assert_matrix_nonempty()?;
    let Some(engine) = engine_for_backend(backend)? else {
        return Ok(None);
    };
    assert_fallback_for_unsupported(lf.clone(), &engine, rules)
        .map(|()| Some(()))
        .map_err(|error| HarnessError::Engine(format!("{backend:?}: {error}")))
}

/// Guard against vacuous runs: fail when the backend matrix is empty
/// (e.g. a misspelled `POLARS_CUBECL_BACKENDS` filter), so no test can pass
/// by iterating over zero backends.
fn assert_matrix_nonempty() -> Result<(), HarnessError> {
    if all_backends().is_empty() {
        return Err(HarnessError::Engine(
            "backend matrix is empty; check POLARS_CUBECL_BACKENDS".to_string(),
        ));
    }
    Ok(())
}

/// Nullable `i32` values with tricky bit patterns and edge magnitudes.
pub fn arb_opt_i32() -> impl Strategy<Value = Option<i32>> {
    prop_oneof![
        Just(None),
        any::<i32>().prop_map(Some),
        Just(Some(0)),
        Just(Some(i32::MIN)),
        Just(Some(i32::MAX)),
    ]
}

/// Nullable `i64` values with edge magnitudes.
pub fn arb_opt_i64() -> impl Strategy<Value = Option<i64>> {
    prop_oneof![
        Just(None),
        any::<i64>().prop_map(Some),
        Just(Some(0)),
        Just(Some(i64::MIN)),
        Just(Some(i64::MAX)),
    ]
}

/// Nullable `f64` values including `NaN`, infinities, and signed zero.
pub fn arb_opt_f64() -> impl Strategy<Value = Option<f64>> {
    prop_oneof![
        Just(None),
        any::<f64>().prop_map(Some),
        Just(Some(0.0)),
        Just(Some(-0.0)),
        Just(Some(f64::NAN)),
        Just(Some(f64::NEG_INFINITY)),
        Just(Some(f64::INFINITY)),
    ]
}

/// Nullable booleans.
pub fn arb_opt_bool() -> impl Strategy<Value = Option<bool>> {
    prop_oneof![Just(None), Just(Some(false)), Just(Some(true))]
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::NamedFrom;

    use super::*;

    #[test]
    fn backend_filter_selects_named_subset() {
        let all = vec![Backend::Cuda, Backend::Metal, Backend::Cpu];
        assert_eq!(select_backends(all.clone(), "metal"), vec![Backend::Metal]);
        assert_eq!(select_backends(all.clone(), "cpu,metal"), vec![Backend::Metal, Backend::Cpu]);
        assert!(
            select_backends(all.clone(), "vulkan").is_empty(),
            "vulkan was not in the input set"
        );
        assert_eq!(select_backends(all, ""), Vec::new());
    }

    fn frame_of(series: Vec<Series>) -> DataFrame {
        let columns: Vec<polars_core::prelude::Column> =
            series.into_iter().map(polars_core::prelude::Column::from).collect();
        match DataFrame::new_infer_height(columns) {
            Ok(frame) => frame,
            Err(_) => DataFrame::empty_with_height(0),
        }
    }

    #[test]
    fn equal_frames_pass() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[1_i32, 2, 3])]);
        let other = frame.clone();
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn nan_matches_nan() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[Some(f64::NAN), None])]);
        let other = frame.clone();
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn nan_does_not_match_number() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[f64::NAN])]);
        let other = frame_of(vec![Series::new("a".into(), &[1.0_f64])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::Value { .. })
        ));
    }

    #[test]
    fn null_position_mismatch_fails() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[Some(1_i32), None])]);
        let other = frame_of(vec![Series::new("a".into(), &[None, Some(1_i32)])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::Null { .. })
        ));
    }

    #[test]
    fn dtype_mismatch_fails() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[1_i32])]);
        let other = frame_of(vec![Series::new("a".into(), &[1_i64])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::DType { .. })
        ));
    }

    #[test]
    fn float_tolerance_accepts_close_values() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[1.0_f64])]);
        let other = frame_of(vec![Series::new("a".into(), &[1.0_f64 + 1e-13])]);
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn float_tolerance_rejects_far_values() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[1.0_f64])]);
        let other = frame_of(vec![Series::new("a".into(), &[1.1_f64])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::Value { .. })
        ));
    }

    #[test]
    fn row_order_matters() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[1_i32, 2])]);
        let other = frame_of(vec![Series::new("a".into(), &[2_i32, 1])]);
        assert!(assert_frame_equal(&frame, &other, &rules).is_err());
    }

    #[test]
    fn signed_zero_matches() {
        // `-0.0 == 0.0` in IEEE 754; the comparator must agree so that
        // surfaces that legitimately use `-0.0` (e.g. `floor`) don't false-positive.
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[0.0_f64])]);
        let other = frame_of(vec![Series::new("a".into(), &[-0.0_f64])]);
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn positive_and_negative_infinity_do_not_match() {
        // The comparator demands exact equality for infinities — a tolerance
        // would make `+inf` match any huge finite value.
        let rules = CompareRules::default();
        let pos_inf = frame_of(vec![Series::new("a".into(), &[f64::INFINITY])]);
        let neg_inf = frame_of(vec![Series::new("a".into(), &[f64::NEG_INFINITY])]);
        assert!(matches!(
            assert_frame_equal(&pos_inf, &neg_inf, &rules),
            Err(CompareError::Value { .. })
        ));
    }

    #[test]
    fn infinity_matches_itself() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[f64::INFINITY])]);
        let other = frame_of(vec![Series::new("a".into(), &[f64::INFINITY])]);
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn nan_does_not_match_infinity() {
        let rules = CompareRules::default();
        let nan_frame = frame_of(vec![Series::new("a".into(), &[f64::NAN])]);
        let inf_frame = frame_of(vec![Series::new("a".into(), &[f64::INFINITY])]);
        assert!(matches!(
            assert_frame_equal(&nan_frame, &inf_frame, &rules),
            Err(CompareError::Value { .. })
        ));
    }

    #[test]
    fn null_vs_value_is_a_null_mismatch() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[Some(1_i32)])]);
        let other = frame_of(vec![Series::new("a".into(), &[None as Option<i32>])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::Null { .. })
        ));
    }

    #[test]
    fn f32_nan_matches_nan() {
        let rules = CompareRules::default();
        let frame = frame_of(vec![Series::new("a".into(), &[f32::NAN])]);
        let other = frame_of(vec![Series::new("a".into(), &[f32::NAN])]);
        assert!(assert_frame_equal(&frame, &other, &rules).is_ok());
    }

    #[test]
    fn mixed_dtype_frame_reports_each_column() {
        // Two columns with different dtypes: the error must name the
        // first mismatched column so the user can find it.
        let rules = CompareRules::default();
        let frame =
            frame_of(vec![Series::new("a".into(), &[1_i32]), Series::new("b".into(), &[1_i32])]);
        let other =
            frame_of(vec![Series::new("a".into(), &[1_i32]), Series::new("b".into(), &[1_i64])]);
        assert!(matches!(
            assert_frame_equal(&frame, &other, &rules),
            Err(CompareError::DType { column, .. }) if column == "b"
        ));
    }
}
