//! M2 differential tests: parquet scan + filter + group-by end to end.
//!
//! Fixtures are written with the CPU engine's parquet writer into
//! process-unique temp files, then queried through both engines.

#![cfg(feature = "parquet")]

use std::fs::File;

use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::Op;
use polars_cubecl_testing::{
    CompareRules, HarnessError, all_backends, check_backend, check_backend_unsorted,
    check_unsorted_or_fallback,
};
use polars_io::parquet::write::ParquetWriter;
use polars_lazy::{
    dsl::{col, len, lit},
    frame::LazyFrame,
};
use polars_utils::pl_path::PlRefPath;

fn rules() -> CompareRules {
    CompareRules::default()
}

fn fixture_frame() -> DataFrame {
    DataFrame::new_infer_height(vec![
        Series::new("g".into(), &[1_i32, 2, 1, 2, 1, 3]).into(),
        Series::new("v".into(), &[Some(10_i64), Some(20), None, Some(40), Some(50), None]).into(),
    ])
    .unwrap_or(DataFrame::empty_with_height(0))
}

/// A parquet fixture file, removed on drop.
struct Fixture {
    path: String,
}

impl Fixture {
    fn write(name: &str) -> Result<Self, HarnessError> {
        let mut frame = fixture_frame();
        let path = std::env::temp_dir()
            .join(format!("polars-cubecl-scan-{}-{name}.parquet", std::process::id()));
        let file = File::create(&path)
            .map_err(|error| HarnessError::Engine(format!("fixture create failed: {error}")))?;
        ParquetWriter::new(file)
            .finish(&mut frame)
            .map_err(|error| HarnessError::Engine(format!("fixture write failed: {error}")))?;
        let path = path
            .into_os_string()
            .into_string()
            .map_err(|_| HarnessError::Engine("fixture path is not valid unicode".to_string()))?;
        Ok(Self { path })
    }

    fn scan(&self) -> Result<LazyFrame, HarnessError> {
        LazyFrame::scan_parquet(PlRefPath::new(&self.path), Default::default())
            .map_err(|error| HarnessError::Engine(format!("scan build failed: {error}")))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn scan_filter_matches_oracle() -> Result<(), HarnessError> {
    let fixture = Fixture::write("filter")?;
    for backend in all_backends() {
        let lf = fixture.scan()?.filter(col("v").gt(lit(15)));
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}

#[test]
fn scan_group_by_matches_oracle() -> Result<(), HarnessError> {
    let fixture = Fixture::write("group")?;
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = fixture.scan()?.group_by([col("g")]).agg([
            col("v").sum().alias("s"),
            col("v").mean().alias("m"),
            len().alias("n"),
        ]);
        // `mean` needs `f64` arithmetic; backends without `double` fall back.
        let Some(_) = check_unsorted_or_fallback(&lf, backend, &keys, &rules(), true)? else {
            continue;
        };
    }
    Ok(())
}

#[test]
fn scan_filter_group_by_matches_oracle() -> Result<(), HarnessError> {
    let fixture = Fixture::write("combo")?;
    let keys = vec!["g".to_string()];
    for backend in all_backends() {
        let lf = fixture
            .scan()?
            .filter(col("v").gt(lit(15)))
            .group_by([col("g")])
            .agg([col("v").sum().alias("s"), len().alias("n")]);
        let Some(report) = check_backend_unsorted(&lf, backend, &keys, &rules())? else {
            continue;
        };
        assert!(report.ops_ran().contains(&Op::GroupBy));
    }
    Ok(())
}

#[test]
fn scan_projection_matches_oracle() -> Result<(), HarnessError> {
    let fixture = Fixture::write("proj")?;
    for backend in all_backends() {
        let lf = fixture.scan()?.select([col("v")]).filter(col("v").gt(lit(15)));
        if let Some(report) = check_backend(&lf, backend, &rules())? {
            assert!(report.ops_ran().contains(&Op::Filter));
        }
    }
    Ok(())
}
