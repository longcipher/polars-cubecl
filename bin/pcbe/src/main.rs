//! `pcbe`: run a synthetic filter/group-by query through `polars-cubecl`.
//!
//! Manual benchmarking and smoke-testing entry point; the automated
//! differential suite lives in the crate tests.

use std::{io::Write, time::Instant};

use clap::Parser;
use eyre::Result;
use polars_core::{
    frame::DataFrame,
    prelude::{NamedFrom, Series},
};
use polars_cubecl::{CubeclEngine, EngineConfig};
use polars_lazy::{
    dsl::{col, lit},
    prelude::IntoLazy,
};

/// Run a synthetic query on the selected backend and report timings.
#[derive(Debug, Parser)]
struct Args {
    /// Backend: auto, cpu, cuda, hip, metal, metal-native, vulkan.
    #[arg(long, default_value = "auto")]
    backend: String,
    /// Row count of the synthetic frame.
    #[arg(long, default_value_t = 1_000_000)]
    rows: usize,
    /// Filter threshold on column `a`.
    #[arg(long, default_value_t = 0)]
    threshold: i64,
    /// Print extra diagnostics (also enabled with `POLARS_VERBOSE=1`).
    #[arg(long, default_value_t = false)]
    verbose: bool,
    /// Read a parquet file instead of generating synthetic data.
    #[arg(long)]
    parquet: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = match args.backend.as_str() {
        "auto" => polars_cubecl::Backend::Auto,
        "cpu" => polars_cubecl::Backend::Cpu,
        "cuda" => polars_cubecl::Backend::Cuda,
        "hip" => polars_cubecl::Backend::Hip,
        "metal" => polars_cubecl::Backend::Metal,
        "metal-native" => polars_cubecl::Backend::MetalNative,
        "vulkan" => polars_cubecl::Backend::Vulkan,
        other => eyre::bail!("unknown backend: {other}"),
    };
    let engine =
        CubeclEngine::new(EngineConfig { backend, raise_on_fail: true, min_rows_for_gpu: 0 })?;

    let frame = match &args.parquet {
        Some(path) => read_parquet(path)?,
        None => synthetic(args.rows)?,
    };
    let query = frame
        .lazy()
        .filter(col("a").gt(lit(args.threshold)))
        .group_by([col("g")])
        .agg([col("b").sum().alias("total")]);

    let started = Instant::now();
    let (result, report) = engine.collect_strict_with_report(query)?;
    let elapsed = started.elapsed();
    let rows = result.height();
    let rows_per_second = rows as f64 / elapsed.as_secs_f64();

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "backend: {:?}", engine.backend_kind())?;
    writeln!(stdout, "engaged: {}", report.engaged())?;
    writeln!(stdout, "ops: {:?}", report.ops_ran())?;
    writeln!(stdout, "rows: {rows}")?;
    writeln!(stdout, "elapsed_ms: {}", elapsed.as_millis())?;
    writeln!(stdout, "rows_per_second: {rows_per_second:.0}")?;
    if args.verbose || polars_config::config().verbose() {
        writeln!(stdout, "input_rows: {}", args.rows)?;
        writeln!(stdout, "threshold: {}", args.threshold)?;
    }
    Ok(())
}

fn synthetic(rows: usize) -> Result<DataFrame> {
    let keys: Vec<i32> = (0..rows).map(|index| (index % 1024) as i32).collect();
    let group: Vec<i32> = (0..rows).map(|index| (index % 64) as i32).collect();
    let values: Vec<i64> = (0..rows).map(|index| index as i64).collect();
    DataFrame::new_infer_height(vec![
        Series::new("a".into(), keys).into(),
        Series::new("g".into(), group).into(),
        Series::new("b".into(), values).into(),
    ])
    .map_err(|error| eyre::eyre!("{error}"))
}

/// Read a parquet file for benchmarking the scan path.
///
/// Only available with the `parquet` feature; without it, using `--parquet`
/// is a usage error.
fn read_parquet(path: &str) -> Result<DataFrame> {
    #[cfg(feature = "parquet")]
    {
        use polars_lazy::frame::LazyFrame;
        use polars_utils::pl_path::PlRefPath;
        LazyFrame::scan_parquet(PlRefPath::new(path), Default::default())
            .and_then(|lf| lf.collect())
            .map_err(|error| eyre::eyre!("{error}"))
    }
    #[cfg(not(feature = "parquet"))]
    {
        let _ = path;
        eyre::bail!("--parquet needs the `parquet` feature")
    }
}
