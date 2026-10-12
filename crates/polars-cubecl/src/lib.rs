//! Third-party, cross-platform execution engine for Polars.
//!
//! [`CubeclEngine`] runs a well-defined core subset of a [`LazyFrame`] on any
//! accelerator CubeCL targets (CUDA, HIP/ROCm, Metal, Vulkan, CPU). Anything
//! outside the subset is left untouched so Polars executes it on its own
//! CPU/streaming engine: the engine can only make a query slower, never
//! wrong.

mod buffer_pool;
mod config;
mod engine;
mod error;
mod exec;
mod groupby;
mod report;

pub use config::EngineConfig;
pub use engine::CubeclEngine;
pub use error::EngineError;
pub use polars_cubecl_runtime::{Backend, BackendKind, RuntimeContext, RuntimeError, supports_f64};
pub use report::{ExecReport, Op};
