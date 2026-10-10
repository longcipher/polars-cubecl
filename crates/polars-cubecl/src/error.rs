//! Engine errors.

use polars_cubecl_runtime::RuntimeError;
use thiserror::Error;

/// Error returned when the engine cannot be created or a strict collection
/// fails.
#[derive(Debug, Error)]
pub enum EngineError {
    /// No usable CubeCL backend.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    /// The plan is outside the supported subset (strict mode only).
    #[error("plan outside the engine subset: {0}")]
    Unsupported(String),
    /// The plan lowered but device execution failed (strict mode only).
    #[error("device execution failed: {0}")]
    Execution(String),
}
