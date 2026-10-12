//! Engine configuration.

use polars_cubecl_runtime::Backend;

/// Configuration for [`CubeclEngine`](crate::CubeclEngine).
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// Which backend to run on; [`Backend::Auto`] picks the most capable one.
    pub backend: Backend,
    /// When true, unsupported plans and execution failures are errors.
    /// When false, they fall back to the Polars CPU engine silently.
    pub raise_on_fail: bool,
    /// Plans with fewer rows stay on the CPU, where transfer would dominate.
    pub min_rows_for_gpu: usize,
}

impl EngineConfig {
    /// Production configuration: automatic backend, silent fallback.
    ///
    /// `min_rows_for_gpu` defaults to 5M because profiling shows GPU transfer
    /// + kernel launch overhead (~1-3 ms amortised after P0-P2 work) dominates
    /// below that threshold. Re-tune after each optimisation pass.
    #[must_use]
    pub const fn auto() -> Self {
        Self { backend: Backend::Auto, raise_on_fail: false, min_rows_for_gpu: 5_000_000 }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::auto()
    }
}
