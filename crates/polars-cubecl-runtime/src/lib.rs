//! Backend selection and CubeCL client management.
//!
//! This is the only crate that names CubeCL runtimes. Everything else works
//! against the [`RuntimeContext`] it hands out, so a CubeCL version bump is
//! contained here.
//!
//! [`Backend::Auto`] discovery order is `cuda`, `hip`, `metal-native`,
//! `metal`/`vulkan` (wgpu), then `cpu`, matching the design document. The
//! `cpu` backend is always available when linked, so the engine always has a
//! target, including on machines without any GPU.

use cubecl::{Device, device, prelude::Client};
use thiserror::Error;

/// Error returned when no usable CubeCL backend can be provided.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RuntimeError {
    /// The explicitly requested backend is not linked into this build.
    #[error("backend {0:?} is not linked into this build (enable its cargo feature)")]
    NotLinked(BackendKind),
    /// The backend is linked but no device is usable on this machine.
    #[error("backend {0:?} is linked but no usable device was found")]
    Unavailable(BackendKind),
    /// No linked backend has a usable device.
    #[error("no usable CubeCL backend found (link at least one backend feature)")]
    NoBackend,
}

/// A backend the caller can request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Pick the most capable usable backend (`cuda` first, `cpu` last).
    #[default]
    Auto,
    /// Force the CUDA backend (NVIDIA).
    Cuda,
    /// Force the HIP backend (AMD ROCm).
    Hip,
    /// Force the native Metal backend (Apple Silicon).
    MetalNative,
    /// Force the wgpu Metal-shading-language path (Apple).
    Metal,
    /// Force the wgpu SPIR-V path (Vulkan devices and iGPUs).
    Vulkan,
    /// Force the CPU backend (always usable when linked).
    Cpu,
}

/// Whether `kind` reliably executes `f64` kernels.
///
/// Apple Silicon has no `double` in MSL, so both Metal stacks reject `f64`
/// shaders at compile time, as does Metal-backed Vulkan (MoltenVK) on Apple.
/// Queries that need `f64` arithmetic must therefore fall back to the CPU
/// engine there. Bitwise movement (gather/slice) over `f64` data still runs
/// on device via `u64` kernels — only arithmetic needs true `double`.
///
/// # Examples
///
/// ```
/// use polars_cubecl_runtime::{BackendKind, supports_f64};
///
/// assert!(supports_f64(BackendKind::Cpu));
/// assert!(supports_f64(BackendKind::Cuda));
/// assert!(!supports_f64(BackendKind::MetalNative));
/// ```
#[must_use]
pub const fn supports_f64(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu | BackendKind::Cuda | BackendKind::Hip => true,
        // Both Metal stacks reject MSL `double`, as does Metal-backed Vulkan.
        BackendKind::MetalNative => false,
        #[cfg(target_vendor = "apple")]
        BackendKind::Wgpu => false,
        #[cfg(not(target_vendor = "apple"))]
        BackendKind::Wgpu => true,
    }
}

/// A backend variant, used in reports and errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// NVIDIA CUDA backend.
    Cuda,
    /// AMD ROCm/HIP backend.
    Hip,
    /// Native Metal backend.
    MetalNative,
    /// wgpu backend (Metal MSL, Vulkan SPIR-V, or WebGPU).
    Wgpu,
    /// CPU backend.
    Cpu,
}

/// A selected device together with the client used to launch kernels on it.
#[derive(Clone)]
pub struct RuntimeContext {
    device: Device,
    client: Client,
    kind: BackendKind,
}

impl std::fmt::Debug for RuntimeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeContext")
            .field("device", &self.device)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl RuntimeContext {
    /// Select a backend and create its client.
    pub fn select(backend: Backend) -> Result<Self, RuntimeError> {
        match backend {
            Backend::Auto => Self::auto(),
            Backend::Cuda => Self::named(BackendKind::Cuda),
            Backend::Hip => Self::named(BackendKind::Hip),
            Backend::MetalNative => Self::named(BackendKind::MetalNative),
            Backend::Metal | Backend::Vulkan => Self::named(BackendKind::Wgpu),
            Backend::Cpu => Self::named(BackendKind::Cpu),
        }
    }

    /// The backend this context runs on.
    #[must_use]
    pub const fn kind(&self) -> BackendKind {
        self.kind
    }

    /// The CubeCL device this context runs on.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// The CubeCL client used to allocate buffers and launch kernels.
    #[must_use]
    pub const fn client(&self) -> &Client {
        &self.client
    }

    /// Backends linked into this build, most capable first.
    #[must_use]
    pub fn linked_backends() -> Vec<BackendKind> {
        let mut kinds = Vec::with_capacity(5);
        #[cfg(feature = "cuda")]
        kinds.push(BackendKind::Cuda);
        #[cfg(feature = "hip")]
        kinds.push(BackendKind::Hip);
        #[cfg(all(feature = "metal-native", target_vendor = "apple"))]
        kinds.push(BackendKind::MetalNative);
        #[cfg(any(feature = "metal", feature = "vulkan"))]
        kinds.push(BackendKind::Wgpu);
        #[cfg(feature = "cpu")]
        kinds.push(BackendKind::Cpu);
        kinds
    }

    fn auto() -> Result<Self, RuntimeError> {
        for kind in Self::linked_backends() {
            if let Ok(ctx) = Self::named(kind) {
                return Ok(ctx);
            }
        }
        Err(RuntimeError::NoBackend)
    }

    fn named(kind: BackendKind) -> Result<Self, RuntimeError> {
        if !Self::linked_backends().contains(&kind) {
            return Err(RuntimeError::NotLinked(kind));
        }
        let device = Self::default_device(kind);
        let client = Self::client_for(kind, &device)?;
        Ok(Self { device, client, kind })
    }

    fn default_device(kind: BackendKind) -> Device {
        match kind {
            BackendKind::Cuda => Device::Cuda(device::CudaDevice::default()),
            BackendKind::Hip => Device::Hip(device::AmdDevice::default()),
            BackendKind::MetalNative => Device::Metal(device::MetalDevice::default()),
            BackendKind::Wgpu => Device::Wgpu(device::WgpuDevice::default()),
            BackendKind::Cpu => Device::Cpu(device::CpuDevice),
        }
    }

    fn client_for(kind: BackendKind, device: &Device) -> Result<Client, RuntimeError> {
        // Client creation probes the driver; a missing or broken driver
        // surfaces as a panic inside the runtime, which we convert into a
        // typed error so backend discovery can fall through to the next
        // candidate instead of crashing the host query.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device.client()))
            .map_err(|_| RuntimeError::Unavailable(kind))
    }
}

#[cfg(not(any(
    feature = "cpu",
    feature = "cuda",
    feature = "hip",
    feature = "metal",
    feature = "metal-native",
    feature = "vulkan"
)))]
compile_error!(
    "polars-cubecl-runtime requires at least one backend feature (cpu, cuda, hip, metal, \
     metal-native, vulkan)"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_selectable_when_linked() -> Result<(), RuntimeError> {
        #[cfg(feature = "cpu")]
        {
            let ctx = RuntimeContext::select(Backend::Cpu)?;
            assert_eq!(ctx.kind(), BackendKind::Cpu);
            assert_eq!(ctx.device(), &Device::Cpu(device::CpuDevice));
        }
        Ok(())
    }

    #[test]
    fn auto_selection_finds_a_backend() -> Result<(), RuntimeError> {
        let ctx = RuntimeContext::select(Backend::Auto)?;
        assert!(RuntimeContext::linked_backends().contains(&ctx.kind()));
        Ok(())
    }

    #[test]
    fn linked_backends_never_empty() {
        assert!(!RuntimeContext::linked_backends().is_empty(), "no backends linked");
    }

    #[test]
    fn explicit_cpu_reports_cpu_kind() -> Result<(), RuntimeError> {
        #[cfg(feature = "cpu")]
        {
            let ctx = RuntimeContext::select(Backend::Cpu)?;
            assert_eq!(format!("{:?}", ctx.kind()), "Cpu");
            assert!(format!("{:?}", ctx).contains("Cpu"));
        }
        Ok(())
    }

    #[test]
    fn f64_support_matches_backend_capability() {
        assert!(supports_f64(BackendKind::Cpu));
        assert!(supports_f64(BackendKind::Cuda));
        assert!(supports_f64(BackendKind::Hip));
        assert!(!supports_f64(BackendKind::MetalNative));
        #[cfg(target_vendor = "apple")]
        assert!(!supports_f64(BackendKind::Wgpu));
        #[cfg(not(target_vendor = "apple"))]
        assert!(supports_f64(BackendKind::Wgpu));
    }

    /// `Backend::Auto` never returns a `NoBackend` when at least one backend
    /// is linked — `cpu` is always available when its feature is on.
    #[test]
    #[expect(clippy::expect_used, reason = "test asserts cpu-linked invariant")]
    fn auto_never_fails_when_cpu_linked() {
        #[cfg(feature = "cpu")]
        {
            let ctx = RuntimeContext::select(Backend::Auto)
                .expect("Auto backend selection should always succeed when cpu is linked");
            // The kind must be one of the linked backends.
            assert!(RuntimeContext::linked_backends().contains(&ctx.kind()));
        }
    }

    /// Selecting `Backend::Cpu` when the `cpu` feature is disabled surfaces a
    /// typed `NotLinked` error instead of panicking.
    #[test]
    fn cpu_reports_not_linked_when_disabled() {
        #[cfg(not(feature = "cpu"))]
        {
            let err =
                RuntimeContext::select(Backend::Cpu).expect_err("cpu backend should be unlinked");
            assert_eq!(err, RuntimeError::NotLinked(BackendKind::Cpu));
        }
    }

    /// `linked_backends` preserves feature declaration order: GPU-first.
    #[test]
    fn linked_backends_order_is_gpu_first() {
        let backends = RuntimeContext::linked_backends();
        // Wherever cpu appears, it must be last (lowest priority).
        if let Some(pos) = backends.iter().position(|b| *b == BackendKind::Cpu) {
            assert_eq!(pos, backends.len() - 1, "Cpu should be last in backend order");
        }
    }

    /// `Debug` output includes the backend kind for diagnostics.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts cpu-linked invariant")]
    fn debug_includes_backend_kind() {
        #[cfg(feature = "cpu")]
        {
            let ctx = RuntimeContext::select(Backend::Cpu).unwrap();
            let debug = format!("{ctx:?}");
            assert!(debug.contains("RuntimeContext"));
            assert!(debug.contains("Cpu"));
        }
    }
}
