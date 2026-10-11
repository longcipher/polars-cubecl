# polars-cubecl

A third-party, cross-platform execution engine for
[Polars](https://github.com/pola-rs/polars), built on
[CubeCL](https://github.com/tracel-ai/cubecl).

One Rust codebase runs a well-defined core subset of a `LazyFrame` on
NVIDIA CUDA, AMD ROCm/HIP, Apple Metal, Vulkan/iGPU, and CPU — the hardware
that CUDA-only engines cannot reach. Anything outside the subset is left
untouched, so Polars executes it on its own CPU engine: results are never
wrong, only potentially slower.

See [`docs/design.md`](docs/design.md) for the architecture,
[`docs/differential-testing.md`](docs/differential-testing.md) for the
correctness strategy, and [`docs/roadmap.md`](docs/roadmap.md) for milestones.

## Compatibility

| Dependency | Pinned version | Reason |
| --- | --- | --- |
| `polars` family | `=0.55.2` | Internal IR; must match exactly |
| `cubecl` family | `=0.11.0` | Pre-1.0 API churn |
| Rust toolchain | nightly (see `rust-toolchain.toml`) | workspace lints |

## Quick start

```rust
use polars_cubecl::{CubeclEngine, EngineConfig};

let engine = CubeclEngine::new(EngineConfig::auto())?;
// Silent CPU fallback for unsupported queries:
let out = engine.collect(lazyframe)?;
// Strict mode for tests: error instead of falling back, with a report:
let (out, report) = engine.collect_strict_with_report(lazyframe)?;
assert!(report.engaged());
```

## Features

Backend selection (exactly one kernel source, compiled per backend):

| Feature | Backend | Default |
| --- | --- | --- |
| `cpu` | CPU (LLVM SIMD) | on |
| `cuda` | NVIDIA CUDA | off |
| `hip` / `rocm` | AMD ROCm | off |
| `metal` | Apple Metal via wgpu/MSL | off |
| `metal-native` | Apple Metal, native runtime | off |
| `vulkan` | Vulkan via wgpu/SPIR-V | off |
| `parquet` | parquet scan support | off |

```bash
cargo test                                   # cpu backend
cargo test --features metal                  # + Apple Metal (this machine)
cargo test --all-features                    # everything incl. parquet
POLARS_CUBECL_REQUIRE_BACKENDS=metal,cpu cargo test --features metal
```

The `POLARS_CUBECL_REQUIRE_BACKENDS` variable turns backend skips into
failures, so CI matrices can never pass vacuously.

## Supported subset (M0–M3)

- Sources: in-memory frames, parquet files (`parquet` feature).
- Ops: `filter` (comparisons, `and`/`or`/`not`, bare bool masks),
  column-only `select`, `slice`, `group_by` (`sum/min/max/mean/count/
  first/last/len`), hybrid partial substitution (supported heads run on
  device under unsupported tails).
- Dtypes: `Int32/Int64/UInt32/UInt64/Float32/Float64/Boolean`, with exact
  Polars null semantics (null comparisons never pass, aggs ignore nulls,
  null keys form their own group, total-order float comparisons with
  `NaN` above `+inf`). Apple GPUs have no MSL `double`: `f64` queries fall
  back to CPU there (pinned by tests), everything else is verified on-device.

## Building offline

The CPU backend JITs through a prebuilt LLVM bundle that CubeCL downloads
at build time. With no direct network access, pre-fetch it once (the build
script then verifies checksums and extracts offline):

```bash
export TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1
# place tracel-llvm-23.1.0-3-macos-AArch64.tar.xz + .checksums.json
# (from tracel-ai/tracel-llvm releases) into ~/.cache/tracel/
```

`vendor/branches` carries a two-line toolchain-compat fix for a transitive
dependency (`branches 0.4.6` uses a `core` intrinsic removed from recent
nightlies) that only the `metal`/`vulkan` backend graph reaches. It applies
to workspace builds via `[patch.crates-io]`; see `docs/design.md`.

## Commands

```bash
just format
just lint
just test
just mutation
cargo bench -p polars-cubecl [--features metal]   # baseline, reported not asserted
cargo run -p pcbe -- --backend metal --rows 1000000
```

## License

Apache-2.0
