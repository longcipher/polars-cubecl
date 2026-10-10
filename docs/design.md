# polars-cubecl — Design

> A third-party, cross-platform execution engine for [Polars](https://github.com/pola-rs/polars),
> built on [CubeCL](https://github.com/tracel-ai/cubecl).
> Status: design proposal. Reference sources: `/Volumes/akext/tmp/polars` (Polars 0.55.1),
> `/Volumes/akext/tmp/cudf` (cudf-polars, the NVIDIA reference engine).

---

## 1. Goal and non-goals

### 1.1 Goal

Provide a drop-in execution backend for Polars that runs a **well-defined core subset** of a
`LazyFrame` on any GPU/accelerator that CubeCL can target:

| Target hardware | CubeCL backend | cudf covers it? |
| --- | --- | --- |
| NVIDIA GPU | `cuda` | yes |
| **AMD GPU (ROCm/HIP)** | `hip` | **no** |
| **Apple Silicon (Metal)** | `metal` / `wgpu-msl` | **no** |
| **Intel / AMD iGPU, ARM Mali (Vulkan)** | `wgpu-spirv` | **no** |
| **ARM64 / x86 CPU (LLVM SIMD)** | `cpu` | **no** |

The differentiator is explicitly **not** peak NVIDIA performance — it is **portability of a
single Rust codebase** across the hardware that cuDF cannot reach. It is shipped as an
independent crate; the Polars main repo is never forked.

### 1.2 Non-goals (explicit)

- Full Polars API parity. This is a *subset* engine with **mandatory CPU fallback**.
- Out-of-core / streaming execution, joins, windows, string/datetime kernels in the first phase.
- Beating cuDF on NVIDIA. If NVIDIA is the only target, use cuDF.
- A stable public ABI. The engine is a plugin and follows Polars' internal IR.

### 1.3 The one hard safety property

> **The engine must never return a wrong answer.**
> If anything is not provably supported, the engine **does not touch the plan** and Polars
> executes it on the CPU/streaming engine. Wrong results are impossible-by-construction;
> the only visible effect of an unsupported query is that it stays on the CPU.

This property is what makes a subset engine acceptable. Everything else (speed, coverage)
is secondary, and it is enforced by the differential test suite (see
[differential-testing.md](./differential-testing.md)).

---

## 2. Feasibility

### 2.1 Can a third-party **pure-Rust** crate plug into Polars? — Yes

Polars does not expose a public "engine trait", but it exposes the two primitives an external
engine needs, both as **public Rust API** in `polars-lazy` 0.55.1:

| Capability | API | Source |
| --- | --- | --- |
| Get the optimized plan | `LazyFrame::optimize(&mut Arena<IR>, &mut Arena<AExpr>) -> PolarsResult<Node>` | `crates/polars-lazy/src/frame/mod.rs:519` |
| Get the optimized plan as one value | `LazyFrame::to_alp_optimized() -> PolarsResult<IRPlan>` | `crates/polars-lazy/src/frame/mod.rs:527` |
| **Inject a post-optimization hook** | `LazyFrame::_collect_post_opt<P>(self, P) -> PolarsResult<DataFrame>` where `P: FnOnce(Node, &mut Arena<IR>, &mut Arena<AExpr>, Option<Duration>) -> PolarsResult<()>` | `crates/polars-lazy/src/frame/mod.rs:607` (four arguments in published 0.55.1; the last is the optimization duration) |
| **Write a computed result back into the plan** | `IR::DataFrameScan { df: Arc<DataFrame>, schema, output_schema }` (not Python-gated) | `crates/polars-plan/src/plans/ir/mod.rs:86` |
| Mutate a plan node | `Arena::replace(Node, IR) -> IR` | `crates/polars-utils/src/arena.rs:132` |
| Rebuild a `Series`/`DataFrame` from Arrow | `Series::from_arrow(name, ArrayRef)`, `Column::from(Series)`, `DataFrame::new_infer_height(Vec<Column>)` | `crates/polars-core/src/series/mod.rs:299`, `.../frame/column/series.rs:22`, `.../frame/dataframe.rs:128` |

This is the **same hook cudf-polars uses**. In Python, cudf-polars receives a `NodeTraverser`,
walks the IR, and calls `NodeTraverser::set_udf(...)` to splice in a `PythonScan`
(`crates/polars-python/src/lazyframe/visit.rs:158`). In pure Rust the equivalent splice target is
`IR::DataFrameScan`, which is available unconditionally.

**Mechanism (pure Rust):**

```rust
lf._collect_post_opt(move |root, lp_arena, expr_arena| {
    // 1. Try to translate the polars IR into our own small "CubePlan".
    let Ok(plan) = lower(root, lp_arena, expr_arena) else {
        return Ok(());            // unsupported → leave plan untouched → Polars runs on CPU
    };
    // 2. Execute on the selected CubeCL backend.
    let Ok(df) = ctx.exec(&plan) else {
        return Ok(());            // runtime failure → fallback (unless strict mode)
    };
    // 3. Splice the result back; Polars now just returns it.
    let schema = df.schema().clone();
    lp_arena.replace(root, IR::DataFrameScan {
        df: Arc::new(df),
        schema,
        output_schema: None,
    });
    Ok(())
})
```

`_collect_post_opt` is called **after** Polars' optimizer and **before** physical-plan creation
(`prepare_collect_post_opt`, `crates/polars-lazy/src/frame/mod.rs:577`), so we see the final,
optimized IR (predicate pushdown, projection pushdown, CSE already applied).

### 2.2 Is CubeCL the right primitive? — Yes, it is the only realistic Rust choice

`cubecl` **0.11.0** (published 2026-10-06, edition 2024, `MIT OR Apache-2.0`) compiles one
`#[cube]` Rust kernel to PTX (CUDA), AMDGPU/HIP (ROCm), MSL (Metal), SPIR-V (Vulkan) and WGSL
(WebGPU), plus an LLVM SIMD CPU backend. Relevant umbrella features:

| Feature | Backend crate |
| --- | --- |
| `cuda` | `cubecl-cuda` |
| `hip` / `rocm` | `cubecl-hip` |
| `metal` (→ `wgpu-msl`) | `cubecl-wgpu` (MSL) |
| `vulkan` (→ `wgpu-spirv`), `webgpu` | `cubecl-wgpu` |
| `metal-native` | `cubecl-metal` |
| `cpu` | `cubecl-cpu` |

Kernel model (from the CubeCL book): a kernel is a `#[cube(launch)]` Rust function over
`&[Vector<T, N>]` slices addressed by `ABSOLUTE_POS`; launches take a `ComputeClient`, a
`CubeCount`, a `CubeDim`, and a vectorization width; parallelism is described on four orthogonal
axes (Vector / Plane / CubeDim / CubeCount) and **read from `PLANE_DIM`** rather than hard-coded,
which is exactly why the same kernel is portable from a 32-lane NVIDIA warp to a 64-lane AMD
wavefront to a plane-of-1 CPU.

### 2.3 Where the real cost is

The GPU kernels are **not** the hard part. The cost of an engine (as `cudf-polars`, tens of
thousands of LOC, demonstrates) is:

1. **IR traversal + support detection** — deciding, soundly, if a subtree is supported.
2. **Expression lowering** — turning Polars `AExpr` trees into kernels / kernel configs.
3. **Null semantics** — matching Polars exactly (comparison-null → false, agg-ignores-null,
   `count` vs `len`, null group keys). This is the #1 source of differential failures.
4. **Data movement** — Arrow ↔ device buffers, validity bitmaps, dtypes, chunking.
5. **Ordering** — `group_by` output order is unspecified without `maintain_order`.

The design below therefore invests in a **conservative subset + fallback**, not in breadth.

### 2.4 Verdict

| Question | Answer |
| --- | --- |
| Can it plug in without forking Polars? | **Yes** — `_collect_post_opt` + `IR::DataFrameScan`, both public. |
| Does CubeCL cover the target hardware? | **Yes** — hip/metal/vulkan/cpu, one kernel. |
| Is a full engine feasible now? | **No** — multi-year scope, parity is out. |
| Is a **core-subset prototype** feasible? | **Yes** — filter + group-by-agg + scan, with fallback. |
| Main technical risk? | Polars IR is **unstable** (the Python traverser pins `VERSION = (15, 2)`). Mitigation: pin the exact `polars` version (§9). |

---

## 3. Architecture

```
                            user code
                                │
                lf.collect()  / engines.collect(lf)          ← §4 public facade
                                │
        ┌───────────────────────▼────────────────────────┐
        │ polars-cubecl            (facade + Engine)      │
        │  · CubeclEngine::collect(LazyFrame)             │
        │  · _collect_post_opt hook + DataFrameScan splice │
        └───────────┬─────────────────────┬───────────────┘
                    │                     │
        ┌───────────▼──────────┐  ┌───────▼────────────────┐
        │ polars-cubecl-lower  │  │ polars-cubecl-runtime  │
        │  IR walk + support   │  │  backend select        │
        │  detection + lower   │  │  ComputeClient, device │
        │  to CubePlan IR      │  │  transfer, warmup/     │
        │  + fallback decision │  │  autotune cache        │
        └───────────┬──────────┘  └───────┬────────────────┘
                    │                     │
        ┌───────────▼─────────────────────▼────────────────┐
        │ polars-cubecl-kernels  (#[cube] kernels)          │
        │  predicate · compact/scan-gather · hash-groupby · │
        │  segmented reductions (sum/min/max/count/mean)    │
        └───────────┬──────────────────────────────────────┘
                    │
        ┌───────────▼──────────────────────────────────────┐
        │ polars-cubecl-buffer                              │
        │  Arrow ⇄ device buffers: values + validity,       │
        │  dtype mapping, chunk flattening, download/pack   │
        └───────────────────────────────────────────────────┘
```

### 3.1 Layering rationale

- **`-lower`** is the only crate that knows Polars internals. Everything else is Polars-agnostic,
  so a Polars IR bump is contained in one place.
- **`-runtime`** is the only crate that knows CubeCL backends/features. A CubeCL bump is contained.
- **`-kernels`** and **`-buffer`** are pure compute/transfer; they take plain typed buffers.
- This keeps the blast radius of the two unstable dependencies (Polars IR, CubeCL) to one crate each.

### 3.2 Execution model (M0: "host-driven, one kernel per op")

We deliberately **do not** build an expression→GPU-codegen compiler at first. Instead:

1. Lower the Polars IR into a **fixed, small `CubePlan`** enum (a linear DAG of ops).
2. Each supported op maps to a **pre-written `#[cube]` kernel** with a small config.
3. Data stays in device buffers between ops when possible; M0 may upload/download per op.

This trades generality for velocity and testability: with ~5 kernels we cover filter + group-by.

---

## 4. Public API (facade)

```rust
// polars-cubecl/src/lib.rs
pub struct CubeclEngine { /* config + runtime context */ }

impl CubeclEngine {
    /// Pick the best available backend, or force one.
    pub fn new(config: EngineConfig) -> Result<Self, EngineError>;

    /// Execute `lf`, falling back to the Polars CPU engine when unsupported.
    pub fn collect(&self, lf: LazyFrame) -> PolarsResult<DataFrame>;

    /// Same as `collect`, but error instead of falling back (used by tests).
    pub fn collect_strict(&self, lf: LazyFrame) -> PolarsResult<DataFrame>;
}

pub struct EngineConfig {
    pub backend: Backend,          // Auto | Cuda | Hip | Metal | Vulkan | Cpu
    pub device_index: Option<usize>,
    pub raise_on_fail: bool,       // strict: error on unsupported, never fall back
    pub min_rows_for_gpu: usize,   // below this, stay on CPU (transfer dominates)
}
```

Usage:

```rust
let engine = CubeclEngine::new(EngineConfig::auto())?;
let out = engine.collect(lf)?;                 // production: silent fallback
// tests: engine.collect_strict(lf) must either run on GPU or error — never silently fall back
```

`Backend::Auto` discovery order: `cuda` → `hip` → `metal` (macOS) → `vulkan` → `cpu`.
The `cpu` backend is always available, so the engine always has a target.

---

## 5. Core operator subset (M0 / M1)

### 5.1 Supported `IR` nodes

| Polars IR node | Support | Notes |
| --- | --- | --- |
| `DataFrameScan` | ✅ M0 | in-memory input; upload to device |
| `Filter` | ✅ M0 | predicate must lower (§5.2) |
| `SimpleProjection` / `Select` (column-only) | ✅ M0 | pure column gather/reorder, no expr eval |
| `Slice` | ✅ M0 | offset + len |
| `Sink` (Memory) | ✅ M0 | unwrap root |
| `GroupBy` | ✅ M1 | keys = plain columns; aggs ⊆ (§5.3) |
| `Scan` (parquet) | ✅ M2 | read on CPU → upload; pushdown projection/predicate |
| `Cache`, `Sort`, `Join`, `Window`, `Distinct`, `HConcat`, `Union`, `MapFunction`, `PythonScan`, … | ❌ | → fallback |

### 5.2 Supported predicate expressions (`AExpr`)

- `Column(name)`, `Literal`
- `BinaryExpr` with `Eq, NotEq, Lt, LtEq, Gt, GtEq` (numeric, bool; floats under the
  total order above)
- `BinaryExpr` with `And, Or`; `Not` (pushed to leaves via De Morgan at lowering time)
- `Int32`/`UInt32` columns against `f64` literals via an exact device-side widening cast
  (the CPU engine widens the integer side to `f64`)

Everything else (strings, arithmetic-in-predicate at M0, `is_null`, `in`, …) → **unsupported → fallback**.

### 5.3 Supported aggregations

On numeric values: `Sum`, `Min`, `Max`, `Mean`, `Count`, `First`, `Last`.
On any value: `Count`, `First`, `Last`.
`Len` on the group. Everything else → fallback.

### 5.4 Supported dtypes

`Int32, Int64, UInt32, UInt64, Float32, Float64, Boolean` (values and keys).
Int8/16, UInt8/16, Decimal, Date/Datetime, String, Categorical, List, etc. → fallback in M0/M1.

> Platform note: Apple Silicon has no `double` in MSL, so **both** Metal stacks
> (`metal` via wgpu/MSL and `metal-native`) reject `f64` shaders at compile
> time, as does Metal-backed `vulkan` (MoltenVK) on Apple. `f64` queries on
> those backends fall back to the CPU engine gracefully (pinned by the
> fallback half of the `f64` differential tests; see `supports_f64` in
> `polars-cubecl-testing`). `f32` and all integer/boolean kernels are fully
> verified on Apple GPUs. On CUDA/HIP/Vulkan-with-fp64 hardware, `f64` is
> differentially verified like every other dtype.

### 5.5 Null semantics (the contract to match)

| Situation | Polars behaviour | Our kernel must |
| --- | --- | --- |
| `x < 3` where `x` is null | row does **not** pass filter | treat null operand as `false` |
| `sum/min/max/mean` over a group | nulls are **ignored** | track `count_valid`; `mean = sum / count_valid` |
| group with all-null values | `sum` → `0` (int) / `0.0`, `count` → `0`, `len` → rows, `min/max/mean` → `null`, `first/last` → `null` | match per-agg nullability |
| group with all-`NaN` float values | `min/max` → `NaN` (valid!), `sum/mean` → `NaN` | skip `NaN` unless nothing else remains, then copy an input `NaN` |
| float comparisons (all ops) | total order: `NaN` sorts above `+inf`, `-0.0 == +0.0` | concrete float kernels using `.is_nan()` (the `x != x` idiom is unsound: wgpu folds it to `false` under fast-math) |
| `count` vs `len` | `count` ignores nulls; `len` counts rows | two distinct kernels |
| null key | nulls form their **own group** | host hash map with `None` key parts |
| boolean `NOT` | `NOT NULL` is `NULL` (excluded) | predicates lower to negation normal form; `NotColumn` is `!values & validity` |
| empty result | typed empty frame | produce `Series` with correct dtype, len 0 |

`mean` must be computed as `sum / count_valid` (not `sum / len`) and matches Polars' f64 promotion
rules for integer inputs.

---

## 6. Kernel design

All kernels are written once and compiled per backend. Topology constants (`ABSOLUTE_POS`,
`PLANE_DIM`, `CUBE_DIM`) are read, never hard-coded, so the AMD 64-lane / CPU 1-lane cases work.

### 6.1 `predicate` (elementwise)

Computes a `u8`/`bool` mask from the predicate DAG. Computed per leaf column → fused by the
lowering layer into one kernel invocation with multiple input buffers (M0 may compose simple
kernels instead of fusing).

### 6.2 `compact` (stable filter)

Filtering cannot be a pure elementwise mask because GPU writes must be ordered. Use a **stable
stream compaction with zero cross-thread communication**:

1. `count` kernel: thread `t` serially counts passing rows in its disjoint contiguous chunk
   (`STRIDE_ROWS = 16` rows), writing one `u32` per thread. No atomics, no shared memory.
2. The host prefix-sums the per-thread tallies (kilobytes of traffic) into scatter bases.
3. `scatter` kernel: each thread serially emits its passing rows at `base[t] + local`.

This preserves row order exactly (Polars `filter` preserves order), which the differential tests
assert. Reuse the same primitive for every column (compact the index vector once, then
`gather` all columns).

> Implementation note: the original proposal used an intra-block Kogge-Stone scan in shared
> memory. That was abandoned after measurement showed `sync_cube()` barriers to be unreliable
> on the CPU backend (racy cross-unit visibility that small single-barrier tests pass by
> timing luck). The strip-mined design needs no barriers, no shared memory, and no atomics,
> so it is deterministic by construction on every backend — including real GPUs, where it
> also avoids barrier overhead.

### 6.3 `gather` / `scatter`

Generic index-based column permutation, used by `compact`, `Slice`, and `SimpleProjection`.

> Movement is bitwise: `gather`/`slice` over `f64` data dispatch the `u64`
> kernel (identical bytes, no arithmetic), so `f64` columns move on device
> even on backends without `double`. Only arithmetic kernels (compare,
> reduce, `mean`, widening casts) require true `f64` and report
> `BackendUnsupported` — hence CPU fallback — where it is unavailable.

### 6.4 `group_by` (host grouping + device reduction)

Grouping (assigning dense group ids) runs on the **host** with an exact hash map over
normalized key tuples in first-seen order; reduction math runs on the **device**:

1. **Download keys** (only key columns cross back to the host) and group rows with
   `FxHashMap` (single-key fast path, `Vec`-tuple path otherwise). Float keys are normalized
   (`NaN` canonicalized, `-0.0` fused into `+0.0`); null key parts hash as their own group.
2. **Upload** the row→group assignment; the host already knows per-group sizes, so segment
   bases come from a host prefix sum — no device counting pass needed.
3. **Invert** on device: each row claims `slots[base[group] + seq[group]++] = row` with `u32`
   atomics (universally available). Slots form a set partition of the rows.
4. **Reduce** with one serial thread per group: `sum/min/max/count/mean` over the segment,
   `first`/`last`/keys gathered through host-computed row-order positions (deterministic,
   unlike atomic slot order). `mean = sum / count_valid` in `f64`; `len` is filled on the host.

Determinism: integer sums and `min/max/count/first/last` are exact regardless of slot order;
float sums may vary in the last ulps (tolerance-scoped, §differential-testing.md §4.5). Group
**order** follows first-seen insertion; tests sort by keys before comparing. `maintain_order`
stays unsupported (strict mode errors).

> Implementation note: a fully on-device hash probe was prototyped on paper and rejected:
> exact collision handling needs either spin-waits (deadlock-prone on some schedulers) or
> float atomics (absent on Metal/WGSL), plus multi-gigabyte tables at scale. The host/device
> split keeps every device kernel barrier-free and portable; a device prober can replace the
> host step later behind the same group-ids interface.

### 6.5 Reductions (no group-by)

`sum/min/max/count/mean` over a whole column: a plane-level tree reduction + global atomics. Used
by global aggregations and as a building block for `hash_group_by`.

---

## 7. Buffer / Arrow layer

```
polars ChunkedArray ──► Arrow ArrayRef (Series::to_arrow)  ──► device Array<T>
                                                               + device validity mask
device Array<T> + mask ──► root-side Vec<T> + Bitmap ──► Series::from_arrow ──► DataFrame
```

- **Values**: copy the Arrow buffer's `&[T]` into a device `Array<T>` (`client.create_from_slice`).
- **Validity**: upload as `Array<u32>` (packed bits) or `Array<u8>`; kernels branch on validity.
  M0 may use a `u8` mask for simplicity, packing later for bandwidth.
- **Chunks**: flatten to a single Arrow array on the host before upload, so kernels see one
  contiguous buffer. Output is a single chunk (chunk boundaries are semantically invisible).
- **Readback**: `client.read(&handle)` → `Vec<T>` → Arrow `PrimitiveArray` with the validity
  bitmap → `Series::from_arrow(name, array)` → `Column::from(series)` →
  `DataFrame::new_infer_height(columns)`.

Download/upload cost is the dominant term for small frames; hence `min_rows_for_gpu`.

---

## 8. Simplest path to a working prototype (M0)

The absolute minimum that proves architecture + correctness + a performance baseline:

**Vertical slice:** `DataFrameScan → Filter → Select(columns) → Slice`.

1. `polars-cubecl-buffer`: Arrow → device → Arrow for `i32/i64/f32/f64/bool`, with/without nulls.
2. `polars-cubecl-kernels`: `predicate`, `block_scan`, `compact`, `gather`.
3. `polars-cubecl-lower`: recognize exactly the M0 node/expr/dtype set; anything else → `Unsupported`.
4. `polars-cubecl`: `collect_strict` + `collect`; splice `IR::DataFrameScan`.
5. Differential tests on `cpu` + every available real backend, plus Criterion benches for filter.

M0 has **no** file IO and **no** group-by, which removes the two hardest subsystems (IO and
hash aggregation) from the critical first step while still exercising every layer end-to-end.

Then, in order, add: `GroupBy` (M1) → parquet `Scan` (M2) → partial/hybrid substitution (M3).
See [roadmap.md](./roadmap.md).

---

## 9. Workspace layout and dependencies

```
polars-cubecl/
├── Cargo.toml                      # workspace (members = ["bin/*", "crates/*"])
├── crates/
│   ├── polars-cubecl/              # facade: CubeclEngine, public API
│   ├── polars-cubecl-lower/        # ONLY crate that depends on polars-plan / polars-lazy IR
│   ├── polars-cubecl-runtime/      # ONLY crate that depends on cubecl backends
│   ├── polars-cubecl-kernels/      # #[cube] kernels (depends on cubecl only)
│   ├── polars-cubecl-buffer/       # Arrow ⇄ device buffers
│   └── polars-cubecl-testing/      # differential harness, generators, oracles (dev-dep)
└── bin/
    └── pcbe/                       # small CLI: run a query on a chosen backend (manual/bench)
```

> The template's `crates/common` and `bin/cli-app` are removed/replaced.

Dependency rules (per workspace `AGENTS.md`):

- Add with `cargo add`, never hand-typed versions. Workspace deps carry numeric versions only.
- Pin the unstable integrations with **exact** versions:
  - `polars = "=0.55.1"`, `polars-plan = "=0.55.1"`, `polars-lazy = "=0.55.1"`,
    `polars-core = "=0.55.1"`, plus `polars-arrow`, `polars-buffer`, `polars-error`,
    `polars-io`, `polars-utils` at `=0.55.1` (transitive-only crates such as
    `polars-config`/`polars-error` must also resolve to 0.55.1 — feature-gated
    code in `polars-plan` already broke once against a drifted 0.55.2).
  - `cubecl = "=0.11.0"` (pre-1.0, API churns), with `default-features = false`:
    the default feature set pulls persistence (`turso`) which cannot build on
    current nightlies; backends enable exactly `cubecl/<backend> + std + stdlib`.
- The CPU backend JITs through a prebuilt LLVM bundle downloaded at build time;
  see README for the offline pre-fetch procedure.
- `vendor/branches` carries a two-line toolchain-compat fix for `branches 0.4.6`
  (removed `core` intrinsic), reached via `turso` when `metal`/`vulkan` builds
  enable cubecl's C++ layer. Workspace-local `[patch.crates-io]` only.
- Backends are **Cargo features** on `polars-cubecl` (`cuda`, `hip`, `metal`, `vulkan`, `cpu`),
  defaulting to `["cpu"]` so the crate builds and tests anywhere (including CI without a GPU).
- Errors: library layer uses `thiserror`; the facade returns `PolarsResult` so fallback is natural.
- No new forbidden deps; Arrow comes via `polars-arrow`/`arrow` already in the Polars graph.

### 9.1 Version compatibility policy

Because Polars' IR is internal and versioned (`NodeTraverser::VERSION = (15, 2)` in
`crates/polars-python/src/lazyframe/visit.rs:61`):

- Supported Polars versions are pinned exactly and listed in `README.md` as a compatibility row.
- `polars-cubecl-lower` embeds a compile-time assertion on the Polars version and, at runtime,
  validates the IR shape it depends on; a mismatch → treat everything as unsupported (fallback),
  never a panic.
- CI runs the differential suite against the pinned version; a bump is a deliberate, tested change.

---

## 10. Performance strategy

1. **Measure first.** Criterion benches per crate: upload bandwidth, download bandwidth, filter
   throughput, group-by throughput, end-to-end vs CPU Polars, over `n = 1e5 … 1e8`.
2. **Threshold, not dogma.** Below `min_rows_for_gpu` rows (transfer-dominated), stay on the CPU.
   The engine should be *honest*: report whether it engaged the GPU.
3. **Keep data resident.** Pipeline ops on device buffers to avoid per-op round trips (M1+).
4. **Compact once, gather many.** Filter computes the index vector once, then gathers each column.
5. **Autotune selectively.** CubeCL autotune searches `CubeDim`/`vector` per kernel; cache results
   (CubeCL persists tuned configs). Do not autotune on first-touch of every query.
6. **Honest baseline reporting.** Report GPU-vs-CPU speedups per backend and per size, including
   the sizes where the GPU *loses*. The point is portability, not a rigged benchmark.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Polars IR changes (unstable, versioned) | engine silently miscompiles the plan | exact version pin; runtime shape check; unsupported ⇒ fallback; CI on pinned version |
| Null semantics mismatch | **wrong results** | conservative subset; exhaustive null differential tests; fallback on ambiguity |
| Float non-associativity (sum/mean order) | differential test "failures" | tolerance-based float compare; exact for integers; document |
| `group_by` order unspecified | spurious diffs | sort-by-keys before compare; no `maintain_order` support until M3 |
| CubeCL pre-1.0 churn | build breakage | isolate in `-runtime`; exact pin; feature-gated backends; `default-features = false` to dodge persistence |
| Backend-specific bugs (plane 32 vs 64 vs 1) | wrong on one backend | same corpus across all backends; cross-backend equality test; `POLARS_CUBECL_REQUIRE_BACKENDS` to forbid vacuous skips; backend features forwarded to the test harness so `-p` runs exercise the real matrix |
| Apple Metal rejects MSL `double` (both stacks + Metal-backed Vulkan) | `f64` queries cannot engage there | `supports_f64` gate: `f64` tests assert graceful fallback on Apple GPU backends, differential correctness elsewhere |
| CPU-backend barrier unreliability | flaky/wrong reductions | **no cross-unit communication anywhere**: strip-mined kernels, host prefix sums, atomics-only coordination |
| Transitive version skew (`polars-config` 0.55.2 vs 0.55.1) | feature-gated compile breakage | pin the whole polars family exactly, including transitive-only crates |
| `branches 0.4.6` vs current nightly (via `turso`, metal builds) | `metal`/`vulkan` features don't build | vendored two-line fix under `vendor/`, workspace-local patch |
| Backend-specific bugs (plane 32 vs 64 vs 1) | wrong on one backend | same corpus across all backends; cross-backend equality test |
| Upload dominates small frames | engine slower than CPU | `min_rows_for_gpu`; report engagement; bench across sizes |
| Silent fallback hides coverage bugs | tests pass vacuously | `raise_on_fail`/strict mode **and** assert the engine engaged (§testing §6) |
| No GPU in CI | untested kernels | `cpu` backend exercises kernels deterministically in CI; GPU jobs are a matrix |

---

## 12. Summary

- **Plug-in:** `_collect_post_opt` + `IR::DataFrameScan`; no fork; both public in Polars 0.55.1.
- **Portability:** CubeCL 0.11 compiles one kernel to CUDA/HIP/Metal/Vulkan/WGSL/CPU.
- **Scope:** filter + group-by-agg + scan over numeric dtypes; everything else falls back.
- **Safety:** unsupported ⇒ untouched plan ⇒ CPU; never wrong.
- **Correctness:** Polars CPU is the oracle; a strict differential suite across backends.

See [differential-testing.md](./differential-testing.md) for the correctness strategy and
[roadmap.md](./roadmap.md) for the milestone breakdown.
