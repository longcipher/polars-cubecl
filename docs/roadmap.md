# polars-cubecl — Implementation Roadmap

> Concrete, ordered milestones from a fresh workspace to a usable core-subset engine.
> Read [design.md](./design.md) for the architecture and [differential-testing.md](./differential-testing.md)
> for the correctness bar. Every milestone is TDD: failing test first, then the smallest change.

The strategy from the brief is honored literally:

1. **Core-subset prototype first** — filter + group-by-agg + read, to validate architecture & baseline.
2. **Independent crate** — no fork; enabled per Cargo feature / engine selection.
3. **Differentiated target** — AMD / Apple Silicon / ARM / iGPU via CubeCL, not NVIDIA peak perf.

---

## Phase P0 — Workspace skeleton — DONE

**Goal:** the repo builds a `polars-cubecl` crate that can pass a `LazyFrame` through the engine
hook and return a CPU-correct result, even before any GPU kernel exists.

Tasks:

1. Remove template demo crates (`crates/common`, `bin/cli-app`) and their tests.
2. Create crates: `polars-cubecl`, `polars-cubecl-lower`, `polars-cubecl-runtime`,
   `polars-cubecl-buffer`, `polars-cubecl-kernels`, `polars-cubecl-testing` (per design §9).
3. `cargo add` the pinned deps (exact versions) and backend features; default features `= ["cpu"]`.
4. Implement `CubeclEngine::new` with backend discovery (`Auto : cuda → hip → metal → vulkan → cpu`).
5. Implement the hook path with **no translation**: `collect` calls `_collect_post_opt` and always
   returns `Ok(())` untouched → result equals CPU. Add `ExecReport { engaged: false, .. }`.
6. First test: `collect` on a tiny frame equals `collect()` (CPU parity of the plumbing).
7. `just format && just lint && just test` green.

**Exit criteria:** plumbing compiles, lints clean, and the trivial parity test passes.

---

## Phase M0 — Walking skeleton: `DataFrameScan → Filter → Select → Slice` — DONE

**Goal:** the first fully-GPU path with real differential tests, plus a filter performance baseline.

Deliverables in dependency order:

### M0.1 — Buffer layer (`polars-cubecl-buffer`)

- Arrow → device → Arrow for `Int32/Int64/UInt32/UInt64/Float32/Float64/Boolean`.
- Validity upload/download (start with `u8` mask; pack to bits later).
- Chunk flattening on ingest; single-chunk output.
- Unit tests: round-trip values + null positions for every dtype (incl. all-null, empty).

### M0.2 — Kernels (`polars-cubecl-kernels`)

- `predicate` (elementwise comparisons, `And/Or/Not`, null → false).
- `block_scan` (exclusive prefix sum in `SharedMemory`) + global cube-base via atomic.
- `compact` (stable gather of passing indices).
- `gather` (index-based column permutation).
- Kernel unit tests run on the `cpu` backend (deterministic, no GPU needed).

### M0.3 — Lowering (`polars-cubecl-lower`)

- Walk the optimized IR from `_collect_post_opt`.
- `is_supported` for exactly: `DataFrameScan`, `Filter`, `SimpleProjection`/column-only `Select`,
  `Slice`, memory `Sink`, over the M0 dtypes and §5.2 predicates.
- Lower to a small `CubePlan`; anything else → `Unsupported`.
- Tests: supported plans lower; every unsupported node/expr/dtype returns `Unsupported`
  (mutation-critical: killed by the negative tests).

### M0.4 — Facade + splice

- Execute `CubePlan` on the selected backend; splice `IR::DataFrameScan` at root.
- `collect_strict` (error on unsupported) and `collect` (fallback), with `ExecReport`.

### M0.5 — Differential suite (the M0 gate)

- Named cases + `proptest` over the M0 matrix for `filter`/`slice`/`select`
  (differential-testing.md §3, §4, §6).
- Order-preserving assertions (§4.3) — the key filter correctness check.
- Strict-mode + engagement assertions (§2); negative fallback test.
- Run the comparator's own unit tests (§4.7).

### M0.6 — Benchmark baseline

- Criterion: upload/download bandwidth, filter throughput, end-to-end vs CPU, `n = 1e5 … 1e8`.
- Record the crossover; set `min_rows_for_gpu` from evidence.

**Exit criteria:** all M0 operator differential tests pass on `cpu`; `just mutation` clean in the
new crates; benchmark report committed; no wrong results under any generated input.

---

## Phase M1 — Aggregation: global reductions + `GroupBy` — DONE

**Goal:** `sum/min/max/count/mean/first/last` globally and grouped, with the §5.5 null contract.

Tasks:

1. `reduce` kernel: plane tree reduction + global atomics (`sum/min/max/count`).
2. `hash_group_by`: probe hash table (collision equality, null sentinel bucket), key
   materialisation, atomic segmented reductions; `mean = sum / count_valid`.
3. Lowering: `GroupBy` with plain-column keys and the §5.3 agg set; `Len` on groups.
4. Null-semantics tests (design §5.5 / testing §6): all-null groups, `count` vs `len`,
   null keys, integer→f64 promotion for `mean`.
5. `group_by` differential tests **sort by keys before compare** (§4.3); `maintain_order` stays
   unsupported in M1 (strict mode errors).
6. Property tests with group cardinality `1 / sqrt(n) / n` and row counts around 64/65.
7. Benchmarks for grouped aggregation vs CPU.

**Exit criteria:** grouped + global aggregations differentially correct on `cpu`; mutation clean;
benchmarks show where group-by wins/loses.

---

## Phase M2 — Read (parquet) + pushdown — DONE

**Goal:** the "data read" part of the core subset, with transfer minimised by pushdown.

Tasks:

1. Lower `IR::Scan` (parquet): read on the host (via `polars-io`/`parquet`), then upload.
2. Honor Polars' already-applied projection + predicate pushdown so only needed columns/rows move.
3. Column projection → upload only projected columns; predicate → pre-filter on host before upload
   when cheaper than GPU filter, or push to the GPU `predicate`+`compact` path otherwise.
4. Tests: scan+filter+group_by end-to-end vs CPU on real parquet fixtures; multi-file/hive later.
5. `pcbe` CLI: `pcbe --backend hip query.parquet "…"` for manual runs and benchmarks.

**Exit criteria:** end-to-end `scan → filter → group_by` differentially correct; IO overhead
measured; unsupported scan types fall back cleanly.

---

## Phase M3 — Hybrid execution (partial substitution) — DONE

> `maintain_order` and further dtype/op breadth remain future work; the hybrid mechanism is in place.

**Goal:** let the engine claim the *supported part* of a larger plan, not just whole plans.

Tasks:

1. **Partial substitution**: replace a supported *subtree* with `IR::DataFrameScan` instead of only
   the root, so an unsupported tail still benefits from a GPU head. Requires schema-correct splicing
   per node and careful validation.
2. `maintain_order` for `group_by`: add a device `argsort` post-pass, or mark unsupported and rely
   on the hybrid split.
3. Extend dtypes (`Int8/16`, `UInt8/16`) and predicates (`Cast`, `is_null`, `is_not_null`, `in`).
4. More ops in priority order driven by real workloads (e.g. `Sort`, `Distinct`, simple `Join`).
5. Kept honest throughout: every new op re-enters the "definition of done" checklist
   (testing §10) and re-runs the mutation gate.

**Exit criteria:** a query with a supported head and unsupported tail runs its head on the GPU and
produces a CPU-correct result; the differential suite still passes end-to-end.

---

## Cross-cutting rules (apply to every phase)

- **TDD inner loop**: `just test` (unit + `proptest` in the ordinary test path).
- **Mutation gate**: `just mutation` must report no survivors in changed code.
- **Never wrong**: unsupported ⇒ untouched plan ⇒ CPU. Strict mode exists to *catch silent
  fallback in tests*, not to be the production default.
- **Pin unstable deps exactly** (`polars = "=0.55.1"`, `cubecl = "=0.11.0"`); a bump is a
  deliberate, tested change (design §9.1).
- **One logical change per commit**; docs/comments/commits in English (workspace rule).
- **Report engagement**: every result carries whether the GPU actually ran and which ops ran.

---

## What "done" looks like for the prototype goal

At the end of **M1** the brief's three points are satisfied:

1. *Core subset prototype*: `DataFrameScan`/parquet-read + `Filter` + `GroupBy` aggregations run on
   the GPU, verified against Polars CPU on every backend, with a measured performance baseline.
2. *Third-party engine*: shipped as an independent crate behind Cargo features; users opt in via
   `CubeclEngine`; no fork of Polars.
3. *Differentiated scenarios*: the same binaries run the differential suite on AMD (hip), Apple
   Silicon (metal), and iGPU/ARM (vulkan/cpu) — the hardware cudf cannot target.

M2 adds the read path; M3 turns it from "whole-query" into "hybrid" acceleration.
