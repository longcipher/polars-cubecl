# polars-cubecl — Differential Testing Strategy

> Correctness is the only reason a subset GPU engine is acceptable. This document defines how
> correctness is proved: **Polars' own CPU engine is the oracle**, and every result produced by
> `polars-cubecl` is compared against it, on every backend, for every supported operator.

See [design.md](./design.md) §1.3 for the safety property this suite enforces.

---

## 1. Core principle: the oracle is Polars CPU

For every query `q` and input `df`, define:

```
expected = lf.clone().collect()?               // Polars CPU / streaming engine
actual   = CubeclEngine::new(cfg).collect_strict(lf)?   // polars-cubecl
assert_frame_equal(actual, expected, rules)
```

`expected` is produced by the *same* `LazyFrame`, so this is a true differential test: it compares
two independent implementations of the same semantics. The oracle is authoritative because Polars
CPU behaviour **is** the specification.

---

## 2. The engine must prove it ran (anti-vacuous-pass)

A naive differential test passes even if the engine silently fell back to CPU. That would let a
completely broken engine "pass" the whole suite. Two mandatory guards:

1. **Strict mode.** Tests always use `EngineConfig { raise_on_fail: true, .. }`
   (`collect_strict`). In strict mode an unsupported plan or kernel error is a **test failure**,
   not a fallback.
2. **Engagement assertion.** The engine returns an `ExecReport`; every differential test asserts
   `report.engaged == true` and, when the plan has known ops, that the expected ops ran
   (`report.ops_ran` contains the op). A test that claims to exercise a kernel must observe it.

```rust
let (actual, report) = engine.collect_strict_with_report(lf)?;
assert!(report.engaged, "engine fell back silently: {report:?}");
assert_eq!(report.ops_ran, [Op::Filter]);
assert_frame_equal(&actual, &expected, &rules)?;
```

Additionally, a **negative test** asserts the opposite: for an *unsupported* query, strict mode
errors and non-strict mode produces a CPU-correct result (`report.engaged == false`). This proves
the fallback path both works and is detectable. Fallback comparisons of order-free operators
(`group_by` without `maintain_order`) sort both sides by key first: the CPU oracle itself
returns groups in nondeterministic order across collects, so an unsorted fallback comparison
would flake independently of the engine.

One caveat: a strict-mode attempt that reaches *device execution* and fails there (e.g. a
shader the backend compiler rejects) can poison the backend's shared client for concurrently
running tests. Fallback checks for queries that fail at launch therefore use the quiet
variant (`check_quiet_fallback*`), which pins only the production behavior — non-strict
never engages and stays CPU-correct — without attempting strict mode. The "unsupported
implies strict errors" property itself is pinned by fallback tests whose queries are
rejected at lowering time (before any launch) on every backend.

---

## 3. Test data generation

Two layers, both in `polars-cubecl-testing`:

### 3.1 Hand-written named cases (readable, must-not-regress)

Explicit, documented edge cases per operator:

- empty frame (0 rows); 1 row; 1 column; all-null column; all-value-null groups
- duplicate group keys; a single group; keys==rows (all unique)
- `i32::MIN/MAX`, `i64::MIN/MAX`, `u32/u64::MAX`, `-0.0`, `+0.0`, `NaN`, `±inf`
- filter predicate that is always-true / always-false
- slice at boundaries: `0..0`, `0..n`, `n..n`, `k..n` where `k>n` (→ empty), negative offset
- mixed null positions in keys and values

### 3.2 Property tests (`proptest`, in the normal `cargo test` loop)

Generators produce `DataFrame`s over a configurable matrix:

| Dimension | Values |
| --- | --- |
| dtypes | `Int32, Int64, UInt32, UInt64, Float32, Float64, Boolean` |
| null density | `0.0, 0.05, 0.5, 1.0` |
| rows | `0, 1, 2, 7, 63, 64, 65, 1000, 100_000` (boundary values around cube/plane sizes) |
| columns | `1, 2, 5` |
| chunks | 1 and (when supported) multi-chunk input |
| group cardinality | `1, sqrt(n), n` distinct keys |

Property: **for all generated inputs and all supported queries, `actual == expected`** under §4.
Row counts spanning 63/64/65 specifically stress block-scan and plane boundaries where
off-by-one compaction bugs live.

Determinism: generators are seeded; failing cases shrink automatically and are promoted into §3.1.

---

## 4. Comparison rules (strict but correct)

`assert_frame_equal` must be *strict enough to catch real bugs* and *correct enough to not fail
on semantically-equal-but-differently-computed results*.

### 4.1 Schema — strictly equal

- Column **names**, in order.
- Column **dtypes**, exactly (e.g. `Int64` vs `Int32` is a failure; Polars does not implicitly
  unify these in `agg`).

### 4.2 Shape

- Row count exactly equal.

### 4.3 Row order

- **Order-preserving ops** (`filter`, `slice`, `select`, `DataFrameScan`): rows must match in
  **original order**, position by position. This is the strongest check and catches any unstable
  compaction.
- **`group_by` without `maintain_order`:** Polars guarantees no order. Normalise both frames by
  **sorting on the key columns** (and, for ties, on all columns) before comparing. Document that
  order is explicitly out of contract; never "fix" the engine to match CPU order.
- When `maintain_order` is requested: unsupported in M1 → strict mode errors (correctly).

### 4.4 Nulls

- Null **position** must match exactly, for every column.
- `NaN` is not null and must be distinguished from it.

### 4.5 Values by dtype

| dtype | rule |
| --- | --- |
| `Int8..64`, `UInt8..64`, `Boolean` | exact equality |
| `Float32/Float64` | `NaN` matches `NaN`; `±inf` exact; else `|a-b| <= max(atol, rtol*|b|)` |
| (later) `String`, `Categorical` | exact |

Float tolerance is **required** because GPU reductions reorder additions; it is **scoped** (never
applied to ints/bools) so it cannot mask a real bug. Defaults: `rtol = 1e-6`,
`atol = 1e-9` (f64) / `1e-5` (f32). Sums of large-magnitude values may need `rtol` raised per
test, with a comment justifying it.

### 4.6 Chunking is invisible

Comparison is over **values**, not chunk layout. Flattening differences between CPU and GPU
output must not fail a test.

### 4.7 The comparator itself is tested

`assert_frame_equal` is unit-tested with known-equal and known-unequal pairs (schema, dtype,
order, null-position, float-tolerance, NaN) so a too-lax comparator cannot silently pass
everything. This is part of the test suite, not an assumption.

---

## 5. Backend matrix

The same corpus runs against every available backend:

```
cpu  (always)  ── parity reference in CI with no GPU
cuda (if NVIDIA)  hip (if AMD)  metal (if macOS/Apple)  vulkan/wgpu (if present)
```

Two equalities are asserted:

1. **engine vs CPU oracle** for each backend (§1).
2. **backend vs backend**: GPU results must equal each other (modulo the same float tolerance).
   This isolates backend-specific lowering bugs (plane 32 vs 64 vs 1, atomics availability).

Backends that are unavailable are skipped with an explicit reason; the `cpu` backend is never
skipped, so the kernels are always exercised.

---

## 6. Operator / semantic coverage

Per operator, a dedicated test module asserts the §5.5 null-semantics contract of design.md:

- `filter`: null-in-predicate → row excluded; `And/Or/Not` truth tables incl. nulls.
- `select`/projection: column order + dtype preserved; zero-width projection.
- `slice`: offset/len boundaries; negative offset; len past end.
- `group_by` aggregations:
  - `sum` all-null group → `0`; `count` ignores nulls; `len` counts rows.
  - `min/max/mean` all-null group → `null`; `mean = sum / count_valid` (integer promotion to f64).
  - null keys form their own group.
  - empty input → typed empty frame.
- dtype coverage: each §5.4 dtype appears in at least one test per operator.

These are the "named business cases" required by the workspace TDD rules; they live next to the
code (`#[cfg(test)]`) where practical, and in `polars-cubecl-testing` where shared.

---

## 7. Mutation testing

Per workspace rules, after the inner loop is green run `just mutation` (cargo-mutants) and kill
surviving mutants. Priority mutation targets:

- subset guards in `-lower` (`require_dtype`, `reject_nans`, the literal validators,
  `supported_column`/`numeric_column`) — a mutant that *widens* the subset must be killed
  by a differential failure or an engagement assertion;
- the null/validity branches in kernels and lowering;
- boundary comparisons in `gather`/`slice`/segment loops (`<` vs `<=`, `+1`/`-1`);
- the fallback path itself (a mutant that turns "unsupported" into "supported" must fail).

A surviving mutant in `-lower`'s subset guards is a **coverage hole in this suite**, and is
treated as a test bug, not merely a code smell. Scoping and documented exclusions live in
`.cargo/mutants.toml`.

---

## 8. Benchmarks (baseline, not pass/fail)

Part of "validate the performance baseline" from the brief. Criterion benches in the affected
crates, run with `just bench`:

- upload / download bandwidth (GB/s) by dtype;
- `filter` throughput (rows/s, GB/s) vs CPU Polars;
- `group_by` throughput vs CPU Polars;
- end-to-end latency vs `n = 1e5 … 1e8`, showing the crossover where GPU starts to win.

Benchmarks are **reported**, not asserted, so they never flake CI. They exist to make the
`min_rows_for_gpu` threshold an evidence-based number and to document where the GPU loses.

---

## 9. CI wiring

| Job | Runs |
| --- | --- |
| `just test` (every push) | unit + property + differential on **`cpu`** backend; strict mode |
| `just mutation` | cargo-mutants on `-lower`, `-buffer`, `-runtime` (see `.cargo/mutants.toml` for scope) |
| `just lint` / `just format` | workspace lints (pedantic/nursery) and rustfmt |
| `bench` (scheduled/manual) | Criterion; artifacts uploaded, no gate |
| GPU matrix (optional, self-hosted) | differential suite on `hip` (AMD), `metal` (macOS), `vulkan` |

The `cpu` backend makes the entire correctness suite runnable on any machine, including CI
without a GPU. GPU jobs add coverage for backend-specific lowering, they are not the primary gate.

---

## 10. Definition of done for any new operator

An operator is "supported" only when **all** hold:

1. Its lowering path in `-lower` accepts exactly the supported shapes and returns `Unsupported`
   for everything else.
2. Named cases (§3.1) for the operator pass on the `cpu` backend.
3. A property test (§3.2) over the dtype/null/row matrix passes.
4. Null-semantics cases (§5.5 / §6) pass.
5. Strict-mode engagement assertions (§2) pass, and the negative fallback test passes.
6. Backend matrix (§5) passes on every available backend.
7. `just mutation` reports no survivors in the new code.
8. `just format && just lint && just test` are green.
