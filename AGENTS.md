# polars-cubecl Agent Instructions

## Non-negotiables

- Surgical edits only: every changed line must trace to the request. No drive-by refactors, no reformatting untouched code, no deleting pre-existing dead code (mention it instead).
- Fix root causes. Never silence a warning, weaken a check, or patch a symptom to make a gate pass.
- Never report completion while any `just` gate below fails. Report the failure verbatim and stop.
- Stop and ask when a request is ambiguous, conflicts with existing conventions, or requires credentials, publishing, or destructive operations.
- When a correction reveals a missing rule, add one specific rule here ("always X for Y", never "be careful with Y"); delete rules that never catch real errors.

## Scope

- `crates/polars-cubecl` is the public facade; `polars-cubecl-lower` lowers a Polars plan to kernels, `polars-cubecl-buffer` owns the Arrow/device buffer boundary, `polars-cubecl-kernels` holds the `#[cube]` kernel bodies, `polars-cubecl-runtime` selects the backend and device, `polars-cubecl-testing` is the differential-test harness.
- `bin/pcbe` is the CLI, `bin/cudf-bench` the Criterion comparison bench. `vendor/branches` is a vendored `[patch.crates-io]` crate, not workspace code.
- The engine implements a documented subset of `LazyFrame`; anything outside it falls back to the Polars CPU engine. Never widen the subset without a matching differential test.
- `docs/design.md`, `docs/differential-testing.md`, `docs/roadmap.md` and `docs/benchmarks.md` hold the architecture, the correctness strategy, the milestone plan and the measured numbers.
- English only in code, comments, docs, and commit messages.

## Execution

- Parallelize aggressively: dispatch subagents for independent research, exploration, and implementation.
- NEVER run two `cargo` commands concurrently, and never overlap file edits with a running `cargo` command (`target/` file lock): finish edits, wait, then run.
- NEVER use `git worktree`. Edit the current branch in place.
- TDD for behavior changes: failing test first, smallest change to pass, then run the mutation gate only when the current spec task declares `Mutation:`, and fix surviving mutants.

## Verification loop

After each feature or bug fix, in order; any failure means the task is not done:

```bash
just format
just lint
just test
```

Commit hooks run the fast subset (`just hooks`): fmt check, cargo sort,
clippy, typos, rumdl, file hygiene, Kingfisher secret scanning. Never add
`cargo test` to the hook; tests stay in `just test`.

Never commit a real credential: keep `.env` and `.envrc` untracked, and
commit a placeholder `.env.example` instead.

Mutation testing is intentionally absent from the default loop and from CI. When a spec task
declares `Mutation: <src-glob>`, follow the `/pb-build` gate: write the changed file's diff to a
temp file and run `cargo mutants --in-diff <diff-file>` for that file only, with at most 3 files /
60 mutants / 300 s. Never add a whole-repo mutation recipe: `just check-mutation-scope` fails
`just lint` when an unbounded mutation command appears in a build or CI file.

`just lint` is the CI gate: `typos`, `rumdl`, `cargo sort --check`, `cargo +nightly fmt --check`, `cargo +nightly clippy -D warnings`, Cargo's native `unused_dependencies` lint. The mutation gate mutates only the changed lines, within the /pb-build budget (at most 3 files / 60 mutants / 300 s).

## Rust workspace rules

- Edition 2024, resolver 3; workspace lints live in the root `Cargo.toml` and every member opts in with `[lints] workspace = true`.
- `polars-*` is pinned to `=0.55.2` and `cubecl*` to `=0.11.0` (internal IR must match exactly, CubeCL is pre-1.0). Bump them in a dedicated change and fix the fallout there.
- `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `print_stdout`, `print_stderr` and `allow_attributes` are `deny`. The only escape hatch is `#[expect(lint, reason = "...")]`.
- Backends are cargo features (`cpu` on by default; `cuda`, `hip`, `metal`, `metal-native`, `vulkan`, `parquet` off). Exactly one kernel source is compiled per backend: never link two GPU stacks into one test process.
- `POLARS_CUBECL_BACKENDS` narrows the test matrix; `POLARS_CUBECL_REQUIRE_BACKENDS` turns a skipped backend into a failure so a CI matrix cannot pass vacuously.
- `just test` deliberately omits `--all-features`: the pinned `polars 0.55.2` tree does not build with `dtype-decimal` enabled (unhandled `Decimal` variant upstream in `unique.rs`). Re-enable it only after a `polars` bump.
- `.cargo/config.toml` sets `rustc-wrapper = "kache"`; clippy invocations must clear it (`RUSTC_WRAPPER=`) exactly as the Justfile does.
- The toolchain is nightly (`rust-toolchain.toml`): use `cargo +nightly fmt` and `cargo +nightly clippy`, never bare `cargo fmt`.
- Never hand-type dependency versions. Shared crates go into root `[workspace.dependencies]` and members reference them with `workspace = true`.

## Rust 1.100 baseline

- Workspace MSRV is Rust 1.100. Prefer newly stabilized features and APIs over older workarounds; do not add feature gates for anything already stable in 1.100.
- `[workspace.lints.cargo]` enables Cargo's native `unused_dependencies` lint, the only unused-dependency check; do not reintroduce a third-party checker. Its level is `warn`, and `.cargo/config.toml` sets `build.warnings = "deny"`, so a finding fails the gate.
- `.cargo/config.toml` sets resolver `incompatible-rust-versions = "fallback"` (MSRV-aligned resolution) and `registry.global-min-publish-age = "2 days"` (Rust 1.100 dependency cooldown); keep both.

## Dependency choices

- The workspace already declares everything it needs: `cubecl`/`cubecl-runtime`, `polars-*`, `bytemuck`, `rustc-hash`, `thiserror`, `tracing`, `proptest`, `criterion`, plus `eyre` and `clap` in `bin/pcbe`.
- Forbidden here: `anyhow`, `log`, `reqwest`/`hpx`, `tokio`, `sqlx`, `dashmap`, `scc`. The engine is synchronous and has no HTTP, database or async runtime.
- `proptest` stays inside the ordinary `cargo test` loop; it is not a separate test command.

## Code conventions

- Errors: `thiserror` in `crates/`, `eyre` in `bin/pcbe`. Concrete error types through `?`; never string errors.
- Logging: `tracing` only. Library crates never write to stdout/stderr.
- `unsafe` exists only in `polars-cubecl-kernels` (the CubeCL launch path) and always carries a `// SAFETY:` comment.
- Null semantics are the contract: null comparisons never pass, aggregations ignore nulls, null keys form their own group, and float ordering is total with `NaN` above `+inf`.
- Apple GPUs have no MSL `double`, so `f64` queries fall back to the CPU backend there. Do not "fix" that by widening the fallback.
- The correctness oracle is the Polars CPU engine, compared by the harness in `polars-cubecl-testing`: schemas must match exactly and row order must match unless the specification says otherwise.
- Keep `clippy::pedantic` clean; the noisy pedantic lints are allowed once in the workspace table, never per call site.
