# Default recipe to display help
default:
  @just --list

# Format markdown, TOML and Rust sources
format:
  rumdl fmt .
  cargo sort -w -g
  cargo +nightly fmt --all

# Auto-fix what `format` cannot: markdown lints, clippy findings, inheritance drift
fix:
  rumdl check --fix .
  RUSTC_WRAPPER= cargo +nightly clippy --fix --all --allow-dirty
  cargo workspace-inheritance-check --fix

# Run all lints (every tool used here is installed by .github/workflows/ci.yml)
lint:
  typos
  rumdl check .
  cargo sort -w -g -c
  cargo +nightly fmt --all -- --check
  RUSTC_WRAPPER= cargo +nightly clippy --workspace --all-targets -- -D warnings
  cargo shear

# cargo-workspace-inheritance-check is not in the CI image, so it is kept out of `lint`.
#
# Check that every member inherits version/edition/license/repository from the workspace
inheritance-check:
  cargo workspace-inheritance-check

# `--all-features` is deliberately not used: the pinned `polars 0.55.2` tree
# fails to compile with the `dtype-decimal` feature enabled (upstream reports
# an unhandled `Decimal` variant in `unique.rs`). Re-enable `--all-features`
# only after bumping `polars` or patching the upstream match.
#
# Run the workspace test suite on the default (cpu) backend
test:
  cargo test --workspace

# `RUSTC_WRAPPER` is cleared on purpose: the `kache` cache has served stale
# objects to mutant scratch builds before (mutants proven killable by hand
# survived a cached run), and a mutation gate must never pass on cache luck.
# See .cargo/mutants.toml for the scope and the excluded-mutant rationale.
#
# Run mutation tests with cargo-mutants on the CPU-testable crates
mutation:
  RUSTC_WRAPPER= POLARS_CUBECL_REQUIRE_BACKENDS=cpu cargo mutants --package polars-cubecl-lower --package polars-cubecl-buffer --package polars-cubecl-runtime

# Run tests with coverage
test-coverage:
  cargo llvm-cov --all-features --workspace --output-dir coverage

# Build entire workspace
build:
  cargo build --workspace

# Check all targets compile
check:
  cargo check --all-targets --all-features

# Publish all crates to crates.io (dry run)
publish-check:
  cargo publish --workspace --dry-run --allow-dirty

# Publish all crates to crates.io (irreversible: run only with a registry token)
publish:
  cargo publish --workspace

# Fail if Chinese characters are present in tracked files
check-cn:
  #!/usr/bin/env bash
  set -euo pipefail
  if rg --line-number --column "\p{Han}" . ; then
    echo "error: Chinese characters found" >&2
    exit 1
  fi
  echo "ok: no Chinese characters found"

# Full CI check
ci: lint test build

# ============================================================
# Maintenance & Tools
# ============================================================

# Clean build artifacts
clean:
  cargo clean

# Install all required development tools
setup:
  cargo install cargo-mutants
  cargo install cargo-shear
  cargo install cargo-sort
  cargo install cargo-workspace-inheritance-check
  cargo install typos-cli
  cargo install rumdl
  cargo install cargo-llvm-cov
  cargo install cargo-nextest
  cargo install cargo-audit
  cargo install cargo-outdated

# Generate documentation for the workspace
docs:
  cargo doc --no-deps --open

# Run security audit
audit:
  cargo audit

# Check for outdated dependencies
outdated:
  cargo outdated

# Run clippy with all lints
clippy-all:
  RUSTC_WRAPPER= cargo +nightly clippy --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo

# Run tests with nextest
test-nextest:
  cargo nextest run --all-features

# Run miri on unsafe code
miri:
  cargo +nightly miri test --all-features

# Check for unused dependencies
unused-deps:
  cargo shear --check

# Update all dependencies
update-deps:
  cargo update

# Full check (lint + test + build + audit)
full-check: lint test build audit
