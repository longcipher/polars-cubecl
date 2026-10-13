# Default recipe to display help
default:
  @just --list

# Format all code
format:
  rumdl fmt .
  cargo sort -w -g
  cargo +nightly fmt --all

# Auto-fix linting issues
fix:
  rumdl check --fix .
  RUSTC_WRAPPER= cargo +nightly clippy --fix --all --allow-dirty
  cargo workspace-inheritance-check --fix

# Run all lints
lint:
  typos
  rumdl check .
  cargo sort -w -g -c
  cargo +nightly fmt --all -- --check
  RUSTC_WRAPPER= cargo +nightly clippy --workspace --all-targets -- -D warnings
  cargo shear
  cargo workspace-inheritance-check

# Run tests
# Note: --all-features is skipped here because polars-ops 0.55.2 has a
# pre-existing compilation error with the `dtype-decimal` feature enabled
# (unhandled `Decimal` variant in `unique.rs`). Use `--all-features` only
# after bumping polars or patching the upstream match.
test:
  cargo test --workspace

# Run mutation tests with cargo-mutants, scoped to the CPU-testable crates
# (see .cargo/mutants.toml for the rationale).
# `RUSTC_WRAPPER` is cleared on purpose: the `kache` cache has served stale
# objects to mutant scratch builds in the past (mutants proven killable by
# hand survived a cached run), and a mutation gate must never pass on cache
# luck. Mutants also need `POLARS_CUBECL_REQUIRE_BACKENDS=cpu` in the
# environment so the (CPU-only) unit suites stay deterministic.
mutation:
  RUSTC_WRAPPER= cargo mutants --package polars-cubecl-lower --package polars-cubecl-buffer --package polars-cubecl-runtime

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

# Publish all crates to crates.io
publish:
  cargo publish --workspace

# Check for Chinese characters
check-cn:
  rg --line-number --column "\p{Han}"

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
  cargo install leptosfmt
  cargo install cargo-llvm-cov
  cargo install cargo-nextest
  cargo install cargo-deny
  cargo install cargo-audit

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
  cargo +nightly clippy --all-targets --all-features -- -W clippy::pedantic -W clippy::nursery -W clippy::cargo

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

# Check licenses
license-check:
  cargo deny check licenses

# Check advisories
advisory-check:
  cargo deny check advisories

# Check bans
ban-check:
  cargo deny check bans

# Full check (lint + test + build + audit)
full-check: lint test build audit
