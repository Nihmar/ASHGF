# AGENTS.md — ASHGF

Dual-language derivative-free optimization: Python (`ashgf/`) is the reference,
Rust (`src/`) is the high-performance port. Architectures mirror each other.

## Immutable directories

- **`src_old/`** — deprecated original Python. Never modify, never use as reference.
- **`thesis/`** — original LaTeX thesis. Never modify.

When in doubt, `ashgf/` is the authoritative implementation for correctness.

## Build & run

### Python

```bash
uv sync                    # create venv + install deps (use --group dev / --group rl as needed)
uv run pytest tests/ -v    # all tests
uv run pytest tests/ -v -m "not slow"  # skip slow tests
uv run python -m ashgf <command> ...   # or: uv run ashgf <command> ...
```

### Rust

```bash
cargo build --release      # always use --release for benchmarks/timings
cargo test                  # unit tests (inline #[cfg(test)]; no separate tests/ dir)
cargo test --release        # also run in release for realistic timing
cargo test -- --ignored     # include #[ignore] golden-file regression tests
cargo test --doc            # doc tests
cargo bench                 # criterion micro-benchmarks (no benches/ dir yet)
```

**Prerequisite on Linux**: `libopenblas-dev` (OpenBLAS headers for `ndarray-linalg`).

## Key test commands for fast feedback

```bash
# Python: single test
uv run pytest tests/test_algorithms.py::test_gd_converges -v

# Rust: single test or module prefix
cargo test gd_improves_on_sphere
cargo test gradient::
cargo test algorithms::
cargo test functions::
```

## Feature flags (Rust)

- `pca` (default) — enables `ASEBO` algorithm via `smartcore`
- `rl` — enables RL environments (not yet implemented in Rust)

## Architecture notes

- Rust tests are **inline** (`#[cfg(test)] mod tests`) in each source file. There is no `tests/` directory for Rust.
- Python tests live in `tests/` with `conftest.py` providing fixtures (seeded RNG, default dim, etc.).
- The `Optimizer` trait / `BaseOptimizer` class uses **Template Method** pattern:
  `optimize()` drives the loop, `grad_estimator()` is the abstract hook.
- All algorithms share early-stopping: NaN/Inf guards, `patience` (stagnation), `eps` (step convergence), `ftol`.
- **Seeds are always explicit** — via `SeededRng` in Rust, `np.random.default_rng(seed)` in Python. No global RNG state.
- Parallel evaluation: `ASHGF_N_JOBS` env var (Python), `--jobs N` CLI flag (Rust benchmark).
- The `benchmark` subcommand in both languages **auto-generates PNG plots** (bar chart, convergence grid, per-function).
- Rust has two extra algorithms beyond Python: `ASHGFNG` (next-gen), `ASHGFS` (simplified).
- `ashgf/` was ported directly; `src_old/` was not used for the Rust port. Bug fixes documented in README §"Bug Fixes and Mathematical Verification" are authoritative over old code.
