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
pip install -e ".[dev]"              # editable install + test deps
pytest tests/ -v                     # all tests (47 pass)
pytest tests/ -v -m "not slow"       # skip slow tests
python -m ashgf benchmark ...        # CLI entry point
```

Python 3.10+ required; confirmed working on 3.14.

### Rust

```bash
cargo build --release      # always use --release for benchmarks/timings
cargo test                  # unit tests (inline #[cfg(test)]; no separate tests/ dir)
cargo test --release        # also run in release for realistic timing
cargo test -- --ignored     # include #[ignore] golden-file regression tests
cargo bench                 # criterion micro-benchmarks
```

**Linux prerequisite**: `libopenblas-dev` (OpenBLAS headers for `ndarray-linalg`).
Windows builds fail — use Python instead.

## Key commands

```bash
# Single test
pytest tests/test_algorithms.py::test_gd_converges -v

# Benchmark: thesis algorithms + ASGF-2S (the recommended algorithm)
python -m ashgf benchmark --algos gd sges asgf ashgf asebo asgf-2s --dims "10,100" --iter 500 --patience 50 --seed 2003 --output results

# Re-run only changed algorithms (existing CSVs are preserved, plots are regenerated)
python -m ashgf benchmark --algos asgf-2s --dim 10 --iter 500 --patience 50 --seed 2003 --output results
```

## Architecture notes

- **Template Method**: `BaseOptimizer.optimize()` drives the loop. Key hooks to override:
  - `grad_estimator(x, f)` — abstract, must return estimated gradient
  - `_compute_step(x, grad, f, maximize)` — computes `x_new`, overridable for line-search  
  - `_before_gradient(x)` — can modify `x` before gradient estimation
  - `_post_iteration(iteration, x, grad, f_val)` — adaptation after each step
  - `_get_step_size()` — returns step size (default 1.0)
  - `_setup(f, dim, x)` — called once before the loop
- **ASGF** saves `self._f_at_x` in `grad_estimator` — subclasses can read it in `_compute_step` instead of calling `f(x)` again.
- **ASGF's basis rotation** uses Householder reflection (`_rotate_basis_householder`, O(d²)), not QR (O(d³)). This was a performance optimization.
- Rust tests are **inline** (`#[cfg(test)]`) in each source file — no `tests/` directory.
- Python tests live in `tests/` with `conftest.py` providing fixtures (seeded RNG, default dim).
- **Seeds always explicit**: `np.random.default_rng(seed)` in Python, `SeededRng` in Rust.
- Parallel function evaluation: `ASHGF_N_JOBS` env var (Python), `--jobs N` CLI flag (Rust).

## Algorithm landscape

| Algorithm | Description | Status |
|-----------|-------------|--------|
| **ASGF-2S** | Frequency-gated 2x step + safety gate + smooth blending | **Recommended** — ~60/78 wins at dim=10,100 |
| ASGF | Adaptive basis rotation + sigma adaptation | Baseline |
| ASHGF | ASGF + gradient-history covariance | Worse than ASGF on ~2/3 of functions |
| GD, SGES, ASEBO | Thesis algorithms | Outperformed by ASGF variants |

The `benchmark` subcommand auto-generates PNG plots (bar chart, convergence grid, per-function).

## Performance

- Householder basis rotation is O(d²) — the main optimization loop bottleneck.
- `gauss_hermite_derivative` evaluates `f` at `d × (m-1)` points per iteration (broadcast-vectorised).
- All function implementations are vectorised NumPy — no Python loops in hot paths.
- The `_cached_arange(n)` helper avoids repeated `np.arange` allocations across function calls.

## Rust feature flags

- `pca` (default) — enables `ASEBO` via `smartcore`
- `rl` — enables RL environments (not yet implemented)
