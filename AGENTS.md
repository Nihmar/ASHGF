# AGENTS.md — ASHGF

Repository for the master's thesis *"Adaptive Stochastic Historical Gradient-Free Optimization"* (Unipd). It contains the LaTeX source of the thesis, the original research code (frozen reference), and a new production-quality Python implementation used to validate, correct, speed up, and compare against the original.

## Repository layout

| Path | Purpose | Status |
|---|---|---|
| `thesis/` | LaTeX source (XeLaTeX, `Dissertate` class). Entry point: `thesis/dissertation.tex`. The math ground truth is in `chapters/stateoftheartalgorithms.tex` and `chapters/thealgorithm.tex`. | Read-only unless explicitly asked |
| `original_python_code/` | Original research prototype (GD, SGES, ASEBO, ASGF, ASHGF, benchmark suite, driver scripts). Run with its own `requirements.txt` (legacy `gym`). | **FROZEN — never modify.** Used only as behavioral ground truth |
| `src/ashgf/` | New implementation (uv-managed package). See "Conventions". | Active development |
| `tests/` | pytest suite: mathematical correctness vs analytic results, parity tests vs the frozen original | Active development |
| `comparisons/` | Markdown reports + generated figures comparing old vs new implementations (correctness, accuracy, runtime, FE counts). One report per experiment batch. | Grows over time |
| `pyproject.toml` | uv project definition | Maintained |

Note: `thesis/chapters/thealgorithm - Copia.tex` is an unreferenced stale copy (candidate for removal via a chore PR).

## Ground rules

1. **Everything in the repo is in English** (code, comments, docs, commits, PRs, reports).
2. `original_python_code/` is read-only. Fixes go into the new implementation; deviations are documented in `comparisons/`, never patched in place.
3. **The thesis formulas are the spec.** Where the original code deviates from the thesis, follow the thesis and document the deviation. Known deviations identified during audit (to be confirmed by tests):
   - `ashgf.py`: Gauss–Hermite estimator uses offset `σ·p_k` and prefactor `2/(σ√π)` instead of the thesis' `√2·σ·p_k` and `√2/(√πσ)` (legacy version has O(σ) bias).
   - `ashgf.py`: Lipschitz pair filter broken by operator precedence (`[i, j] or [j, i] not in buffer` is always true) → `L_j` taken over all pairs instead of the paper's set `I`.
   - `sges.py`: RNG re-seeded inside `grad_estimator` every iteration, and SGES directions overwritten by fresh random ones (guidance effectively disabled).
   - `functions.py`: `extended_feudenstein_and_roth` and `trid` differ from the thesis table; duplicate `liarwhd` definition.
4. All randomness comes from explicit `numpy.random.Generator` instances seeded from run configuration — never global `np.random.seed`.
5. Every algorithm tracks function evaluations (FEs). Results are stored as structured records enabling More & Sugrue performance/data profiles.
6. Naming compatibility: keep public names aligned with the thesis tables (e.g. the baseline called "GD" in the original code is actually a gradient-free central-difference random search — keep the name, clarify in docstrings).

## Conventions for the new implementation

- Senior-level quality: type hints, dataclasses for config/state, no mutable class-level state, no bare `except`, small testable modules, vectorized numpy (no Python loops over directions where avoidable).
- Dependencies managed exclusively with `uv` (`pyproject.toml` + lockfile). Runtime deps: `numpy`, `scipy`, `matplotlib`, `pandas`, `gymnasium`, `tqdm`. Dev deps: `pytest`, `ruff`.
- Benchmark functions must support batched evaluation `(n_points, dim)` for speed.
- Legacy behavior (where the original was wrong) may be exposed behind explicit flags purely for reproduction/comparison studies; defaults must match the thesis.

## Workflow

- One branch per piece (`feat/*`, `fix/*`, `docs/*`, `chore/*`), frequent small commits, push often, one PR per branch. Use `gh pr create` / `gh pr merge`.
- CI (GitHub Actions): `ruff check` + `pytest` on every PR.
- Comparison experiments land in `comparisons/<yyyy-mm-dd>-<topic>/` with a `report.md` and figures.

## Commands

```bash
uv sync                  # install dependencies
uv run pytest            # run tests
uv run ruff check src tests
uv run python -m ashgf --help   # CLI entry point
```
