# AGENTS.md — ASHGF

Repository for the master's thesis *"Adaptive Stochastic Historical Gradient-Free Optimization"* (Unipd). It contains the LaTeX source of the thesis, the original research code (frozen reference), and a new production-quality Python implementation used to validate, correct, speed up, and compare against the original.

## Repository layout

| Path | Purpose | Status |
|---|---|---|
| `thesis/` | LaTeX source (XeLaTeX, `Dissertate` class). Entry point: `thesis/dissertation.tex`. The math ground truth is in `chapters/stateoftheartalgorithms.tex` and `chapters/thealgorithm.tex`. | Read-only unless explicitly asked |
| `original_python_code/` | Original research prototype (GD, SGES, ASEBO, ASGF, ASHGF, benchmark suite, driver scripts). Run with its own `requirements.txt` (legacy `gym`). | **FROZEN — never modify.** Used only as behavioral ground truth |
| `src/ashgf/` | New implementation (uv-managed package). See "Conventions". | Planned — not yet created |
| `tests/` | pytest suite: mathematical correctness vs analytic results, parity tests vs the frozen original | Planned — not yet created |
| `comparisons/` | Markdown reports + generated figures comparing old vs new implementations (correctness, accuracy, runtime, FE counts). One report per experiment batch. | Grows over time |
| `references/papers/` | PDFs of the three SOTA papers (ASEBO NeurIPS'19, SGES IJCAI'20, ASGF arXiv'20) | Read-only reference |
| `requirements.txt` | Legacy dependency list of the frozen original (`gym`, `scikit-learn`, ...). Not used by the new implementation. | Frozen context only |
| `pyproject.toml` | uv project definition | To be created |

Note: `thesis/chapters/thealgorithm - Copia.tex` is an unreferenced stale copy (candidate for removal via a chore PR).

## Ground rules

1. **Everything in the repo is in English** (code, comments, docs, commits, PRs, reports).
2. `original_python_code/` is read-only. Fixes go into the new implementation; deviations are documented in `comparisons/`, never patched in place.
3. **The thesis formulas are the spec.** Where the original code deviates from the thesis, follow the thesis and document the deviation. Known deviations identified during audit (to be confirmed by tests):
   - `ashgf.py` / `asgf.py`: Gauss–Hermite estimator uses offset `σ·p_k` and prefactor `2/(σ√π)` instead of the thesis' `√2·σ·p_k` and `√2/(√πσ)`. Both rules are exact for linear `F` and consistent (error `O(σ²)`), but they quadrature different integrals, so their asymptotic error constants differ — confirm impact numerically before judging severity.
   - `ashgf.py`: Lipschitz pair dedup broken by operator precedence (`[i, j] or [j, i] not in buffer` is always truthy), so `buffer` holds both orders of each admissible pair. The set itself matches the thesis' `I` (index shift `floor(m/2)-1` 1-based ≡ `int(m/2)` 0-based), and since the quantity is symmetric in `(i,k)`, the resulting `L_j` is likely identical to the thesis value — verify with a test rather than assume.
   - `sges.py`: RNG re-seeded inside `grad_estimator` every iteration (the same direction matrix is regenerated at every step), and the guided directions returned by `compute_directions_sges` are immediately overwritten by `directions = self.compute_directions(dim)` — guidance is fully disabled.
   - `functions.py`: `trid` uses `+ Σ x_i x_{i-1}` where the thesis (and classical More–Sugrue Trid) has `− Σ x_i x_{i-1}`; `extended_feudenstein_and_roth` second term differs from the thesis table (`(5−x_d)x_d−2` vs `(x_d+1)x_d−14`); duplicate `liarwhd` definitions (the second silently wins).
   - `asebo.py`: depends on `sklearn.decomposition.PCA`; RL environments use the deprecated legacy `gym` API (`env.seed`).
   - `ashgf.py` / `asgf.py`: mutable class-level `data` dict shared across instances (`sigma_zero` rewritten per run); `σ₀` derived as `‖x₀‖/10` instead of being an input.
4. All randomness comes from explicit `numpy.random.Generator` instances seeded from run configuration — never global `np.random.seed`.
5. Every algorithm tracks function evaluations (FEs). Results are stored as structured records enabling More & Sugrue performance/data profiles.
6. Naming compatibility: keep public names aligned with the thesis tables (e.g. the baseline called "GD" in the original code is actually a gradient-free central-difference random search (= Nesterov random search / Salimans ES with M=d) — keep the name, clarify in docstrings).
7. Running the frozen original requires a separate environment: legacy `gym` (not `gymnasium`) and probably Python ≤ 3.11. Use a dedicated uv-managed venv so it never pollutes the main dev env.
8. Benchmark setup to replicate from `numericalexperiments.tex`: dims `{10, 100, 1000}`, functions marked `*` in `original_python_code/functions.txt` (~28), 10 seeds `[6177, 2832, 7361, 2778, 5416, 9652, 125, 978, 1487, 4156]`, FE budgets up to 1e8, tolerances τ ∈ {1e-2 … 1e-10}; FE accounting per iteration: ASGF/ASHGF ≈ `dim·(m−1)+1` (center node reuses f(x)), others ≈ `2·dim`.

## Conventions for the new implementation

- Senior-level quality: type hints, dataclasses for config/state, no mutable class-level state, no bare `except`, small testable modules, vectorized numpy (no Python loops over directions where avoidable).
- Dependencies managed exclusively with `uv` (`pyproject.toml` + lockfile). Runtime deps: `numpy`, `scipy`, `matplotlib`, `pandas`, `gymnasium`, `tqdm`. Dev deps: `pytest`, `ruff`.
- Benchmark functions must support batched evaluation `(n_points, dim)` for speed.
- Legacy behavior (where the original was wrong) may be exposed behind explicit flags purely for reproduction/comparison studies; defaults must match the thesis.

## Roadmap

1. **Setup** — uv project scaffold (`pyproject.toml`, lockfile, ruff config), GitHub Actions CI (`ruff check` + `pytest`), commit `references/papers/` (currently untracked), README refresh.
2. **Core math library** (`src/ashgf/`) — batched benchmark function suite implementing the thesis formulas exactly (legacy variants behind explicit flags); Gauss–Hermite DGS estimator module; analytic-gradient tests proving estimator accuracy and the O(σ²) behavior.
3. **Algorithms** — dataclass-based implementations of GD (random-search baseline), SGES, ASEBO, ASGF, ASHGF faithful to the thesis, with explicit `np.random.Generator`s, exact FE counting, structured `RunResult` records (trajectory, FEs, best-so-far) ready for More & Sugrue profiles.
4. **Parity & comparison harness** — port the frozen original into its own legacy env; parity tests where behavior must match; speedup/correctness benchmarks landing in `comparisons/`.
5. **Thesis replication** — reproduce the numerical-experiments chapter (dims 10/100/1000, selected functions, 10 seeds, performance/data profiles) with both implementations side by side.

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
