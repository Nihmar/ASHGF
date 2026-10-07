# Project Plan — ASHGF Reimplementation

Status: active. Last updated: 2026-07-07.

This document tracks the plan to build a production-quality Python implementation of the
algorithms in the master's thesis *"Adaptive Stochastic Historical Gradient-Free Optimization"*
(Unipd), validate it against the mathematical specification, speed it up, and compare it side by
side with the frozen original prototype in `original_python_code/`.

Binding rules live in [AGENTS.md](./AGENTS.md); this file is the execution roadmap.

## Objectives

1. **Correctness** — every algorithm must implement the thesis formulas exactly (the thesis is the
   spec). Where the original code deviates from the thesis, follow the thesis; expose legacy
   behavior behind explicit flags purely for reproduction/comparison studies.
2. **Speed** — vectorized, batched numpy implementations. Target orders-of-magnitude speedups over
   the original at dim ≥ 100, measured in `comparisons/`.
3. **Reproducibility** — all randomness driven by explicit `numpy.random.Generator`s seeded from run
   configuration (never global seeding); exact function-evaluation (FE) counting; structured result
   records enabling More & Sugrue performance/data profiles.
4. **Comparison** — markdown reports (correctness, accuracy, runtime, FE counts) comparing old vs
   new implementations, one report per experiment batch under `comparisons/<yyyy-mm-dd>-<topic>/`.

## Audit summary (frozen original vs thesis)

| Area | Finding | Severity / action |
|---|---|---|
| `ashgf.py`, `asgf.py` — GH estimator | Offset `σ·p_k`, prefactor `2/(σ√π)` instead of the thesis' `√2·σ·p_k`, `√2/(√πσ)`. Both rules are exact for linear `F` and `O(σ²)`-consistent, but they quadrature different integrals → different asymptotic error constants. | Implement thesis formula; quantify impact numerically in Phase 1 tests |
| `ashgf.py` — Lipschitz pair filter | `[i, j] or [j, i] not in buffer` always truthy → dedup never happens, both orders stored. The pair set still equals the thesis' `I` (index shift only) and the quantity is symmetric in `(i,k)` → likely numerically harmless. | Confirm with a dedicated test, do not assume |
| `sges.py` | RNG re-seeded inside `grad_estimator` every iteration (same direction matrix each step) AND guided directions overwritten by fresh random ones (`directions = self.compute_directions(dim)`) → guidance fully disabled. | New impl follows thesis Algorithm SGES; flag for legacy parity |
| `functions.py` — `trid` | Uses `+ Σ x_i x_{i-1}` where thesis/classic More–Sugrue Trid has `− Σ x_i x_{i-1}`. | Thesis wins; legacy variant behind flag |
| `functions.py` — `extended_feudenstein_and_roth` | Second term `(5−x_d)x_d − 2` differs from thesis table's `(x_d+1)x_d − 14`. | Thesis wins; legacy variant behind flag |
| `functions.py` — `liarwhd` | Defined twice; second definition silently wins. | Deduplicate; keep the effective one, note in docstring |
| `asebo.py` | Depends on `sklearn.decomposition.PCA`; RL environments use deprecated legacy `gym` API (`env.seed`). | Drop sklearn (use SVD/eigh); port RL envs to `gymnasium` |
| `ashgf.py`, `asgf.py` — state | Mutable class-level `data` dict shared across instances (`sigma_zero` rewritten per run); `σ₀` derived as `‖x₀‖/10` rather than an input. | Dataclass-based immutable config; `σ₀` an explicit input (keep derivation available as option) |

## Roadmap

### Phase 0 — Setup

- uv project scaffold: `pyproject.toml` (runtime: `numpy`, `scipy`, `matplotlib`, `pandas`,
  `gymnasium`, `tqdm`; dev group: `pytest`, `ruff`), lockfile, src layout `src/ashgf/`, ruff config.
- GitHub Actions CI: `ruff check` + `pytest` on every PR.
- Commit `references/papers/` (currently untracked), refresh README.

Branches: `docs/save-plan` (this file + AGENTS.md audit fixes), then `feat/project-scaffold`.

### Phase 1 — Core math library (`feat/core-math`)

- `src/ashgf/functions.py`: benchmark suite. Each function implements its thesis-table formula
  exactly, with batched evaluation `evaluate(points: (n, d)) -> (n,)`. Legacy variants (old `trid`
  sign, old Feudenstein–Roth term) available behind explicit flags for parity studies.
- `src/ashgf/quadrature.py`: Gauss–Hermite nodes/weights plus the DGS directional-derivative
  estimator per eq. (GHApproximationSmothedSection):
  `D̃^M[G_σ(0|x,ξ)] = (√2/(σ√π)) · Σ_m w_m v_m F(x + √2 σ v_m ξ)`, reusing `f(x)` at the center node.
- Local Lipschitz constant estimation per eq. (Lipschitz constants) over the pair set `I`.
- Tests:
  - estimator recovers analytic gradients (sphere, power, generalized Rosenbrock, ...);
  - error scales as `O(σ²)` for smooth non-quadratic functions;
  - FE count exact: `dim·(m−1)+1` evaluations per ASGF/ASHGF iteration.

### Phase 2 — Algorithms (`feat/algorithms-*`)

Dataclass-based implementation, one module per algorithm under `src/ashgf/algorithms/`:

- `RandomSearch` (public name kept as **"GD"**: central-difference random search, M=d directions).
- `SGES`: gradient buffer of size k, historical sampling with probability α, adaptive α per
  eq. (adapt alpha), warmup `T_w ≥ k`.
- `ASEBO`: active subspace via covariance eigen-decomposition, exploration/exploitation mixing,
  renormalized marginals.
- `ASGF`: DGS estimator, first direction pinned to previous gradient, adaptive σ / learning-rate
  subroutine (Algorithm subroutineASGF).
- `ASHGF`: full thesis algorithm (Algorithms ASHGF + ASHGFSubroutine), m=5 GH points per direction,
  reset logic (`r`, `ρ`), warmup `T_w`.

Common infrastructure:

- `RunConfig` dataclass: function, dim, maxiter, seed, x_init, hyperparameters, budget.
- `RunResult` dataclass: iterates, values, best-so-far, total FEs, wall time, per-algorithm extras
  (σ trajectory, α trajectory, L_nabla trajectory, ...).
- One `np.random.Generator` per run derived from the config seed; no global seeding anywhere.
- Exact FE counter wrapping every objective call.

Acceptance: deterministic runs given a seed; unit tests asserting single-step behavior against
hand-computed expectations where tractable.

### Phase 3 — Parity & comparison harness (`feat/comparison-harness`)

- Dedicated legacy environment for the frozen original (separate uv-managed venv, Python ≤ 3.11,
  legacy `gym`), driven by a small adapter that emits the same structured records (values, FEs,
  wall time) as the new package.
- Parity tests: wherever behavior must match (function evaluations, shared RNG streams with aligned
  seeds), assert equality within tolerance.
- First report `comparisons/<date>-old-vs-new-basics/`: correctness deltas, speedups at
  dim ∈ {10, 100, 1000}, FE-accounting validation.

### Phase 4 — Thesis replication (`feat/thesis-replication`)

- Reproduce `numericalexperiments.tex`: dims `{10, 100, 1000}`, the ~28 starred functions in
  `original_python_code/functions.txt`, 10 fixed seeds `[6177, 2832, 7361, 2778, 5416, 9652, 125,
  978, 1487, 4156]`, FE budgets up to 1e8, tolerances τ ∈ {1e-2 … 1e-10}.
- More & Sugrue performance/data profiles for GD, SGES, ASEBO, ASGF, ASHGF — both implementations.
- Report `comparisons/<date>-thesis-replication/report.md` with figures; document any deviations
  between reproduced figures and the thesis originals.

## Key decisions

- **Spec hierarchy**: thesis > original code. For SOTA algorithms, their papers are authoritative
  where the thesis summarizes them loosely.
- **FE accounting**: ASGF/ASHGF ≈ `dim·(m−1)+1` per iteration (center GH node reuses f(x));
  central-difference methods ≈ `2·dim` (+1 if f(x) itself is counted).
- **RL environments** (Pendulum/CartPole): port to `gymnasium`; maximization handled via sign flip
  as in the original; kept out of the fast CI path (slow, stochastic).
- **Naming**: public names follow the thesis tables; "GD" stays (it is the random-search baseline,
  clarified in docstrings).

## Immediate next steps

1. Merge `docs/save-plan` (this file + AGENTS.md audit fixes).
2. `feat/project-scaffold`: pyproject, lockfile, ruff config, CI workflow, smoke test, README → PR → merge.
3. `feat/core-math`: functions + quadrature + tests → PR → merge.
