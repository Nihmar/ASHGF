# Project Plan — ASHGF Reimplementation

Status: active. Last updated: 2026-10-08.

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
| `sges.py`, `ashgf.py` — `compute_directions_sges` | Calls `int(np.random.choice([0, 1], size=1, ...))`, which raises under NumPy >= 2 (`TypeError: only 0-dimensional arrays can be converted to Python scalars`). Under modern numpy the frozen SGES aborts at its first guided iteration (i = t) and the frozen ASHF at its first historical iteration (i = t); the signature features of both algorithms are therefore unexecutable without patching the frozen code. | Documented in `comparisons/2026-10-07-old-vs-new-basics/`; not patched (frozen) |
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

**Status (2026-10-07):** all five algorithms implemented and tested (determinism, exact FE
accounting, mathematical invariants); awaiting parity harness in Phase 3.

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

**Status (2026-10-07):** done.

- No separate legacy venv: the frozen modules run unmodified inside the main environment through
  `ashgf.parity.run_original`, which (a) installs a minimal `gym` stand-in when legacy gym is
  absent (benchmark functions never touch it), (b) counts objective calls by wrapping
  `Function.evaluate`, (c) reconstructs the full iterate sequence (the prototype's returned list
  always drops the final iterate) and flags runs aborted by the prototype's swallowed exceptions.
  Deviation from the original plan (dedicated legacy env): unnecessary because no deprecated numpy
  APIs appear at import time; revisit if RL environments enter the comparison scope.
- Parity tests (`tests/test_parity.py`): exact FE accounting per analytic formula, determinism,
  bit-close GD trajectory parity with aligned MT19937 streams (`BaseRunConfig.rng` hook),
  quantified ASEBO warm-up cost deviation, legacy-rule flag sanity checks.
- First report `comparisons/2026-10-07-old-vs-new-basics/` generated by
  `tools/run_basics_comparison.py`: grid {sphere, generalized_rosenbrock} × dims {10, 50} ×
  seeds {2003, 7}, maxiter 400; data.csv + figures + report. Key findings:
  - frozen SGES/ASHF abort at i = t under NumPy ≥ 2 (guidance machinery unexecutable);
  - silent false convergence observed in frozen ASEBO on Rosenbrock (divergence to O(1e78),
    perturbations round away, zero gradient, eps declares convergence after 5 iterations);
  - frozen ASGF diverged on Rosenbrock d=50 until inf/NaN made `scipy.linalg.orth` raise a
    swallowed ValueError;
  - frozen ASEBO spends ~8x more FEs than canonical at dim=10 (fixed 100-direction warm-up);
  - throughput gains up to ~8.5x per unit work (GD, dim 50).

Planned extension: dims {10, 100, 1000} move to Phase 4 (thesis replication grid).

### Phase 4 — Thesis replication (`feat/thesis-replication`)

**Status (2026-10-08):** protocol fully reconstructed from the frozen code
(`original_python_code/profiles.py` + `notebook_profiles.ipynb`); driver implemented at
`tools/run_thesis_replication.py` and extended with `--all-functions` (entire canonical
catalog, 78 functions) and `--workers` (thread-parallel runs phase); the full-catalog grid
is launched in stages by `tools/launch_full_grid.sh` into
`comparisons/<date>-full-grid/`.

Reconstructed protocol (supersedes earlier guess of 10 fixed seeds):

* single trajectory per (function, dim, solver) cell, up to `it = 10000` iterations,
  early stop `||x_{i+1} - x_i|| < 1e-8`;
* every original constructor defaults to `seed = 2003`; each run re-seeds the global
  legacy MT19937 stream, so all runs of a given dimension share the start point
  `np.random.seed(2003); np.random.randn(dim)`;
* constructor hyperparameters: GD/SGES/ASEBO `(lr=1e-4, sigma=1e-4)`, ASGF/ASHHF class
  defaults; dims `{10, 100, 1000}`; budgets `mu_L ∈ {1e4 … 1e8}`; tolerances
  `tau ∈ {1e-2 … 1e-10}`; alpha grids: performance `[1.0, 1.2, …, 3.0]`, data
  `[0, 50, …, 950]`.
* The notebook derives profiles offline from stored descent curves with an idealized
  FE model: `1 + 2d` FEs/iteration for GD/SGES/ASEBO, `1 + 4d` for ASGF/ASHHF
  (m = 5 GH nodes, center node reused). Our driver stores *exact* cumulative FEs per
  iterate instead, which is equivalent for canonical solvers (constant per-iteration
  cost matching the model) and strictly more accurate where the prototype deviates
  (ASEBO warm-up).

Driver design (data layout under `comparisons/<date>-thesis-replication/`):

* one `.npz` per run in flat directories `data/{new|old}/dim{d}__{func}__{solver}.npz`
  (values + exact cumulative fes + run metadata) plus a single append-only
  `data/manifest.csv` index — compact, resume-safe, tidy overview;
* three independent phases (`runs`, `profiles`, `report`) so expensive executions are
  done once and profile analysis can be re-run freely; figures use the thesis' exact
  naming `performance_profile_{dim}_{mu}_{tau}.png` / `data_profile_…`;
* frozen side reproduces the thesis exactly (no external start point, internal RNG
  stream untouched); new side starts from the same MT19937 vector with canonical
  PCG64 direction streams; frozen side limited to dims ≤ 100 (scalar loops; guidance
  machinery unexecutable on modern NumPy anyway).

Scope decision (2026-10-08): per explicit request the grid covers the **entire canonical
function catalog**, not just the 26 starred functions. Staged launch:

* Stage A — all 78 functions x dims {10, 100} x both sides (~1560 cells);
* Stage B — 26 starred functions x dim 1000 x new side;
* non-starred dim-1000 cells deferred until the ASEBO eigenstep optimization lands
  (two full `eigh(1000)` per iteration make ASEBO dim-1000 cells ~1 s/iter today,
  ~2.8 h/cell at max budget — the long pole of the grid); they will be added later with
  `--phase runs --side new --dims 1000 --all-functions` (resume-safe).

Measured cell costs used for scheduling: frozen side dim 100 ≈ 0.1–47 s/cell (SGES aborts
instantly under NumPy ≥ 2; ASGF/GD grind to the full budget when diverging); new side
dim 1000 ≈ 0.25–0.35 s/iter for ASGF/ASHHF worst case. Expected wall time with 3 workers
on the shared 12-core box: Stage A ≈ 1–2 days, Stage B ≈ 1–2 days.

Remaining work: monitor stage logs and manifest statuses, add the deferred dim-1000
non-starred cells after the ASEBO perf fix, review coverage gaps, write final report
comparing reproduced figures against the thesis originals.

### Phase 5 — GPU acceleration via C + HIP/ROCm (planned, not started)

**Motivation (measured 2026-10-08, CPU):** per-iteration costs that dominate high-dim runs:

| Operation | dim 100 | dim 1000 |
|---|---|---|
| `gram_schmidt_complete` (sequential Python MGS) | ~6 ms | ~805 ms |
| batched projection + reduced QR completion | ~16 ms | ~279 ms |
| blocked (GEMM-batched) MGS completion | ~7 ms | ~163 ms |
| batched objective eval throughput (sphere / Ackley) | — | 471 / 47 M points·s⁻¹ |

At dim 1000 a full-length ASGF/ASHGF cell (10 000 iterations) spends most of its wall time
in basis completion and large dense linear algebra on CPU. The plan is to rewrite the hot
numerical core in **C/C++ with HIP kernels (AMD ROCm)**, delegating all matrix operations
to the GPU while keeping the Python package as the orchestration layer and the single source
of truth for FE accounting, configs, and result records.

Staged outline:

1. **Hotspot inventory & kernel spec.** Final profile-driven list of GPU-bound kernels:
   batched benchmark evaluations (elementwise ops + reductions); central-difference gradient
   assembly (`diff @ directions`, GEMV/GEMM); DGS estimator (batched evals + weighted sums);
   orthonormal completion (blocked Gram–Schmidt / `dgeqrf` via rocBLAS); subspace bases
   (`gesvd`/`syevd` via rocSOLVER for SGES/ASEBO/ASHGF history).
2. **Architecture.** Thin C++ core exposing device arrays through pybind11/nanobind bindings;
   existing dataclass API (`RunConfig`, `RunResult`) unchanged; explicit device selection with
   the current numpy path kept as the always-available CPU reference (auto-fallback when no
   ROCm device is present). One `curand`/host-seeded stream per run, preserving the
   "explicit generator, no global seeding" rule.
3. **Correctness gate.** Bit-close parity tests vs the CPU implementation per kernel
   (tolerance-scaled), then trajectory-level agreement within solver tolerance before any
   comparison numbers are taken from the GPU path.
4. **Acceptance.** Orders-of-magnitude speedup targets at dim ≥ 1000 measured in
   `comparisons/`; More & Sugrue profiles identical (same FEs, same trajectories modulo
   floating-point tolerance).

Risks / open questions: ROCm toolchain availability on target hardware; development cost of
maintaining dual CPU/GPU paths (mitigated by keeping the CPU path canonical); memory-bandwidth
limits for many small vectors (direction matrices are `(d, d)` — fine at d ≤ 1000); whether to
port only the new-side algorithms (frozen prototype stays CPU-only by definition).

Prerequisite: finish Phase 4 first, so baseline CPU timings and thesis-reproduced figures exist
for the speedup claims.

## Key decisions

- **Spec hierarchy**: thesis > original code. For SOTA algorithms, their papers are authoritative
  where the thesis summarizes them loosely.
- **FE accounting**: ASGF/ASHGF ≈ `dim·(m−1)+1` per iteration (center GH node reuses f(x));
  central-difference methods ≈ `2·dim` (+1 if f(x) itself is counted).
- **RL environments** (Pendulum/CartPole): port to `gymnasium`; maximization handled via sign flip
  as in the original; kept out of the fast CI path (slow, stochastic).
- **Naming**: public names follow the thesis tables; "GD" stays (it is the random-search baseline,
  clarified in docstrings).
- **GPU strategy (Phase 5)**: C/C++ + HIP core behind the existing Python API; the numpy CPU
  path remains the reference implementation and default fallback; GPU results must pass
  kernel-level parity tests before use.

## Immediate next steps

1. ~~Merge `feat/comparison-harness`~~ — done (PR #5).
2. ~~Replication driver + protocol reconstruction~~ — done (PR #6); blocked Gram–Schmidt
   completion merged after A/B validation proving unchanged results (PR #7);
   full-catalog parallel launch tooling merged (PR #8).
3. Monitor the running full-catalog grid (`comparisons/2026-10-08-full-grid/`, staged A → B →
   profiles → report; per-stage logs in its `logs/` dir); review coverage gaps and manifest
   statuses as stages complete.
4. `perf/asebo-eigenstep`: replace ASEBO's two full `eigh(d)` per iteration with a provably
   equivalent reduced spectrum computation (the λ-decayed covariance is numerically low-rank;
   tail eigenvalues beyond ~30 recent gradients are below double precision); validate with an
   A/B study like `tools/ab_gs_blocking.py`; then top up the deferred non-starred dim-1000 cells
   with `--phase runs --side new --dims 1000 --all-functions` (resume-safe).
5. Phase 5 planning: once Phase 4 baselines exist, scope the C + HIP/ROCm rewrite per the
   staged outline above (kernel spec → architecture → correctness gate → acceptance).
