"""More & Sugrue performance/data profile machinery.

Implements the definitions used in the thesis chapter "Numerical Results":

- **Convergence test.** With ``F_L`` the best value achieved by any solver within the
  evaluation budget ``mu_L``, a solver satisfies the test at tolerance ``tau`` when
  ``F(x_k) <= F_L + tau * (F(x_0) - F_L)``.
- **Performance measure** ``t_{p,s}``: the number of function evaluations needed to
  satisfy the test (``inf`` when the test is never satisfied within budget).
- **Performance profile** ``rho_s(alpha)``: fraction of problems with
  ``t_{p,s} / min_{s'} t_{p,s'} <= alpha``.
- **Data profile** ``d_s(alpha)``: fraction of problems with ``t_{p,s} <= alpha*(n_p+1)``,
  where ``n_p`` is the problem dimension.

Trajectories carry exact cumulative FE counts (one entry per iterate, aligned with the
value sequence), so ``t_{p,s}`` is computed from real accounting rather than an assumed
per-iteration cost. Budget truncation keeps every iterate whose cumulative count does
not exceed ``mu_L`` — equivalent to the thesis notebook's ``iterations(mu, dim, algo)``
truncation up to one iterate, since all algorithms here have constant per-iteration
cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Alpha grid of the thesis performance profiles: 1.0, 1.2, ..., 3.0
ALPHA_GRID_PERFORMANCE: NDArray[np.float64] = np.array([i / 5.0 for i in range(5, 16)])

#: Alpha grid of the thesis data profiles: 0, 50, ..., 950
ALPHA_GRID_DATA: NDArray[np.float64] = np.arange(0, 1000, 50, dtype=np.float64)


@dataclass(frozen=True)
class Trajectory:
    """One completed run of one solver on one problem.

    Attributes
    ----------
    solver:
        Solver label (e.g. ``"gd"``, ``"ashgf"``).
    function:
        Benchmark name.
    dim:
        Problem dimension.
    values:
        Objective values at the iterates, including ``F(x_0)``.
    fes:
        Cumulative function-evaluation count aligned with ``values``; entry ``i`` is the
        total number of objective evaluations performed up to and including ``F(x_i)``.
    """

    solver: str
    function: str
    dim: int
    values: NDArray[np.float64]
    fes: NDArray[np.int64]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.fes):
            raise ValueError("values and fes must be aligned")
        if self.fes[0] < 1 or not np.all(np.diff(self.fes) >= 0):
            raise ValueError("fes must start at >= 1 and be non-decreasing")

    @property
    def initial_value(self) -> float:
        return float(self.values[0])

    def truncate_by_budget(self, mu: float) -> Trajectory:
        """Keep only the iterates affordable within ``mu`` total evaluations."""
        mask = self.fes <= mu
        return Trajectory(
            solver=self.solver,
            function=self.function,
            dim=self.dim,
            values=self.values[mask],
            fes=self.fes[mask],
        )


def convergence_fe(traj: Trajectory, f_l: float, tau: float) -> float:
    """Evaluations needed to satisfy the convergence test, or ``inf``.

    Parameters
    ----------
    traj:
        The (already budget-truncated) trajectory.
    f_l:
        Best value achieved by any solver within the same budget.
    tau:
        Tolerance parameter in (0, 1).
    """
    if not np.isfinite(f_l):
        return np.inf
    threshold = f_l + tau * (traj.initial_value - f_l)
    ok = traj.values <= threshold
    if not ok.any():
        return np.inf
    return float(traj.fes[int(np.argmax(ok))])


def solve_cell(
    trajectories: dict[str, Trajectory], mu: float, taus: NDArray[np.float64]
) -> tuple[float, dict[str, dict[float, float]]]:
    """Solve one problem cell (all solvers on one function) for a given budget.

    Returns
    -------
    (f_l, results)
        ``f_l`` is the best value over all solvers within budget; ``results[s][tau]``
        is the performance measure ``t_{p,s}`` (possibly ``inf``).
    """
    truncated = {s: trj.truncate_by_budget(mu) for s, trj in trajectories.items()}
    finite_mins = [
        float(np.min(trj.values[np.isfinite(trj.values)]))
        for trj in truncated.values()
        if np.isfinite(trj.values).any()
    ]
    f_l = min(finite_mins) if finite_mins else np.inf
    results: dict[str, dict[float, float]] = {}
    for solver, trj in truncated.items():
        results[solver] = {float(tau): convergence_fe(trj, f_l, float(tau)) for tau in taus}
    return f_l, results


def _ratio_matrix(
    t_by_solver: dict[str, dict[str, float]], solvers: list[str], functions: list[str]
) -> dict[str, dict[str, float | np.nan]]:
    """Per-problem ratios t_{p,s} / min_{s'} t_{p,s'}, NaN where undefined."""
    ratios: dict[str, dict[str, float | np.nan]] = {s: {} for s in solvers}
    for p in functions:
        finite = [
            t_by_solver[s][p]
            for s in solvers
            if p in t_by_solver.get(s, {}) and np.isfinite(t_by_solver[s][p])
        ]
        if not finite:
            continue
        m = min(finite)
        for s in solvers:
            t = t_by_solver.get(s, {}).get(p)
            ratios[s][p] = (t / m) if (t is not None and np.isfinite(t)) else np.nan
    return ratios


def performance_profile(
    t_by_solver: dict[str, dict[str, float]],
    solvers: list[str],
    functions: list[str],
    alpha_grid: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """rho_s(alpha) for every solver, evaluated on ``alpha_grid``."""
    grid = ALPHA_GRID_PERFORMANCE if alpha_grid is None else alpha_grid
    ratios = _ratio_matrix(t_by_solver, solvers, functions)
    n_problems = max(len(functions), 1)
    profiles: dict[str, NDArray[np.float64]] = {}
    for s in solvers:
        vals = np.array([ratios[s].get(p, np.nan) for p in functions])
        counts = np.array([(vals <= a).sum() for a in grid], dtype=float)
        # Problems with no finite solver are counted as unsolved for everyone.
        profiles[s] = counts / n_problems
    return profiles


def data_profile(
    t_by_solver: dict[str, dict[str, float]],
    solvers: list[str],
    dim: int,
    alpha_grid: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """d_s(alpha) for every solver, evaluated on ``alpha_grid``."""
    grid = ALPHA_GRID_DATA if alpha_grid is None else alpha_grid
    functions = sorted({p for s in solvers for p in t_by_solver.get(s, {})})
    n_problems = max(len(functions), 1)
    profiles: dict[str, NDArray[np.float64]] = {}
    for s in solvers:
        ts = np.array(
            [t_by_solver.get(s, {}).get(p, np.inf) for p in functions], dtype=np.float64
        )
        counts = np.array([(ts <= a * (dim + 1)).sum() for a in grid], dtype=float)
        profiles[s] = counts / n_problems
    return profiles


def ideal_iterations(mu: float, dim: int, algorithm: str) -> int:
    """The thesis notebook's iteration model for a budget of ``mu`` evaluations.

    Central-difference solvers cost ``2*dim + 1`` evaluations per iterate, Gauss-Hermite
    solvers with m=5 cost ``4*dim + 1``.
    """
    if algorithm in ("asgf", "ashgf"):
        return int(mu / (1 + dim * 4))
    return int(mu / (1 + dim * 2))


def save_profiles_csv(
    path: str,
    profile: dict[str, NDArray[np.float64]],
    solvers: list[str],
    grid: NDArray[np.float64],
) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["alpha"] + solvers)
        for i, a in enumerate(grid):
            writer.writerow([f"{a:g}"] + [f"{profile[s][i]:.6f}" for s in solvers])
