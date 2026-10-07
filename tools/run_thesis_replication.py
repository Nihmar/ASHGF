"""Thesis replication driver: More & Sugrue profiles over the starred benchmark set.

Protocol (faithful to ``thesis/chapters/numericalexperiments.tex`` and the original
notebook ``original_python_code/notebook_profiles.ipynb``):

* problem set: the 26 starred functions of ``original_python_code/functions.txt``;
* dimensions {10, 100, 1000}; budgets ``mu_L`` in {1e4, ..., 1e8}; tolerances
  ``tau`` in {1e-2, ..., 1e-10};
* every (function, dim, solver) cell is executed once with ``--maxiter`` iterations
  (the notebook ran 10000); trajectories store values and *exact cumulative FEs*,
  so all ``(mu, tau)`` combinations are derived offline from the stored data, with
  budget truncation equivalent to the notebook's iteration model;
* both sides are reproduced: the new package (canonical hyperparameters) and the
  frozen prototype through ``ashgf.parity.run_original`` (its own quirks included).

Data layout written under ``--out``::

    data/manifest.csv                     one row per completed run
    data/new/dim{d}__{func}__{solver}.npz  values + exact cumulative fes
    data/old/dim{d}__{func}__{solver}.npz  same, from the frozen prototype
    tables/fL_{side}_{dim}.csv            best-in-budget value per function/mu
    tables/profile_*_{side}_{dim}_*.csv   full profile curves
    tables/summary_{side}_{dim}.csv       final profile values per (mu, tau, solver)
    figures/{side}/performance_profile_{dim}_{mu}_{tau}.png   (thesis naming)
    figures/{side}/data_profile_{dim}_{mu}_{tau}.png
    report.md                             coverage, key numbers, deviations

Resume-safe: existing run files are skipped. Phases can be run independently:

    uv run python tools/run_thesis_replication.py --phase runs
    uv run python tools/run_thesis_replication.py --phase profiles
    uv run python tools/run_thesis_replication.py --phase report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashgf.algorithms import (  # noqa: E402
    ASEBO,
    ASGF,
    ASHGF,
    GD,
    SGES,
    ASEBOConfig,
    ASGFConfig,
    ASHGFConfig,
    GDConfig,
    SGESConfig,
)
from ashgf.functions import THESIS_STARRED_FUNCTIONS  # noqa: E402
from ashgf.profiles import (  # noqa: E402
    ALPHA_GRID_DATA,
    ALPHA_GRID_PERFORMANCE,
    Trajectory,
    data_profile,
    performance_profile,
    save_profiles_csv,
    solve_cell,
)

SOLVERS = ("gd", "sges", "asebo", "asgf", "ashgf")
DEFAULT_BUDGETS = [1e4, 1e5, 1e6, 1e7, 1e8]
DEFAULT_TAUS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
MANIFEST_COLUMNS = [
    "side",
    "dim",
    "function",
    "solver",
    "n_iterations",
    "total_function_evaluations",
    "wall_time_seconds",
    "status",
    "exact_fes_reconstructed",
    "file",
]


# ---------------------------------------------------------------------------
# Configuration builders
# ---------------------------------------------------------------------------


def _base_kwargs(function: str, dim: int, maxiter: int, seed: int, x_init: np.ndarray):
    return {
        "function": function,
        "dim": dim,
        "maxiter": maxiter,
        "seed": seed,
        "eps": 1e-8,
        "x_init": x_init,
    }


def build_new_run(solver: str, function: str, dim: int, maxiter: int,
                  seed: int, x_init: np.ndarray):
    """Instantiate one canonical-side run for a solver."""
    kw = _base_kwargs(function, dim, maxiter, seed, x_init)
    if solver == "gd":
        return GD(GDConfig(**kw, sigma=1e-4, lr=1e-4))
    if solver == "sges":
        return SGES(SGESConfig(**kw, sigma=1e-4, lr=1e-4))
    if solver == "asebo":
        return ASEBO(ASEBOConfig(**kw, sigma=1e-4, lr=1e-4, lambd=0.1, thresh=1e-4, l_full=50))
    if solver == "asgf":
        return ASGF(ASGFConfig(**kw))
    if solver == "ashgf":
        return ASHGF(ASHGFConfig(**kw))
    raise ValueError(f"unknown solver {solver!r}")


# ---------------------------------------------------------------------------
# Runs phase
# ---------------------------------------------------------------------------


def run_file_path(out: Path, side: str, dim: int, function: str, solver: str) -> Path:
    return out / "data" / side / f"dim{dim}__{function}__{solver}.npz"


def status_of(values: np.ndarray, terminated_normally: bool | None) -> str:
    if not np.isfinite(values).all():
        return "diverged"
    if terminated_normally is False:
        return "aborted"
    return "completed"


def append_manifest(manifest: Path, row: dict) -> None:
    new = not manifest.exists()
    frame = pd.DataFrame([{c: row.get(c, np.nan) for c in MANIFEST_COLUMNS}])
    frame.to_csv(manifest, mode="a", header=new, index=False)


def save_run(
    out: Path,
    side: str,
    dim: int,
    function: str,
    solver: str,
    values: np.ndarray,
    fes: np.ndarray,
    meta: dict,
) -> None:
    path = run_file_path(out, side, dim, function, solver)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, fes=fes, **meta)
    append_manifest(
        out / "data" / "manifest.csv",
        {"side": side, "dim": dim, "function": function, "solver": solver, **meta,
         "file": str(path.relative_to(out))},
    )


def execute_new_cell(out: Path, function: str, dim: int, maxiter: int, seed: int,
                     x_init: np.ndarray, solver: str) -> dict:
    algo = build_new_run(solver, function, dim, maxiter, seed, x_init)
    result = algo.run()
    values = np.asarray(result.values, dtype=np.float64)
    fes = np.asarray(result.fes_per_iterate, dtype=np.int64)
    meta = {
        "n_iterations": result.n_iterations,
        "total_function_evaluations": int(result.total_function_evaluations),
        "wall_time_seconds": float(result.wall_time_seconds),
        "status": status_of(values, True),
    }
    save_run(out, "new", dim, function, solver, values, fes, meta)
    return meta


def execute_old_cell(out: Path, function: str, dim: int, maxiter: int, seed: int,
                     solver: str) -> dict:
    """Run the frozen prototype exactly as the thesis did: no external start
    point, so its internal ``np.random.seed`` + ``randn`` stream is untouched."""
    from ashgf.parity.original_runner import run_original

    rec = run_original(
        algorithm=solver,
        function=function,
        dim=dim,
        it=maxiter,
        seed=seed,
        eps=1e-8,
    )
    values = np.asarray(rec.values, dtype=np.float64)
    if rec.fes_per_iterate is not None:
        fes = np.asarray(rec.fes_per_iterate, dtype=np.int64)
    else:  # anomalous termination: fall back to linear interpolation of the total
        n = len(values)
        fes = np.round(np.linspace(1, rec.total_function_evaluations, n)).astype(np.int64)
    meta = {
        "n_iterations": rec.n_iterations,
        "total_function_evaluations": int(rec.total_function_evaluations),
        "wall_time_seconds": float(rec.wall_time_seconds),
        "status": status_of(values, rec.terminated_normally),
        "exact_fes_reconstructed": rec.fes_per_iterate is not None,
    }
    save_run(out, "old", dim, function, solver, values, fes, meta)
    return meta


def canonical_x0(seed: int, dim: int) -> np.ndarray:
    """Thesis-protocol initial point: legacy MT19937 stream re-seeded per run.

    The prototype calls ``np.random.seed(self.seed)`` and then draws
    ``np.random.randn(dim)`` when no start point is given, so this reproduces
    exactly the vector every thesis run started from (and keeps both sides on
    the same start).
    """
    return np.random.RandomState(seed).randn(dim)


def phase_runs(args: argparse.Namespace) -> None:
    out = Path(args.out)
    functions = list(args.functions or THESIS_STARRED_FUNCTIONS)
    started = time.perf_counter()
    sides = ("new", "old") if args.side == "both" else (args.side,)
    for side in sides:
        dims = args.dims if side == "new" else args.old_dims
        for dim in dims:
            x_init = canonical_x0(args.seed, dim)
            for function in functions:
                for solver in SOLVERS:
                    path = run_file_path(out, side, dim, function, solver)
                    if path.exists():
                        continue
                    t0 = time.perf_counter()
                    try:
                        if side == "new":
                            meta = execute_new_cell(out, function, dim, args.maxiter, args.seed,
                                                    x_init, solver)
                        else:
                            meta = execute_old_cell(out, function, dim, args.maxiter, args.seed,
                                                    solver)
                    except Exception as exc:  # keep going; record failure in manifest
                        meta = {"status": f"error: {type(exc).__name__}: {exc}"}
                        append_manifest(
                            out / "data" / "manifest.csv",
                            {"side": side, "dim": dim, "function": function, "solver": solver,
                             **meta},
                        )
                    elapsed = time.perf_counter() - t0
                    print(f"[{side}] dim={dim} {function} {solver}: {meta.get('status')} "
                          f"in {elapsed:.1f}s ({meta.get('total_function_evaluations', '?')} FEs)",
                          flush=True)
    print(f"runs phase finished in {time.perf_counter() - started:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Profiles phase
# ---------------------------------------------------------------------------

SOLVER_LABELS = {"gd": "GD (ES)", "sges": "SGES", "asebo": "ASEBO", "asgf": "ASGF",
                 "ashgf": "ASHGF"}


def load_trajectories(
    out: Path, side: str, dim: int, functions: list[str]
) -> tuple[dict[str, dict[str, Trajectory]], list[str]]:
    """Load all runs of one (side, dim); returns (per-function trajectories, coverage gaps)."""
    cells: dict[str, dict[str, Trajectory]] = {}
    gaps: list[str] = []
    for function in functions:
        cell: dict[str, Trajectory] = {}
        for solver in SOLVERS:
            path = run_file_path(out, side, dim, function, solver)
            if not path.exists():
                continue
            with np.load(path) as data:
                cell[solver] = Trajectory(
                    solver=solver,
                    function=function,
                    dim=int(dim),
                    values=np.asarray(data["values"], dtype=np.float64),
                    fes=np.asarray(data["fes"], dtype=np.int64),
                )
        if len(cell) < len(SOLVERS):
            missing = [s for s in SOLVERS if s not in cell]
            gaps.append(f"{function}: missing {','.join(missing)}")
        else:
            cells[function] = cell
    return cells, gaps


def _plot_profile(profiles: dict[str, np.ndarray], grid: np.ndarray,
                  path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(6.4, 5.6))
    plt.rcParams.update({"font.size": 14})
    for solver in SOLVERS:
        plt.plot(grid, profiles[solver], label=SOLVER_LABELS[solver])
    plt.title(title)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(ylabel)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()


def compute_side_profiles(
    out: Path, side: str, dim: int, functions: list[str], budgets: list[float], taus: list[float]
) -> list[str]:
    """Compute and store all profiles for one (side, dim). Returns coverage gap notes."""
    cells, gaps = load_trajectories(out, side, dim, functions)
    fig_dir = out / "figures" / side
    tbl_dir = out / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    problem_names = sorted(cells)

    fl_rows: list[dict] = []
    summary_rows: list[dict] = []
    for mu in budgets:
        ts_per_tau: dict[float, dict[str, dict[str, float]]] = {
            tau: {s: {} for s in SOLVERS} for tau in taus
        }
        for function in problem_names:
            f_l, results = solve_cell(cells[function], mu=mu, taus=np.array(taus))
            fl_rows.append({"function": function, "dim": dim, "mu": mu, "f_L": f_l})
            for solver in SOLVERS:
                for tau in taus:
                    ts_per_tau[tau][solver][function] = results[solver][float(tau)]
        for tau in taus:
            perf = performance_profile(ts_per_tau[tau], list(SOLVERS), problem_names)
            dat = data_profile(ts_per_tau[tau], list(SOLVERS), dim)
            tau_str = f"{tau:.0e}"
            _plot_profile(perf, ALPHA_GRID_PERFORMANCE,
                          fig_dir / f"performance_profile_{dim}_{int(mu)}_{tau_str}.png",
                          r"$\tau = $" + tau_str, r"$\rho_s(\alpha)$")
            _plot_profile(dat, ALPHA_GRID_DATA,
                          fig_dir / f"data_profile_{dim}_{int(mu)}_{tau_str}.png",
                          r"$\tau = $" + tau_str, r"$d_s(\alpha)$")
            perf_path = tbl_dir / f"profile_performance_{side}_{dim}_{int(mu)}_{tau_str}.csv"
            dat_path = tbl_dir / f"profile_data_{side}_{dim}_{int(mu)}_{tau_str}.csv"
            save_profiles_csv(str(perf_path), perf, list(SOLVERS), ALPHA_GRID_PERFORMANCE)
            save_profiles_csv(str(dat_path), dat, list(SOLVERS), ALPHA_GRID_DATA)
            for solver in SOLVERS:
                summary_rows.append({
                    "side": side, "dim": dim, "mu": mu, "tau": tau, "solver": solver,
                    "n_problems": len(problem_names),
                    "rho_at_alpha_max": float(perf[solver][-1]),
                    "d_at_alpha_max": float(dat[solver][-1]),
                })
    pd.DataFrame(fl_rows).to_csv(tbl_dir / f"fL_{side}_{dim}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(tbl_dir / f"summary_{side}_{dim}.csv", index=False)
    return gaps


def phase_profiles(args: argparse.Namespace) -> None:
    out = Path(args.out)
    functions = list(args.functions or THESIS_STARRED_FUNCTIONS)
    sides = ("new", "old") if args.side == "both" else (args.side,)
    started = time.perf_counter()
    for side in sides:
        dims = args.dims if side == "new" else args.old_dims
        for dim in dims:
            gaps = compute_side_profiles(out, side, dim, functions, args.budgets, args.taus)
            print(f"[{side}] dim={dim}: profiles over {len(functions)} requested problems",
                  flush=True)
            for gap in gaps:
                print(f"[{side}] dim={dim} coverage gap - {gap}", flush=True)
    print(f"profiles phase finished in {time.perf_counter() - started:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Report phase
# ---------------------------------------------------------------------------


def _flatten_colname(c: tuple) -> str:
    """Flatten a (possibly hierarchical) column label into a plain string."""
    if len(c) == 2 and c[1]:
        return f"{c[0]}/{c[1]}"
    return str(c[0])


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without extra deps."""
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        vals = []
        for c in frame.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _coverage_frame(out: Path, side: str, dims: list[int],
                    functions: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict] = []
    notes: list[str] = []
    for dim in dims:
        expected = len(functions) * len(SOLVERS)
        done = 0
        status_counts: dict[str, int] = {}
        missing: list[str] = []
        for function in functions:
            for solver in SOLVERS:
                path = run_file_path(out, side, dim, function, solver)
                if not path.exists():
                    missing.append(f"{function}/{solver}")
                    continue
                done += 1
                with np.load(path) as data:
                    status = str(data["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
        if missing:
            shown = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
            notes.append(f"{side} dim={dim}: {len(missing)}/{expected} runs missing ({shown})")
        rows.append({"side": side, "dim": dim, "runs_done": done, "runs_expected": expected,
                     **{f"status_{k}": v for k, v in sorted(status_counts.items())}})
    return notes, pd.DataFrame(rows)


REPORT_PROTOCOL_LINES = [
    "- Problem set: the 26 starred functions of `original_python_code/functions.txt`.",
    "- Dimensions: new side {dims_new}; frozen side {dims_old} (dim 1000 is out of scope on the "
    "frozen side: scalar-loop runtime plus known breakage, see below).",
    "- Iterations per cell: up to `{maxiter}` (the thesis notebook ran 10000), early stop on "
    "`||x_{{i+1}} - x_i|| < 1e-8`.",
    "- Budgets mu_L: {budgets}; tolerances tau: {taus}.",
    "- Shared start point per dimension: the thesis protocol's own draw, "
    "`np.random.seed({seed}); np.random.randn(d)` (legacy MT19937): used directly by the new "
    "side and reproduced bit-for-bit inside the frozen side (which keeps its own "
    "re-seeding behaviour).",
    "- Hyperparameters: lr = sigma = 1e-4 for GD/SGES/ASEBO; canonical defaults for ASGF/ASHGF "
    "(thesis formulas); frozen side uses its own constructor defaults via "
    "`ashgf.parity.run_original`.",
    "- Performance measures use exact cumulative FE counts stored per iterate (equivalent to the "
    "notebook's idealized 2d+1 / 4d+1 model up to one iterate, since every algorithm has constant "
    "per-iteration cost).",
]

REPORT_DEVIATION_LINES = [
    "- Frozen SGES aborts at its first guided iteration under NumPy >= 2 (`int()` on a size-1 "
    "array inside `compute_directions_sges`); affected cells appear as short trajectories or "
    "unsolved problems.",
    "- Frozen ASHF aborts analogously at its first historical iteration.",
    "- Frozen ASEBO samples 100 directions during warm-up regardless of dimension (~8x FE "
    "inflation vs canonical at dim 10) and can declare silent false convergence after divergence.",
    "- Diverged runs (non-finite values) count as unsolved problems, matching how the convergence "
    "test would treat them.",
    "- The frozen side was not run at dim 1000 (scalar loops take hours per function; its guidance "
    "machinery is unexecutable anyway). New-side dim-1000 figures are the reference for "
    "high-dimensional behaviour.",
]


def phase_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    functions = list(args.functions or THESIS_STARRED_FUNCTIONS)
    sides = ("new", "old") if args.side == "both" else (args.side,)
    dims_by_side = {"new": args.dims, "old": args.old_dims}

    notes: list[str] = []
    cov_frames: list[pd.DataFrame] = []
    for side in sides:
        side_notes, frame = _coverage_frame(out, side, dims_by_side[side], functions)
        notes.extend(side_notes)
        cov_frames.append(frame)
    coverage = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()

    summary_frames = []
    for side in sides:
        for dim in dims_by_side[side]:
            path = out / "tables" / f"summary_{side}_{dim}.csv"
            if path.exists():
                summary_frames.append(pd.read_csv(path))
    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else None

    lines: list[str] = []
    add = lines.append
    add("# Thesis replication — More & Sugrue profiles")
    add("")
    add(f"Generated by `tools/run_thesis_replication.py` on {time.strftime('%Y-%m-%d %H:%M')}.")
    add("")
    add("## Protocol")
    add("")
    for template in REPORT_PROTOCOL_LINES:
        add(template.format(dims_new=dims_by_side["new"], dims_old=dims_by_side["old"],
                             maxiter=args.maxiter, budgets=[int(b) for b in args.budgets],
                             taus=[float(t) for t in args.taus], seed=args.seed))
    add("")
    add("## Coverage")
    add("")
    if not coverage.empty:
        add(_md_table(coverage))
        add("")
    if notes:
        add("Coverage gaps:")
        for note in notes:
            add(f"- {note}")
        add("")
    add("## Key results")
    add("")
    add("Final profile values: rho_s at alpha=3 (fraction of problems solved no more than 3x "
        "slower than the best solver) and d_s at alpha=950 (fraction solved within "
        "950*(dim+1) FEs).")
    add("")
    if summaries is not None and not summaries.empty:
        for dim in sorted(int(v) for v in summaries["dim"].unique()):
            for mu in sorted(float(v) for v in summaries["mu"].unique()):
                sel = summaries[(summaries["dim"] == dim) & (summaries["mu"] == mu)]
                if sel.empty:
                    continue
                piv = sel.pivot_table(
                    index="tau",
                    columns=["side", "solver"],
                    values="rho_at_alpha_max",
                    aggfunc="first",
                )
                add(f"### dim = {dim}, mu_L = {int(mu)} — rho_s(alpha=3)")
                add("")
                flat = piv.reset_index()
                flat.columns = [_flatten_colname(c) for c in flat.columns]
                rows = []
                for _, r in flat.iterrows():
                    entry = {"tau": float(r["tau"])}
                    for col in flat.columns:
                        if col == "tau":
                            continue
                        entry[col] = round(float(r[col]), 3)
                    rows.append(entry)
                add(_md_table(pd.DataFrame(rows)))
                add("")
                add("")
    else:
        add("No profile data yet — run the `profiles` phase first.")
        add("")
    add("## Deviations and caveats")
    add("")
    lines.extend(REPORT_DEVIATION_LINES)
    add("")
    add("## Figures")
    add("")
    fig_dir = out / "figures"
    if fig_dir.exists():
        for png in sorted(fig_dir.rglob("*.png")):
            add(f"- `{png.relative_to(out)}`")
    else:
        add("No figures generated yet.")

    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["runs", "profiles", "report", "all"], default="all")
    parser.add_argument("--side", choices=["new", "old", "both"], default="both")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--old-dims", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--maxiter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS)
    parser.add_argument("--functions", nargs="*", default=None)
    parser.add_argument("--out",
                        default=str(REPO_ROOT / "comparisons" / "2026-10-07-thesis-replication"))
    args = parser.parse_args()
    if args.phase in ("runs", "all"):
        phase_runs(args)
    if args.phase in ("profiles", "all"):
        phase_profiles(args)
    if args.phase in ("report", "all"):
        phase_report(args)


if __name__ == "__main__":
    main()
