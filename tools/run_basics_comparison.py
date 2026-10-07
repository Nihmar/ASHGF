"""Generate the old-vs-new basics comparison bundle under ``comparisons/``.

Runs every algorithm on both sides (the frozen prototype in
``original_python_code/`` and the new implementation) over a grid of
benchmark functions, dimensions and seeds, then writes:

* ``data.csv``  — one row per (algorithm, side, case)
* ``figures/<algorithm>.png`` — convergence curves (log scale)
* ``report.md`` — methodology, tables and documented deviations

Usage::

    uv run python tools/run_basics_comparison.py [--maxiter N] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ashgf.algorithms.asebo import ASEBO, ASEBOConfig  # noqa: E402
from ashgf.algorithms.asgf import ASGF, ASGFConfig  # noqa: E402
from ashgf.algorithms.ashgf import ASHGF, ASHGFConfig  # noqa: E402
from ashgf.algorithms.random_search import GD, GDConfig  # noqa: E402
from ashgf.algorithms.sges import SGES, SGESConfig  # noqa: E402
from ashgf.parity import run_original  # noqa: E402

FUNCTIONS = ["sphere", "generalized_rosenbrock"]
DIMS = [10, 50]
SEEDS = [2003, 7]
ALGORITHMS = ("gd", "sges", "asebo", "asgf", "ashgf")


def build_new_runner(algorithm: str, base: dict):
    """Instantiate a new-side algorithm with thesis-default hyperparameters."""
    if algorithm == "gd":
        return GD(GDConfig(**base, sigma=1e-4, lr=1e-4))
    if algorithm == "sges":
        return SGES(SGESConfig(**base, sigma=1e-4, lr=1e-4, k=50, alpha=0.5))
    if algorithm == "asebo":
        return ASEBO(ASEBOConfig(**base, sigma=1e-4, lr=1e-4, lambd=0.1, l_full=50))
    if algorithm == "asgf":
        return ASGF(ASGFConfig(**base, m=5))
    return ASHGF(ASHGFConfig(**base, m=5, k=50, t_warmup=50, alpha=0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y-%m-%d")
    out_dir = args.out or REPO_ROOT / "comparisons" / f"{stamp}-old-vs-new-basics"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    trajectories: dict[tuple[str, str], dict] = {}  # (algo, func) -> {(dim, seed, side): values}

    for algo in ALGORITHMS:
        for func in FUNCTIONS:
            for dim in DIMS:
                for seed in SEEDS:
                    base = {"function": func, "dim": dim, "maxiter": args.maxiter, "seed": seed}

                    t0 = time.perf_counter()
                    result = build_new_runner(algo, base).run()
                    wall_new = time.perf_counter() - t0
                    rows.append({
                        "algorithm": algo,
                        "side": "new",
                        "function": func,
                        "dim": dim,
                        "seed": seed,
                        "n_iterations": result.n_iterations,
                        "best_value": result.best_value,
                        "total_function_evaluations": result.total_function_evaluations,
                        "wall_time_seconds": wall_new,
                        "terminated_normally": True,
                    })
                    trajectories.setdefault((algo, func), {})[(dim, seed, "new")] = result.values

                    rec = run_original(algo, function=func, dim=dim, it=args.maxiter, seed=seed)
                    rows.append({
                        "algorithm": algo,
                        "side": "old",
                        "function": func,
                        "dim": dim,
                        "seed": seed,
                        "n_iterations": rec.n_iterations,
                        "best_value": rec.best_value,
                        "total_function_evaluations": rec.total_function_evaluations,
                        "wall_time_seconds": rec.wall_time_seconds,
                        "terminated_normally": rec.terminated_normally,
                    })
                    trajectories[(algo, func)][(dim, seed, "old")] = np.asarray(rec.values)
                    msg = (f"{algo:6s} {func:22s} d={dim:2d} seed={seed}: "
                           f"new best={result.best_value:.6g}"
                           f" ({result.total_function_evaluations} FE, {wall_new * 1e3:.0f} ms)")
                    msg += (f" | old best={rec.best_value:.6g}"
                            f" ({rec.total_function_evaluations} FE, "
                            f"{rec.wall_time_seconds * 1e3:.0f} ms)")
                    print(msg)

    fieldnames = [
        "algorithm", "side", "function", "dim", "seed", "n_iterations",
        "best_value", "total_function_evaluations", "wall_time_seconds",
        "terminated_normally",
    ]
    with open(out_dir / "data.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for (algo, _func), data in trajectories.items():
        fig, axes = plt.subplots(len(FUNCTIONS), len(DIMS), figsize=(9, 6.5))
        for r, fr in enumerate(FUNCTIONS):
            for c, d in enumerate(DIMS):
                ax = axes[r, c]
                for seed in SEEDS:
                    for side, style in (("old", "--"), ("new", "-")):
                        vals = data.get((d, seed, side))
                        if vals is None:
                            continue
                        best_so_far = np.minimum.accumulate(vals)
                        ax.plot(np.arange(len(best_so_far)), best_so_far, style, lw=1.2,
                                label=f"{side} s{seed}")
                ax.set_title(f"{fr}, dim={d}", fontsize=9)
                ax.set_yscale("log")
                ax.grid(alpha=0.3)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"{algo}: frozen prototype (-- dashed) vs new implementation (- solid)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figures_dir / f"{algo}.png", dpi=140)
        plt.close(fig)

    report = render_report(args.maxiter, rows, trajectories)
    (out_dir / "report.md").write_text(report)
    print(f"\nWrote bundle to {out_dir}")


def _ms_per_1k(row_sel: list[dict]) -> float:
    wall = sum(r["wall_time_seconds"] for r in row_sel)
    fes = sum(r["total_function_evaluations"] for r in row_sel)
    return 1e6 * wall / fes if fes else float("nan")


def render_report(maxiter: int, rows: list[dict],
                  trajectories: dict[tuple[str, str], dict]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Old vs New — basics comparison")
    add("")
    grid = f"`{'`, `'.join(FUNCTIONS)}`"
    add(f"Grid: functions {grid}, dims `{DIMS}`, seeds `{SEEDS}`, "
        f"`maxiter={maxiter}` on both sides.")
    add("")
    add("## Methodology")
    add("")
    add("- **Old side**: the unmodified modules in `original_python_code/` are driven by")
    add("  `ashgf.parity.run_original`, which counts objective calls by wrapping")
    add("  `Function.evaluate` and reconstructs the full iterate sequence (the prototype's")
    add("  returned list always drops the final iterate).")
    add("- **New side**: `src/ashgf` algorithms with thesis-default hyperparameters.")
    add("- Random streams differ between the two implementations (legacy MT19937 via")
    add("  `np.random.seed` vs PCG64 generators), so same-seed curves are a qualitative")
    add("  convergence comparison, not a trajectory parity check. Exact-stream parity for GD")
    add("  is covered separately in `tests/test_parity.py`.")
    add("")
    add("## Results")
    add("")
    header = ("| algorithm | function | dim | seed | side | iterations "
              "| best value | FEs | wall time (ms) | terminated |")
    add(header)
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        term = "yes" if row["terminated_normally"] else "**no**"
        add(f"| {row['algorithm']} | {row['function']} | {row['dim']} | {row['seed']}"
            f" | {row['side']} | {row['n_iterations']} | {row['best_value']:.6g}"
            f" | {row['total_function_evaluations']}"
            f" | {row['wall_time_seconds'] * 1e3:.0f} | {term} |")
    add("")
    add("Wall-time totals mix early terminations of the frozen prototype, so the fair")
    add("throughput comparison is cost per unit work:")
    add("")
    add("| algorithm | function | dim | old ms / 1k FEs | new ms / 1k FEs"
        " | throughput gain (old/new) |")
    add("|---|---|---|---|---|---|")
    for algo in ALGORITHMS:
        for func in FUNCTIONS:
            for dim in DIMS:
                old_sel = [r for r in rows if r["algorithm"] == algo and r["function"] == func
                           and r["dim"] == dim and r["side"] == "old"]
                new_sel = [r for r in rows if r["algorithm"] == algo and r["function"] == func
                           and r["dim"] == dim and r["side"] == "new"]
                o, n = _ms_per_1k(old_sel), _ms_per_1k(new_sel)
                add(f"| {algo} | {func} | {dim} | {o:.1f} | {n:.1f} | {o / n:.1f}x |")
    add("")
    add("## Documented deviations observed in this run")
    add("")
    sges_crashed = [r for r in rows
                    if r["side"] == "old" and r["algorithm"] == "sges"
                    and not r["terminated_normally"]]
    ashf_crashed = [r for r in rows
                    if r["side"] == "old" and r["algorithm"] == "ashgf"
                    and not r["terminated_normally"]]
    other_crashed = [r for r in rows
                     if r["side"] == "old" and r["algorithm"] not in ("sges", "ashgf")
                     and not r["terminated_normally"]]
    if sges_crashed or ashf_crashed:
        where = []
        if sges_crashed:
            where.append(f"SGES ({len(sges_crashed)} runs)")
        if ashf_crashed:
            where.append(f"ASHF ({len(ashf_crashed)} Rosenbrock runs)")
        add("- **NumPy >= 2 incompatibility in the guidance machinery.** The helper"
            " `compute_directions_sges` (shared by `sges.py` and `ashgf.py`) calls"
            " `int(np.random.choice(...))`, which raises under NumPy >= 2. The frozen"
            f" {' and '.join(where)} therefore abort at their first guided/historical"
            " iteration (i = t); on sphere they simply converge during the plain warm-up"
            " phase instead.")
        add("  Even under the legacy environment where that call succeeded, freshly drawn"
            " random directions overwrite the guided ones inside `grad_estimator`, so"
            " self-guidance never takes effect; moreover SGES re-seeds the global RNG at"
            " every estimator call, so its warm-up iterations reuse the identical direction"
            " matrix.")
    if other_crashed:
        lst = ", ".join(sorted({
            f'{r["algorithm"]} ({r["function"]}, d={r["dim"]}, seed={r["seed"]})'
            for r in other_crashed
        }))
        add("- **Silent failures.** Bare `except Exception` handlers print 'Something has"
            " gone wrong!' and stop without surfacing the error."
            f" Observed here: {lst}. In the ASGF case the trajectory diverged on"
            " Rosenbrock until inf/NaN values made `scipy.linalg.orth` raise a ValueError"
            " that was swallowed.")
    suspect_early = []
    for r in rows:
        if r["side"] != "old" or not r["terminated_normally"]:
            continue
        if r["n_iterations"] >= maxiter // 5:
            continue
        data = trajectories.get((r["algorithm"], r["function"]), {})
        vals = data.get((r["dim"], r["seed"], "old"))
        # False convergence: never beat the initial value despite early stopping.
        if vals is not None and len(vals) > 0 and r["best_value"] >= float(vals[0]):
            suspect_early.append(r)
    if suspect_early:
        lst = ", ".join(sorted({
            f'{r["algorithm"]} ({r["function"]}, d={r["dim"]}, seed={r["seed"]})'
            for r in suspect_early
        }))
        add(f"- **Silent false convergence.** {lst}: the trajectory diverged until values"
            " reached O(1e78); at that scale the +/- sigma*xi perturbations round away"
            " (x + sigma*xi == x - sigma*xi exactly), the estimated gradient becomes"
            " exactly zero, and the eps criterion declares convergence after a handful of"
            " iterations with best value equal to the initial one — no error is reported.")
    asebo_old_fes = [r["total_function_evaluations"] for r in rows
                     if r["algorithm"] == "asebo" and r["side"] == "old" and r["dim"] == 10
                     and r["terminated_normally"]]
    asebo_new_fes = [r["total_function_evaluations"] for r in rows
                     if r["algorithm"] == "asebo" and r["side"] == "new" and r["dim"] == 10]
    if asebo_old_fes and asebo_new_fes:
        ratio = float(np.mean(asebo_old_fes)) / max(float(np.mean(asebo_new_fes)), 1.0)
        add("- Frozen **ASEBO warm-up samples 100 directions regardless of dimension**"
            " (thesis: n_t = d during the l warm-up iterations); at dim=10 it spent on"
            f" average {ratio:.1f}x more evaluations than the canonical implementation,"
            " and its total FE budget per run is dimension-independent because the fixed"
            " 100-direction sampling dominates both dims.")
    add("- Frozen **ASGF/ASHF Gauss-Hermite rule** uses offset sigma*p_k and prefactor"
        " 2/(sigma*sqrt(pi)) instead of the thesis sqrt(2)*sigma*p_k and"
        " sqrt(2)/(sqrt(pi)*sigma) (O(sigma)-biased legacy estimator; reproducible via"
        " `legacy_rule=True`).")
    add("- Prototype returns drop the final iterate's value (off-by-one), expose no FE"
        " counter, re-seed the global RNG inside estimators, and keep mutable class-level"
        " state in ASGF/ASHF.")
    add("")
    add("## Figures")
    add("")
    for algo in ALGORITHMS:
        add(f"![{algo}](figures/{algo}.png)")
        add("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
