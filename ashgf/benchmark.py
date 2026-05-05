"""
Benchmark and statistical testing utilities for ASHGF.

Provides functions to run mass testing of the entire function suite
with all algorithms, across different dimensions and random seeds.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

try:
    import colorama

    colorama.init()
except ImportError:
    pass

from ashgf.algorithms import (
    ASEBO, ASGF, ASGF2A, ASGF2F, ASGF2G, ASGF2H, ASGF2I, ASGF2J, ASGF2P,
    ASGF2S, ASGF2SA, ASGF2SAW, ASGF2SD, ASGF2SG, ASGF2SL, ASGF2SLA, ASGF2SLB, ASGF2SLD, ASGF2SLP, ASGF2SLR, ASGF2SLS, ASGF2SLT, ASGF2SLV, ASGF2SLV2, ASGF2SLV2P, ASGF2SLV2PS, ASGF2SLV2S, ASGF2SLVP, ASGF2SLVPS, ASGF2SLVS,
    ASGF2SM, ASGF2SMA, ASGF2SMC, ASGF2SMI, ASGF2SN, ASGF2SQ,
    ASGF2SR, ASGF2SW, ASGF2T, ASGF2X, ASGFC, ASGFHX, ASGFM,
    ASGFAQ, ASGFBW, ASGFCD, ASGFLS, ASGFLS2, ASGFLS3, ASGFLS4, ASGFLS5,
    ASGFRS, ASGFSS, ASHGF, ASHGF2F, ASHGF2FD,     ASHGF2SMA, ASHGF2SLV, ASHGF2SLV0, ASHGF2SLV2G, ASHGF2SLV2GA, ASHGF2SLV2GD, ASHGF2SLVA, ASHGF2X, ASHGFD, ASHGFNG, ASHGFS, GD, SGES,
)
from ashgf.algorithms.base import BaseOptimizer
from ashgf.functions import get_function, list_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper – sanitize sequences for log-scale plotting
# ---------------------------------------------------------------------------


def _sanitize_for_log(values, default: float = 1e-30, max_val: float = 1e150):
    """Return a copy of *values* safe for log-scale axes.

    Replaces ``inf``, ``-inf``, ``NaN``, zero, and negative values
    with *default* (a tiny positive number).
    Also clips values above *max_val* so that the log-locator
    never tries to compute ``10**decade`` for ``decade > 308``.
    """
    arr = np.asarray(values, dtype=float)
    mask = ~np.isfinite(arr) | (arr <= 0.0)
    out = np.clip(arr, default, max_val)
    out[mask] = default
    return out.tolist()


def _safe_log_scale(ax, default_ylim=(1e-30, 1e2)):
    """Set log scale on *ax* and enforce finite, positive y-limits."""
    ax.set_yscale("log", nonpositive="clip")
    lo, hi = ax.get_ylim()
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0):
        lo, hi = default_ylim
    if hi / lo < 10:
        hi = lo * 10
    ax.set_ylim(lo, hi)


# ---------------------------------------------------------------------------
# Registry of available algorithms
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, type[BaseOptimizer]] = {
    "GD": GD,
    "SGES": SGES,
    "ASGF": ASGF,
    "ASHGF": ASHGF,
    "ASEBO": ASEBO,
    "ASHGF-NG": ASHGFNG,
    "ASHGF-S": ASHGFS,
    "ASGF-RS": ASGFRS,
    "ASGF-LS": ASGFLS,
    "ASGF-LS2": ASGFLS2,
    "ASGF-LS3": ASGFLS3,
    "ASGF-LS4": ASGFLS4,
    "ASGF-LS5": ASGFLS5,
    "ASGF-M": ASGFM,
    "ASGF-2X": ASGF2X,
    "ASGF-C": ASGFC,
    "ASGF-HX": ASGFHX,
    "ASGF-2A": ASGF2A,
    "ASGF-2F": ASGF2F,
    "ASGF-2G": ASGF2G,
    "ASGF-2H": ASGF2H,
    "ASGF-2I": ASGF2I,
    "ASGF-2J": ASGF2J,
    "ASGF-2P": ASGF2P,
    "ASGF-2S": ASGF2S,
    "ASGF-2SA": ASGF2SA,
    "ASGF-2SAW": ASGF2SAW,
    "ASGF-2SD": ASGF2SD,
    "ASGF-2SG": ASGF2SG,
    "ASGF-2SL": ASGF2SL,
    "ASGF-2SLA": ASGF2SLA,
    "ASGF-2SLB": ASGF2SLB,
    "ASGF-2SLD": ASGF2SLD,
    "ASGF-2SLP": ASGF2SLP,
    "ASGF-2SLR": ASGF2SLR,
    "ASGF-2SLS": ASGF2SLS,
    "ASGF-2SLT": ASGF2SLT,
    "ASGF-2SLV": ASGF2SLV,
    "ASGF-2SLV2": ASGF2SLV2,
    "ASGF-2SLV2P": ASGF2SLV2P,
    "ASGF-2SLV2PS": ASGF2SLV2PS,
    "ASGF-2SLV2S": ASGF2SLV2S,
    "ASGF-2SLVP": ASGF2SLVP,
    "ASGF-2SLVPS": ASGF2SLVPS,
    "ASGF-2SLVS": ASGF2SLVS,
    "ASGF-2SM": ASGF2SM,
    "ASGF-2SMA": ASGF2SMA,
    "ASGF-2SMC": ASGF2SMC,
    "ASGF-2SMI": ASGF2SMI,
    "ASGF-2SN": ASGF2SN,
    "ASGF-2SQ": ASGF2SQ,
    "ASGF-2SR": ASGF2SR,
    "ASGF-2SW": ASGF2SW,
    "ASGF-2T": ASGF2T,
    "ASHGF-2F": ASHGF2F,
    "ASHGF-2FD": ASHGF2FD,
    "ASHGF-2SMA": ASHGF2SMA,
    "ASHGF-2SLV": ASHGF2SLV,
    "ASHGF-2SLV0": ASHGF2SLV0,
    "ASHGF-2SLV2G": ASHGF2SLV2G,
    "ASHGF-2SLV2GA": ASHGF2SLV2GA,
    "ASHGF-2SLV2GD": ASHGF2SLV2GD,
    "ASHGF-2SLVA": ASHGF2SLVA,
    "ASHGF-2X": ASHGF2X,
    "ASHGF-D": ASHGFD,
    "ASGF-CD": ASGFCD,
    "ASGF-SS": ASGFSS,
    "ASGF-AQ": ASGFAQ,
    "ASGF-BW": ASGFBW,
}

DEFAULT_LR = 1e-4
DEFAULT_SIGMA = 1e-4


def _make_algorithm(
    algo_name: str,
    seed: int = 2003,
    lr: float = DEFAULT_LR,
    sigma: float = DEFAULT_SIGMA,
) -> BaseOptimizer:
    """Instantiate an algorithm with appropriate default parameters."""
    algo_cls = ALGORITHMS[algo_name]
    if algo_name in ("GD", "SGES", "ASEBO"):
        return algo_cls(lr=lr, sigma=sigma, seed=seed)  # type: ignore[call-arg]
    elif algo_name == "ASHGF-NG":
        return algo_cls(seed=seed, n_jobs=1)  # type: ignore[call-arg]
    elif algo_name == "ASHGF-S":
        return algo_cls(seed=seed)  # type: ignore[call-arg]
    else:
        # ASGF and ASHGF are fully adaptive
        return algo_cls(seed=seed)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Benchmark: run all algorithms on all functions
# ---------------------------------------------------------------------------


def _run_benchmark_task(
    algo_name: str,
    func_name: str,
    dim: int,
    max_iter: int,
    seed: int,
    lr: float,
    sigma: float,
    patience: int | None,
    ftol: float | None,
    progress_cb: Callable[[int, float], None] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Run a single (algorithm, function) pair — picklable for ProcessPoolExecutor."""
    f = get_function(func_name)
    t_start = time.perf_counter()
    try:
        algo_run = _make_algorithm(algo_name, seed=seed, lr=lr, sigma=sigma)
        best_vals, all_vals = algo_run.optimize(
            f,
            dim=dim,
            max_iter=max_iter,
            debug=False,
            patience=patience,
            ftol=ftol,
            progress_cb=progress_cb,
        )
        elapsed = time.perf_counter() - t_start
        result = {
            "best": best_vals[-1][1] if best_vals else float("nan"),
            "final": all_vals[-1] if all_vals else float("nan"),
            "values": all_vals,
            "iterations": len(all_vals),
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        result = {
            "best": float("nan"),
            "final": float("nan"),
            "values": [],
            "iterations": 0,
            "elapsed": elapsed,
            "error": str(e),
        }
    return algo_name, func_name, result


def benchmark(
    functions: list[str] | None = None,
    algorithms: list[str] | None = None,
    dim: int = 100,
    max_iter: int = 1000,
    seed: int = 2003,
    lr: float = DEFAULT_LR,
    sigma: float = DEFAULT_SIGMA,
    output_dir: str | None = None,
    debug: bool = False,
    pattern: str | None = None,
    patience: int | None = None,
    ftol: float | None = None,
    n_jobs: int = 1,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run a full benchmark across functions and algorithms.

    Parameters
    ----------
    functions : list of str or None
        Function names to benchmark. If None, all registered functions
        are used.
    algorithms : list of str or None
        Algorithm names to benchmark. If None, all available algorithms
        are used.
    dim : int
        Problem dimension.
    max_iter : int
        Number of iterations per run.
    seed : int
        Random seed.
    lr : float
        Learning rate for algorithms that use a fixed LR (GD, SGES, ASEBO).
    sigma : float
        Smoothing bandwidth for algorithms that use a fixed sigma.
    output_dir : str or None
        If provided, CSV files ``<algorithm>_<function>.csv`` are written
        into this directory.
    debug : bool
        If True, print per-run details.
    pattern : str or None
        If provided, only functions whose name contains ``pattern`` are
        included (case-insensitive substring match).

    Returns
    -------
    results : dict
        Nested dict: ``results[algo_name][func_name]`` containing keys
        ``"best"``, ``"final"``, ``"values"``, ``"iterations"``,
        ``"elapsed"``.
    """
    if functions is None:
        functions = list_functions()
        # Exclude RL environments by default (require gymnasium, slow)
        functions = [f for f in functions if not f.startswith("RL")]
    if pattern is not None:
        pat = pattern.lower()
        functions = [f for f in functions if pat in f.lower()]
    if algorithms is None:
        algorithms = sorted(ALGORITHMS.keys())

    results: dict[str, dict[str, dict[str, Any]]] = {algo: {} for algo in algorithms}

    # Build flat list of tasks
    tasks = [
        (algo_name, func_name, dim, max_iter, seed, lr, sigma, patience, ftol)
        for algo_name in algorithms
        for func_name in functions
    ]

    if n_jobs > 1:
        # Parallel execution with global progress bar
        total_tasks = len(tasks)
        pbar = tqdm(
            total=total_tasks,
            desc="Benchmark",
            bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            disable=not debug,
            leave=True,
            file=sys.stdout,
        )
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_run_benchmark_task, *task): task[:2] for task in tasks
            }
            for fut in as_completed(futures):
                algo_name, func_name = futures[fut]
                try:
                    a_name, f_name, result = fut.result()
                    pbar.update(1)
                    pbar.set_postfix_str(f"{f_name} d={dim}")
                    if debug:
                        logger.info(
                            "Benchmark %-6s on %-35s dim=%-4d max_iter=%d (elapsed=%.2fs)",
                            a_name,
                            f_name,
                            dim,
                            max_iter,
                            result.get("elapsed", 0),
                        )
                    results[a_name][f_name] = result
                except Exception as e:
                    pbar.update(1)
                    logger.error(
                        "Benchmark %-6s on %-35s FAILED: %s",
                        algo_name,
                        func_name,
                        e,
                    )
                    results[algo_name][func_name] = {
                        "best": float("nan"),
                        "final": float("nan"),
                        "values": [],
                        "iterations": 0,
                        "elapsed": 0.0,
                        "error": str(e),
                    }
        pbar.close()
    else:
        # Sequential execution with live per-task progress bars (matching Rust style)
        for algo_name, func_name, _, _, _, _, _, _, _ in tasks:
            desc = f"dim={dim:<4} {func_name:<25} {algo_name:<10}"
            with tqdm(
                total=max_iter,
                desc=desc,
                bar_format="{desc} {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                disable=not debug,
                leave=False,
                file=sys.stdout,
            ) as pbar:

                pb_ref = pbar  # capture for closure

                def _cb(iteration: int, fval: float) -> None:
                    pb_ref.update(1)
                    pb_ref.set_postfix_str(f"f={fval:.4e}")

                a_name, f_name, result = _run_benchmark_task(
                    algo_name, func_name, dim, max_iter, seed, lr, sigma, patience, ftol,
                    progress_cb=_cb,
                )
            if debug:
                logger.info(
                    "Benchmark %-6s on %-35s dim=%-4d max_iter=%d (elapsed=%.2fs)",
                    a_name,
                    f_name,
                    dim,
                    max_iter,
                    result.get("elapsed", 0),
                )
            results[a_name][f_name] = result

    # Optionally write CSV files
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for algo_name in algorithms:
            for func_name in functions:
                vals = results[algo_name][func_name].get("values", [])
                if vals:
                    csv_path = os.path.join(output_dir, f"{algo_name}_{func_name}.csv")
                    with open(csv_path, "w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(["iteration", "value"])
                        for i, v in enumerate(vals):
                            writer.writerow([i, v])

    return results


def print_benchmark_summary(
    results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Print a summary table of benchmark results (Rust-style flat format)."""
    algorithms = sorted(results.keys())

    rows: list[tuple[str, str, float, float, int]] = []
    for algo in algorithms:
        for fn, entry in sorted(results[algo].items()):
            best = entry.get("best", float("nan"))
            final = entry.get("final", float("nan"))
            iters = entry.get("iterations", 0)
            rows.append((algo, fn, final, best, iters))

    rows.sort(key=lambda r: (r[1], r[3] if np.isfinite(r[3]) else float("inf")))

    header = f"{'ALGO':<8} {'FUNCTION':<35} {'FINAL':>14} {'BEST':>14} {'ITER':>6}"
    print()
    print(header)
    print("-" * 80)

    for algo, func, final, best, iters in rows:
        f_str = f"{final:>14.6e}" if np.isfinite(final) else f"{'FAIL':>14}"
        b_str = f"{best:>14.6e}" if np.isfinite(best) else f"{'FAIL':>14}"
        print(f"{algo:<8} {func:<35} {f_str} {b_str} {iters:>6}")


# ---------------------------------------------------------------------------
# Statistical testing: multiple independent runs
# ---------------------------------------------------------------------------


def _run_stats_task(
    algo_name: str,
    function: str,
    dim: int,
    max_iter: int,
    run_seed: int,
    lr: float,
    sigma: float,
    patience: int | None,
    ftol: float | None,
) -> tuple[str, list[float]]:
    """Run a single (algorithm, seed) trial — picklable for ProcessPoolExecutor."""
    f = get_function(function)
    algo = _make_algorithm(algo_name, seed=run_seed, lr=lr, sigma=sigma)
    _best_vals, all_vals = algo.optimize(
        f,
        dim=dim,
        max_iter=max_iter,
        debug=False,
        patience=patience,
        ftol=ftol,
    )
    return algo_name, all_vals


def statistics(
    function: str,
    algorithms: list[str] | None = None,
    dim: int = 100,
    max_iter: int = 1000,
    n_runs: int = 30,
    seed: int = 2003,
    lr: float = DEFAULT_LR,
    sigma: float = DEFAULT_SIGMA,
    output_dir: str | None = None,
    debug: bool = True,
    patience: int | None = None,
    ftol: float | None = None,
    n_jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Run multiple independent trials and compute convergence statistics.

    Parameters
    ----------
    function : str
        Test function name.
    algorithms : list of str or None
        Algorithm names. If None, all algorithms are used.
    dim : int
        Problem dimension.
    max_iter : int
        Number of iterations per run.
    n_runs : int
        Number of independent repetitions.
    seed : int
        Base random seed. Each run uses ``seed + i``.
    lr : float
        Learning rate for fixed-LR algorithms.
    sigma : float
        Smoothing bandwidth for fixed-sigma algorithms.
    output_dir : str or None
        If provided, pickled results and/or CSVs are saved.
    debug : bool
        If True, log progress.

    Returns
    -------
    stats : dict
        Nested dict ``stats[algo_name]`` containing keys:
        - ``all_sequences``: list of value sequences (one per run)
        - ``mean``: array of per-iteration means
        - ``std``: array of per-iteration standard deviations
        - ``min``: array of per-iteration minima
        - ``max``: array of per-iteration maxima
        - ``best_mean``: mean of best values across runs
        - ``best_std``: std of best values across runs
    """
    if algorithms is None:
        algorithms = sorted(ALGORITHMS.keys())

    f = get_function(function)

    stats: dict[str, dict[str, Any]] = {}

    for algo_name in algorithms:
        all_sequences: list[list[float]] = []
        best_finals: list[float] = []

        # Build task list: (algo_name, function, dim, max_iter, run_seed, lr, sigma, patience, ftol)
        run_tasks = [
            (algo_name, function, dim, max_iter, seed + run, lr, sigma, patience, ftol)
            for run in range(n_runs)
        ]

        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(_run_stats_task, *task): task[4]
                    for task in run_tasks
                }
                for fut in as_completed(futures):
                    run_seed = futures[fut]
                    run_idx = run_seed - seed
                    try:
                        a_name, all_vals = fut.result()
                        if debug:
                            logger.info(
                                "Stats %-6s on %-35s run %3d/%d",
                                a_name,
                                function,
                                run_idx + 1,
                                n_runs,
                            )
                        all_sequences.append(all_vals)
                        best_finals.append(min(all_vals) if all_vals else float("nan"))
                    except Exception as e:
                        logger.error(
                            "Stats %-6s on %-35s run %d FAILED: %s",
                            algo_name,
                            function,
                            run_idx,
                            e,
                        )
        else:
            for run in range(n_runs):
                run_seed = seed + run
                if debug:
                    logger.info(
                        "Stats %-6s on %-35s run %3d/%d",
                        algo_name,
                        function,
                        run + 1,
                        n_runs,
                    )

                try:
                    algo = _make_algorithm(algo_name, seed=run_seed, lr=lr, sigma=sigma)
                    _best_vals, all_vals = algo.optimize(
                        f,
                        dim=dim,
                        max_iter=max_iter,
                        debug=False,
                        patience=patience,
                        ftol=ftol,
                    )
                    all_sequences.append(all_vals)
                    best_finals.append(min(all_vals) if all_vals else float("nan"))
                except Exception as e:
                    logger.error(
                        "Stats %-6s on %-35s run %d FAILED: %s",
                        algo_name,
                        function,
                        run,
                        e,
                    )

        if not all_sequences:
            stats[algo_name] = {
                "all_sequences": [],
                "mean": np.array([]),
                "std": np.array([]),
                "min": np.array([]),
                "max": np.array([]),
                "best_mean": float("nan"),
                "best_std": float("nan"),
                "error": "All runs failed",
            }
            continue

        # Align sequences to the minimum length across runs
        min_len = min(len(seq) for seq in all_sequences)
        aligned = np.array([seq[:min_len] for seq in all_sequences])

        mean_seq = np.nanmean(aligned, axis=0)
        std_seq = np.nanstd(aligned, axis=0)
        min_seq = np.nanmin(aligned, axis=0)
        max_seq = np.nanmax(aligned, axis=0)

        best_mean = float(np.mean(best_finals))
        best_std = float(np.std(best_finals))

        stats[algo_name] = {
            "all_sequences": all_sequences,
            "mean": mean_seq,
            "std": std_seq,
            "min": min_seq,
            "max": max_seq,
            "best_mean": best_mean,
            "best_std": best_std,
        }

    # Optionally save to disk
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        import pickle

        for algo_name in algorithms:
            if algo_name in stats:
                pkl_path = os.path.join(output_dir, f"stats_{function}_{algo_name}.pkl")
                with open(pkl_path, "wb") as fh:
                    pickle.dump(stats[algo_name], fh)

    return stats


def print_statistics_summary(
    stats: dict[str, dict[str, Any]],
    function: str,
) -> None:
    """Print a summary table of statistical results."""
    print(f"\nStatistical summary for '{function}':")
    header = f"{'Algorithm':>8}  {'Best (mean)':>14}  {'Best (std)':>14}  {'Runs':>6}"
    print(header)
    print("-" * len(header))

    for algo_name in sorted(stats.keys()):
        s = stats[algo_name]
        n = len(s.get("all_sequences", []))
        best_m = s.get("best_mean", float("nan"))
        best_s = s.get("best_std", float("nan"))
        print(f"{algo_name:>8}  {best_m:>14.6e}  {best_s:>14.6e}  {n:>6d}")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def plot_statistics(
    stats: dict[str, dict[str, Any]],
    function: str,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Generate convergence plots with mean ± std bands.

    Parameters
    ----------
    stats : dict
        Output of :func:`statistics`.
    function : str
        Function name (for title).
    output_path : str or None
        If provided, save the figure to this file path.
    show : bool
        If True, display the plot interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.get_cmap("tab10")
    n_algos = len(stats)
    color_idx = 0

    for algo_name in sorted(stats.keys()):
        s = stats[algo_name]
        if "mean" not in s or len(s["mean"]) == 0:
            continue

        color = cmap(color_idx / max(1, n_algos - 1)) if n_algos > 1 else cmap(0.0)
        color_idx += 1

        iters = np.arange(len(s["mean"]))

        # Plot mean
        mean_clean = _sanitize_for_log(s["mean"])
        ax1.plot(iters, mean_clean, label=algo_name, color=color)
        if len(s["std"]) > 0:
            low = np.maximum(np.asarray(s["mean"]) - np.asarray(s["std"]), 1e-16)
            high = np.asarray(s["mean"]) + np.asarray(s["std"])
            ax1.fill_between(
                iters,
                _sanitize_for_log(low),
                _sanitize_for_log(high),
                alpha=0.2,
                color=color,
            )

        # Plot min / max envelope
        ax2.plot(
            iters,
            _sanitize_for_log(s["min"]),
            label=f"{algo_name} min",
            color=color,
            alpha=0.6,
        )
        ax2.plot(
            iters,
            _sanitize_for_log(s["max"]),
            label=f"{algo_name} max",
            color=color,
            alpha=0.6,
            linestyle="--",
        )

    _safe_log_scale(ax1)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)  (mean ± std)")
    ax1.set_title(f"{function} — Mean convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    _safe_log_scale(ax2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)  (min / max)")
    ax2.set_title(f"{function} — Envelope")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except Exception:
        logger.warning(
            "tight_layout failed for statistics plot – skipping layout adjustment"
        )

    if output_path:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Plot saved to %s", output_path)
        except Exception:
            logger.exception("Failed to save statistics plot to %s", output_path)

    if show:
        try:
            plt.show()
        except Exception:
            logger.exception("plt.show() failed")
    else:
        plt.close(fig)


# Appended to benchmark.py

# ---------------------------------------------------------------------------
# Multi-dimension benchmark
# ---------------------------------------------------------------------------


def benchmark_multi(
    functions: list[str] | None = None,
    algorithms: list[str] | None = None,
    dims: list[int] | None = None,
    max_iter: int = 1000,
    seed: int = 2003,
    lr: float = DEFAULT_LR,
    sigma: float = DEFAULT_SIGMA,
    output_dir: str | None = None,
    debug: bool = False,
    pattern: str | None = None,
    patience: int | None = None,
    ftol: float | None = None,
    n_jobs: int = 1,
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    """Run benchmark across multiple dimensions.

    Returns nested dict: ``results[dim][algo][func]``.
    """
    if dims is None:
        dims = [10, 100, 1000]

    all_results: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}

    for dim in dims:
        if debug:
            logger.info("=== Dimension %d ===", dim)
        all_results[dim] = benchmark(
            functions=functions,
            algorithms=algorithms,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            lr=lr,
            sigma=sigma,
            output_dir=(os.path.join(output_dir, f"dim_{dim}") if output_dir else None),
            debug=debug,
            pattern=pattern,
            patience=patience,
            ftol=ftol,
            n_jobs=n_jobs,
        )

    return all_results


def print_benchmark_multi_summary(
    results: dict[int, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    """Print one summary table per dimension."""
    for dim in sorted(results.keys()):
        print(f"\n{'=' * 80}")
        print(f"  Dimension: {dim}")
        print(f"{'=' * 80}")
        print_benchmark_summary(results[dim])


# ---------------------------------------------------------------------------
# Report generation (REPORT.md)
# ---------------------------------------------------------------------------


def _compute_wins(flat: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Compute per-algorithm wins: lowest best_value on each (dim, func)."""
    from collections import defaultdict

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in flat:
        groups[(r["dim"], r["func"])].append(r)

    win_counts: dict[str, int] = defaultdict(int)
    for (_dim, _func), entries in groups.items():
        valid = [(e["algo"], e["best"]) for e in entries if np.isfinite(e["best"])]
        if not valid:
            continue
        best = min(v for _a, v in valid)
        for algo, val in valid:
            if abs(val - best) < 1e-12:
                win_counts[algo] += 1

    return sorted(win_counts.items(), key=lambda x: (-x[1], x[0]))


def _convergence_stats(
    flat: list[dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Compute (converged, total) per algorithm."""
    from collections import defaultdict

    stats: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in flat:
        algo = r["algo"]
        conv, total = stats[algo]
        stats[algo] = (conv + (1 if r["converged"] else 0), total + 1)
    return dict(stats)


def _flatten_results(
    results: dict[int, dict[str, dict[str, dict[str, Any]]]],
    max_iter: int,
) -> list[dict[str, Any]]:
    """Flatten multi-dim benchmark results into a list of per-run dicts."""
    flat: list[dict[str, Any]] = []
    for dim, dim_data in results.items():
        for algo, funcs in dim_data.items():
            for func, entry in funcs.items():
                iters = entry.get("iterations", 0)
                flat.append({
                    "dim": dim,
                    "algo": algo,
                    "func": func,
                    "best": entry.get("best", float("nan")),
                    "final": entry.get("final", float("nan")),
                    "iters": iters,
                    "converged": iters > 0 and iters < max_iter,
                })
    return flat


def generate_report(
    results: dict[int, dict[str, dict[str, dict[str, Any]]]],
    seed: int,
    max_iter: int,
    output_dir: str,
) -> None:
    """Generate a Markdown benchmark report (matching Rust REPORT.md format).

    Parameters
    ----------
    results : dict
        Output of :func:`benchmark_multi` (or ``{dim: results}`` wrapping
        the output of :func:`benchmark`).
    seed : int
        Random seed used for the benchmark.
    max_iter : int
        Maximum iterations per run.
    output_dir : str
        Directory where ``REPORT.md`` will be written.
    """
    import datetime

    os.makedirs(output_dir, exist_ok=True)

    dims = sorted(results.keys())
    flat = _flatten_results(results, max_iter)

    if not flat:
        logger.warning("No benchmark results to report.")
        return

    func_set = sorted({r["func"] for r in flat})
    algo_set = sorted({r["algo"] for r in flat})
    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    lines: list[str] = []
    lines.append("# Benchmark Report — ASHGF")
    lines.append("")
    lines.append(f"**Generated**: {now}")
    lines.append(f"**Seed**: {seed}")
    lines.append(f"**Max iterations**: {max_iter}")
    lines.append(f"**Dimensions**: {', '.join(str(d) for d in dims)}")
    lines.append(f"**Functions**: {len(func_set)}")
    lines.append(f"**Algorithms**: {', '.join(algo_set)}")
    lines.append("")

    # --- 1. Overall Leaderboard ---
    lines.append("## Overall Leaderboard (wins per function)")
    lines.append("")
    wins = _compute_wins(flat)
    lines.append("| Algorithm | Wins | Win % |")
    lines.append("|-----------|------|-------|")
    total_funcs = len(func_set) * len(dims)
    for algo, w in wins:
        pct = w / total_funcs * 100 if total_funcs > 0 else 0.0
        lines.append(f"| {algo} | {w} | {pct:.1f}% |")
    lines.append("")

    # --- 2. Convergence Rate ---
    lines.append("## Convergence Rate")
    lines.append("")
    conv_stats = _convergence_stats(flat)
    lines.append("| Algorithm | Converged | Total | Rate |")
    lines.append("|-----------|-----------|-------|------|")
    for algo in sorted(conv_stats):
        conv, total = conv_stats[algo]
        rate = conv / total * 100 if total > 0 else 0.0
        lines.append(f"| {algo} | {conv} | {total} | {rate:.1f}% |")
    lines.append("")

    # --- 3. Per-dimension tables ---
    for dim in dims:
        dim_flat = [r for r in flat if r["dim"] == dim]
        if not dim_flat:
            continue
        lines.append("---")
        lines.append("")
        lines.append(f"## Dimension d = {dim}")
        lines.append("")

        dim_funcs = sorted({r["func"] for r in dim_flat})
        dim_algos = sorted({r["algo"] for r in dim_flat})

        # Best-value map: (algo, func) -> best
        best_map: dict[tuple[str, str], float] = {}
        for r in dim_flat:
            key = (r["algo"], r["func"])
            val = r["best"]
            if np.isfinite(val):
                best_map[key] = (
                    min(best_map[key], val) if key in best_map else val
                )

        # Header row
        header = "| Function " + "".join(f"| {a} " for a in dim_algos) + "|"
        lines.append(header)
        sep = "|----------" + "|----------" * len(dim_algos) + "|"
        lines.append(sep)

        # Data rows
        for func in dim_funcs:
            func_bests: list[float] = []
            cols: list[str] = []
            for algo in dim_algos:
                val = best_map.get((algo, func))
                if val is not None and np.isfinite(val):
                    func_bests.append(val)
                else:
                    func_bests.append(float("nan"))

            best = min(func_bests) if func_bests else float("nan")

            for idx, algo in enumerate(dim_algos):
                val = best_map.get((algo, func))
                if val is not None and np.isfinite(val):
                    if np.isfinite(best) and abs(val - best) < 1e-12:
                        cols.append(f"| **{val:.4e}** ")
                    else:
                        cols.append(f"| {val:.4e} ")
                else:
                    cols.append("| — ")

            lines.append(f"| {func} " + "".join(cols) + "|")
        lines.append("")

        # Per-dimension wins
        dim_wins = _compute_wins(dim_flat)
        lines.append(f"### Wins (d={dim})")
        lines.append("")
        lines.append("| Algorithm | Wins |")
        lines.append("|-----------|------|")
        for algo, w in dim_wins:
            lines.append(f"| {algo} | {w} |")
        lines.append("")

    # Write to file
    report_path = os.path.join(output_dir, "REPORT.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("REPORT.md written to %s", report_path)
    except Exception:
        logger.exception("Cannot write REPORT.md")


# ---------------------------------------------------------------------------
# Comparison plots (multi-algorithm, multi-dimension)
# ---------------------------------------------------------------------------


def plot_benchmark_comparison(
    results: dict[int, dict[str, dict[str, dict[str, Any]]]],
    output_path: str | None = None,
    show: bool = False,
    top_n: int | None = None,
) -> None:
    """Generate multi-panel comparison plots from benchmark results.

    Creates one subplot per dimension.  Each subplot is a grouped bar chart
    comparing algorithms on each function (best value found, log scale).

    Parameters
    ----------
    results : dict
        Output of :func:`benchmark_multi`:
        ``results[dim][algo][func]``.
    output_path : str or None
        Save figure to this path.
    show : bool
        Display interactively.
    top_n : int or None
        Show only the first ``top_n`` functions (None = all).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return

    dims = sorted(results.keys())
    n_dims = len(dims)

    # Collect all (algo, func) pairs from the first dimension
    first_dim = dims[0]
    algos = sorted(results[first_dim].keys())
    func_names: set[str] = set()
    for algo in algos:
        func_names.update(results[first_dim][algo].keys())
    func_names_sorted = (
        sorted(func_names) if top_n is None else sorted(func_names)[:top_n]
    )

    n_funcs = len(func_names_sorted)
    n_algos = len(algos)

    fig, axes = plt.subplots(
        1,
        n_dims,
        figsize=(6 * n_dims, max(6, n_funcs * 0.4)),
        squeeze=False,
    )

    x = np.arange(n_funcs)
    width = 0.8 / n_algos
    cmap = plt.get_cmap("tab10")

    for col, dim in enumerate(dims):
        ax = axes[0, col]
        for i, algo in enumerate(algos):
            color = cmap(i / max(1, n_algos - 1)) if n_algos > 1 else cmap(0.0)
            values = []
            for fn in func_names_sorted:
                entry = results[dim][algo].get(fn, {})
                best = entry.get("best", float("nan"))
                values.append(best if np.isfinite(best) else np.nan)
            # sanitize for log scale (nan -> small positive)
            clean_vals = _sanitize_for_log(values)
            ax.bar(x + i * width, clean_vals, width, label=algo, color=color)

        _safe_log_scale(ax)
        ax.set_xticks(x + width * (n_algos - 1) / 2)
        ax.set_xticklabels(func_names_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Best f(x)")
        ax.set_title(f"Dimension = {dim}")
        ax.grid(axis="y", alpha=0.3)
        if col == n_dims - 1:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.suptitle(
        "Benchmark: best value per function, algorithm, and dimension",
        fontsize=14,
        fontweight="bold",
    )

    try:
        plt.tight_layout()
    except Exception:
        logger.warning(
            "tight_layout failed for comparison plot – skipping layout adjustment"
        )

    if output_path:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Comparison plot saved to %s", output_path)
        except Exception:
            logger.exception("Failed to save comparison plot to %s", output_path)

    if show:
        try:
            plt.show()
        except Exception:
            logger.exception("plt.show() failed")
    else:
        plt.close(fig)


def plot_convergence_grid(
    results: dict[int, dict[str, dict[str, dict[str, Any]]]],
    functions: list[str] | None = None,
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Grid of convergence curves: rows = functions, columns = dimensions.

    Each cell shows f(x) vs iterations for all algorithms on that
    function/dimension combination.

    Parameters
    ----------
    results : dict
        Output of :func:`benchmark_multi`.
    functions : list of str or None
        Which functions to plot. If None, first 9 alphabetically.
    output_path : str or None
        Save figure to this path.
    show : bool
        Display interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return

    dims = sorted(results.keys())
    n_dims = len(dims)
    algos = sorted(results[dims[0]].keys())

    # Pick functions
    if functions is None:
        all_funcs: set[str] = set()
        for algo in algos:
            all_funcs.update(results[dims[0]][algo].keys())
        functions = sorted(all_funcs)
        # Exclude RL environments
        functions = [f for f in functions if not f.startswith("RL")]

    n_funcs = len(functions)
    n_algos = len(algos)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        n_funcs,
        n_dims,
        figsize=(5 * n_dims, 3 * n_funcs),
        squeeze=False,
    )

    for row, fn in enumerate(functions):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            for i, algo in enumerate(algos):
                color = cmap(i / max(1, n_algos - 1)) if n_algos > 1 else cmap(0.0)
                entry = results[dim][algo].get(fn, {})
                vals = entry.get("values", [])
                if vals:
                    clean = _sanitize_for_log(vals)
                    ax.plot(clean, label=algo, color=color, linewidth=0.8)
                else:
                    ax.plot([1e-30, 1e-30], label=algo, color=color, linewidth=0.8)
            _safe_log_scale(ax)
            ax.set_xlabel("Iteration")
            if col == 0:
                ax.set_ylabel(fn[:30])
            if row == 0:
                ax.set_title(f"dim={dim}")
            ax.grid(True, alpha=0.2)

    # Single legend at the bottom
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=cmap(i / max(1, n_algos - 1)), label=algo)
        for i, algo in enumerate(algos)
    ]
    fig.legend(handles, algos, loc="lower center", ncol=n_algos, fontsize=8)
    fig.suptitle("Convergence curves", fontsize=14, fontweight="bold")

    try:
        plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    except Exception:
        logger.warning(
            "tight_layout failed for convergence grid – skipping layout adjustment"
        )

    if output_path:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Convergence grid saved to %s", output_path)
        except Exception:
            logger.exception("Failed to save convergence grid to %s", output_path)

    if show:
        try:
            plt.show()
        except Exception:
            logger.exception("plt.show() failed")
    else:
        plt.close(fig)


def plot_per_function(
    results,
    output_dir="results",
    functions=None,
    show=False,
):
    import logging
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    logger = logging.getLogger(__name__)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return []

    os.makedirs(output_dir, exist_ok=True)

    dims = sorted(results.keys())
    n_dims = len(dims)
    algos = sorted(results[dims[0]].keys())
    n_algos = len(algos)
    cmap = plt.get_cmap("tab10")

    if functions is None:
        all_funcs = set()
        for algo in algos:
            all_funcs.update(results[dims[0]][algo].keys())
        functions = sorted(all_funcs)
        functions = [f for f in functions if not f.startswith("RL")]

    saved_paths = []

    for fn in functions:
        fig, axes = plt.subplots(
            n_dims,
            n_algos,
            figsize=(4 * n_algos, 3 * n_dims),
            squeeze=False,
        )

        for row, dim in enumerate(dims):
            for col, algo in enumerate(algos):
                ax = axes[row, col]
                color = cmap(col / max(1, n_algos - 1)) if n_algos > 1 else cmap(0.0)
                entry = results[dim][algo].get(fn, {})
                vals = entry.get("values", [])
                if vals:
                    clean = _sanitize_for_log(vals)
                    ax.plot(clean, color=color, linewidth=0.8)
                    best_idx = int(np.argmin(clean))
                    ax.axvline(x=best_idx, color=color, linestyle=":", alpha=0.5)
                else:
                    # still need a non-empty safe range so log scale doesn't break
                    ax.plot([1e-30, 1e-30], color=color, linewidth=0.8)
                _safe_log_scale(ax)
                ax.grid(True, alpha=0.2)
                if row == 0:
                    ax.set_title(algo, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"dim={dim}")
                if row == n_dims - 1:
                    ax.set_xlabel("Iteration")

        fig.suptitle(
            f"{fn}  -  Convergence per algorithm and dimension",
            fontsize=12,
            fontweight="bold",
        )

        try:
            plt.tight_layout()
        except Exception:
            logger.warning(
                "tight_layout failed for %s – skipping layout adjustment", fn
            )

        fname = f"{fn}.png"
        path = os.path.join(output_dir, fname)
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            saved_paths.append(path)
            logger.info("Saved %s", path)
        except Exception:
            logger.exception("Failed to save plot for %s", fn)
        finally:
            plt.close(fig)

    return saved_paths
