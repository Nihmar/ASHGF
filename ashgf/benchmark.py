"""
Benchmark and statistical testing utilities for ASHGF.

Provides functions to run mass testing of the entire function suite
with all algorithms, across different dimensions and random seeds.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any

import numpy as np

from ashgf.algorithms import ASEBO, ASGF, ASHGF, GD, SGES
from ashgf.algorithms.base import BaseOptimizer
from ashgf.functions import get_function, list_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of available algorithms
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, type[BaseOptimizer]] = {
    "GD": GD,
    "SGES": SGES,
    "ASGF": ASGF,
    "ASHGF": ASHGF,
    "ASEBO": ASEBO,
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
    else:
        # ASGF and ASHGF are fully adaptive
        return algo_cls(seed=seed)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Benchmark: run all algorithms on all functions
# ---------------------------------------------------------------------------


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
    if pattern is not None:
        pat = pattern.lower()
        functions = [f for f in functions if pat in f.lower()]
    if algorithms is None:
        algorithms = sorted(ALGORITHMS.keys())

    results: dict[str, dict[str, dict[str, Any]]] = {algo: {} for algo in algorithms}

    for algo_name in algorithms:
        for func_name in functions:
            f = get_function(func_name)

            if debug:
                logger.info(
                    "Benchmark %-6s on %-35s dim=%-4d max_iter=%d",
                    algo_name,
                    func_name,
                    dim,
                    max_iter,
                )

            t_start = time.perf_counter()
            try:
                # Re-instantiate with the same seed for reproducibility
                algo_run = _make_algorithm(algo_name, seed=seed, lr=lr, sigma=sigma)
                best_vals, all_vals = algo_run.optimize(
                    f, dim=dim, max_iter=max_iter, debug=False
                )
                elapsed = time.perf_counter() - t_start

                results[algo_name][func_name] = {
                    "best": best_vals[-1][1] if best_vals else float("nan"),
                    "final": all_vals[-1] if all_vals else float("nan"),
                    "values": all_vals,
                    "iterations": len(all_vals),
                    "elapsed": elapsed,
                }
            except Exception as e:
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
                    "elapsed": time.perf_counter() - t_start,
                    "error": str(e),
                }

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
    """Print a summary table of benchmark results."""
    algorithms = sorted(results.keys())

    # Gather all function names
    func_names: set[str] = set()
    for algo in algorithms:
        func_names.update(results[algo].keys())
    func_names_sorted = sorted(func_names)

    # Header
    header = f"{'Function':<40}" + "".join(f"{a:>14}" for a in algorithms)
    print(header)
    print("-" * len(header))

    for fn in func_names_sorted:
        row = f"{fn:<40}"
        for algo in algorithms:
            entry = results[algo].get(fn, {})
            best = entry.get("best", float("nan"))
            if np.isfinite(best):
                row += f"{best:>14.6e}"
            else:
                row += f"{'FAIL':>14}"
        print(row)


# ---------------------------------------------------------------------------
# Statistical testing: multiple independent runs
# ---------------------------------------------------------------------------


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
                    f, dim=dim, max_iter=max_iter, debug=False
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
        ax1.plot(iters, s["mean"], label=algo_name, color=color)
        if len(s["std"]) > 0:
            ax1.fill_between(
                iters,
                np.maximum(s["mean"] - s["std"], 1e-16),
                s["mean"] + s["std"],
                alpha=0.2,
                color=color,
            )

        # Plot min / max envelope
        ax2.plot(iters, s["min"], label=f"{algo_name} min", color=color, alpha=0.6)
        ax2.plot(
            iters,
            s["max"],
            label=f"{algo_name} max",
            color=color,
            alpha=0.6,
            linestyle="--",
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("f(x)  (mean ± std)")
    ax1.set_title(f"{function} — Mean convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("f(x)  (min / max)")
    ax2.set_title(f"{function} — Envelope")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Plot saved to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
