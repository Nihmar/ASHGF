#!/usr/bin/env python
"""Comparative benchmark: ASHGF vs its 4 structural variants.

Tests each variant **individually** against the baseline ASHGF on all
registered benchmark functions.  The goal is to understand which
structural change actually improves performance.

Variants
--------
- ASHGF      : baseline (original algorithm)
- ASHGF-BT   : backtracking safeguard (inline, no ensemble)
- ASHGF-PID  : PID sigma controller (replaces bang-bang)
- ASHGF-SOFT : soft basis evolution (replaces hard reset)
- ASHGF-SMALPHA : smoothed alpha update (replaces binary rule)

Usage
-----
    python test_variants.py                  # default: dim=100, max_iter=500, seed=2003
    python test_variants.py --dim 50         # smaller dimension
    python test_variants.py --quick          # quick test: dim=30, max_iter=200
    python test_variants.py --full           # full test: dim=100, max_iter=1000
    python test_variants.py --classic-only   # only classic functions
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np

from ashgf.algorithms import ASHGF, ASHGFBT, ASHGFPID, ASHGFSMALPHA, ASHGFSOFT
from ashgf.algorithms.base import BaseOptimizer
from ashgf.functions import get_function, list_functions

# ---------------------------------------------------------------------------
# Registry of variants to test
# ---------------------------------------------------------------------------

VARIANTS: dict[str, type[BaseOptimizer]] = {
    "ASHGF": ASHGF,
    "ASHGF-BT": ASHGFBT,
    "ASHGF-PID": ASHGFPID,
    "ASHGF-SOFT": ASHGFSOFT,
    "ASHGF-SMALPHA": ASHGFSMALPHA,
}


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------


def run_one(
    algo_cls: type[BaseOptimizer],
    func_name: str,
    func: Any,
    dim: int,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    """Run one (algorithm, function) pair and return results."""
    t_start = time.perf_counter()
    try:
        algo = algo_cls(seed=seed)
        best_vals, all_vals = algo.optimize(
            func,
            dim=dim,
            max_iter=max_iter,
            debug=False,
        )
        elapsed = time.perf_counter() - t_start
        return {
            "best": best_vals[-1][1] if best_vals else float("nan"),
            "final": all_vals[-1] if all_vals else float("nan"),
            "iterations": len(all_vals),
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        return {
            "best": float("nan"),
            "final": float("nan"),
            "iterations": 0,
            "elapsed": elapsed,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ASHGF structural variants")
    parser.add_argument("--dim", type=int, default=100, help="Problem dimension")
    parser.add_argument("--max-iter", type=int, default=500, help="Max iterations")
    parser.add_argument("--seed", type=int, default=2003, help="Random seed")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test: dim=30, max_iter=200"
    )
    parser.add_argument(
        "--full", action="store_true", help="Full test: dim=100, max_iter=1000"
    )
    parser.add_argument(
        "--classic-only", action="store_true", help="Only classic test functions"
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only CUTEst-style benchmark functions",
    )
    parser.add_argument(
        "--function", type=str, default=None, help="Run a single function only"
    )
    args = parser.parse_args()

    # Override with presets
    dim = args.dim
    max_iter = args.max_iter
    if args.quick:
        dim = 30
        max_iter = 200
    elif args.full:
        dim = 100
        max_iter = 1000

    seed = args.seed

    # Build function list
    all_funcs = list_functions()

    # Categorize
    classic_names = {
        "sphere",
        "ackley",
        "rastrigin",
        "levy",
        "rosenbrock",
        "griewank",
        "schwefel",
        "sincos",
        "sine",
        "cosine",
        "trid",
        "zakharov",
        "sum_of_different_powers",
    }
    rl_names = {f for f in all_funcs if f.startswith("RL")}

    if args.function:
        func_list = [args.function]
    elif args.classic_only:
        func_list = sorted(classic_names & set(all_funcs))
    elif args.benchmark_only:
        func_list = sorted(set(all_funcs) - classic_names - rl_names)
    else:
        # All non-RL functions
        func_list = sorted(set(all_funcs) - rl_names)

    # Exclude functions known to cause issues (domain errors at random init)
    # Some functions have sqrt/div by zero issues for random starting points
    skip_funcs = {
        "cube",  # x^(3/4) domain issue with negative x from randn
        "fh3",  # similar domain issues
        "bdexp",  # exp overflow
        "genhumps",  # sinh overflow
    }
    func_list = [f for f in func_list if f not in skip_funcs]

    print(f"{'=' * 90}")
    print(f"  ASHGF Structural Variants Benchmark")
    print(f"  dim={dim}  max_iter={max_iter}  seed={seed}")
    print(f"  Functions: {len(func_list)}")
    print(f"  Variants: {list(VARIANTS.keys())}")
    print(f"{'=' * 90}")
    print()

    # Run all
    results: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in VARIANTS}

    for i, func_name in enumerate(func_list):
        try:
            f = get_function(func_name)
        except KeyError:
            print(f"  [{i + 1}/{len(func_list)}] {func_name:<45s} SKIP (not found)")
            continue

        print(f"  [{i + 1}/{len(func_list)}] {func_name:<45s}", end="", flush=True)

        func_results: dict[str, dict[str, Any]] = {}
        for var_name, var_cls in VARIANTS.items():
            result = run_one(var_cls, func_name, f, dim, max_iter, seed)
            func_results[var_name] = result
            results[var_name][func_name] = result

        # Quick per-function summary
        baseline_best = func_results["ASHGF"]["best"]
        best_among_variants = baseline_best
        best_variant = "ASHGF"
        for var_name in ["ASHGF-BT", "ASHGF-PID", "ASHGF-SOFT", "ASHGF-SMALPHA"]:
            val = func_results[var_name]["best"]
            if np.isfinite(val) and (
                not np.isfinite(best_among_variants) or val < best_among_variants
            ):
                best_among_variants = val
                best_variant = var_name

        if np.isfinite(baseline_best):
            print(f"  best={baseline_best:.4e}", end="")
            if best_variant != "ASHGF":
                improvement = baseline_best - best_among_variants
                print(
                    f"  → {best_variant}: {best_among_variants:.4e} (Δ={improvement:.2e})",
                    end="",
                )
        else:
            print(f"  FAIL", end="")
        print()

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print()
    print(f"{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")
    print()

    # Count wins, losses, ties for each variant vs ASHGF
    header = f"{'Function':<45s}" + "".join(f"{v:>14s}" for v in VARIANTS)
    print(header)
    print("-" * len(header))

    wins = {v: 0 for v in VARIANTS if v != "ASHGF"}
    better = {v: 0.0 for v in VARIANTS if v != "ASHGF"}  # sum of improvement
    total_valid = 0

    for fn in func_list:
        row = f"{fn:<45s}"
        for var_name in VARIANTS:
            entry = results[var_name].get(fn, {})
            best = entry.get("best", float("nan"))
            if np.isfinite(best):
                row += f"{best:>14.6e}"
            else:
                row += f"{'FAIL':>14s}"
        print(row)

        # Track wins
        ashgf_best = results["ASHGF"].get(fn, {}).get("best", float("nan"))
        if np.isfinite(ashgf_best):
            total_valid += 1
            for var_name in ["ASHGF-BT", "ASHGF-PID", "ASHGF-SOFT", "ASHGF-SMALPHA"]:
                var_best = results[var_name].get(fn, {}).get("best", float("nan"))
                if np.isfinite(var_best) and var_best < ashgf_best:
                    wins[var_name] += 1
                    better[var_name] += ashgf_best - var_best

    print()
    print(f"{'=' * 90}")
    print(f"  WIN/LOSS vs ASHGF baseline (out of {total_valid} valid functions)")
    print(f"{'=' * 90}")
    print(
        f"  {'Variant':<20s} {'Wins':>6s}  {'Losses':>6s}  {'Ties':>6s}  {'Σ Improvement':>15s}"
    )
    print(f"  {'-' * 65}")
    for var_name in ["ASHGF-BT", "ASHGF-PID", "ASHGF-SOFT", "ASHGF-SMALPHA"]:
        w = wins[var_name]
        # losses: functions where ASHGF is better
        losses = 0
        for fn in func_list:
            ashgf_best = results["ASHGF"].get(fn, {}).get("best", float("nan"))
            var_best = results[var_name].get(fn, {}).get("best", float("nan"))
            if (
                np.isfinite(ashgf_best)
                and np.isfinite(var_best)
                and ashgf_best < var_best
            ):
                losses += 1
        ties = total_valid - w - losses
        print(
            f"  {var_name:<20s} {w:>6d}  {losses:>6d}  {ties:>6d}  {better[var_name]:>15.6e}"
        )

    print()
    print("Note: 'Wins' = variant best < ASHGF best")
    print("      'Losses' = ASHGF best < variant best")
    print("      'Σ Improvement' = sum of (ASHGF_best - variant_best) over wins")


if __name__ == "__main__":
    main()
