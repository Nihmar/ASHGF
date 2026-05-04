"""CLI entry point for running ASHGF algorithms."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ashgf import __version__
from ashgf.algorithms import (
    ASEBO, ASGF, ASGFAQ, ASGFBW, ASGFCD, ASGFLS, ASGFRS, ASGFSS,
    ASHGF, ASHGFNG, ASHGFS, GD, SGES,
)
from ashgf.benchmark import (
    benchmark,
    benchmark_multi,
    plot_benchmark_comparison,
    plot_convergence_grid,
    plot_per_function,
    plot_statistics,
    print_benchmark_multi_summary,
    print_benchmark_summary,
    print_statistics_summary,
    statistics,
)
from ashgf.functions import get_function, list_functions
from ashgf.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _safe_plot(plot_fn, *args, **kwargs):
    """Call *plot_fn* swallowing any exception so plotting never crashes the run."""
    try:
        return plot_fn(*args, **kwargs)
    except Exception:
        logger.exception(
            "Plot call %s failed – continuing", getattr(plot_fn, "__name__", plot_fn)
        )
        return None


ALGORITHMS = {
    "gd": GD,
    "sges": SGES,
    "asgf": ASGF,
    "ashgf": ASHGF,
    "asebo": ASEBO,
    "ashgf-ng": ASHGFNG,
    "ashgf-s": ASHGFS,
    "asgf-rs": ASGFRS,
    "asgf-ls": ASGFLS,
    "asgf-cd": ASGFCD,
    "asgf-ss": ASGFSS,
    "asgf-aq": ASGFAQ,
    "asgf-bw": ASGFBW,
}


def _parse_dims(dims_str: str) -> list[int]:
    """Parse a comma-separated list of dimensions."""
    return [int(d.strip()) for d in dims_str.split(",")]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ashgf",
        description="Adaptive Stochastic Historical Gradient-Free optimization",
    )
    parser.add_argument("--version", action="version", version=f"ashgf {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run command ----
    run_parser = subparsers.add_parser(
        "run", help="Run an algorithm on a test function"
    )
    run_parser.add_argument(
        "--algo", choices=list(ALGORITHMS), default="gd", help="Algorithm to use"
    )
    run_parser.add_argument(
        "--function",
        required=True,
        help="Test function name (use 'list' to list all)",
    )
    run_parser.add_argument("--dim", type=int, default=100, help="Problem dimension")
    run_parser.add_argument(
        "--iter",
        type=int,
        default=1000,
        dest="max_iter",
        help="Number of iterations",
    )
    run_parser.add_argument("--seed", type=int, default=2003, help="Random seed")
    run_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (for GD, SGES, ASEBO)",
    )
    run_parser.add_argument(
        "--sigma", type=float, default=1e-4, help="Smoothing bandwidth"
    )
    run_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop if no improvement for N iterations",
    )
    run_parser.add_argument(
        "--ftol",
        type=float,
        default=None,
        help="Tolerance on f(x) change for stagnation (requires --patience)",
    )
    run_parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output"
    )

    # ---- compare command ----
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple algorithms"
    )
    compare_parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGORITHMS),
        default=["gd", "sges"],
        help="Algorithms to compare",
    )
    compare_parser.add_argument("--function", required=True, help="Test function name")
    compare_parser.add_argument(
        "--dim", type=int, default=100, help="Problem dimension"
    )
    compare_parser.add_argument("--iter", type=int, default=1000, dest="max_iter")
    compare_parser.add_argument("--seed", type=int, default=2003)
    compare_parser.add_argument("--patience", type=int, default=None)
    compare_parser.add_argument("--ftol", type=float, default=None)
    compare_parser.add_argument("--quiet", action="store_true")

    # ---- list command ----
    subparsers.add_parser("list", help="List all available test functions")

    # ---- benchmark command ----
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run all algorithms on all test functions (mass testing)",
    )
    bench_parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGORITHMS),
        default=None,
        help="Algorithms to include (default: all)",
    )
    bench_parser.add_argument(
        "--pattern",
        default=None,
        help="Only include functions matching this substring (case-insensitive)",
    )
    bench_parser.add_argument(
        "--dim", type=int, default=None, help="Single dimension (default: 100)"
    )
    bench_parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated dimensions, e.g. '10,100,1000' "
        "(overrides --dim, enables multi-dim benchmark)",
    )
    bench_parser.add_argument(
        "--iter",
        type=int,
        default=1000,
        dest="max_iter",
        help="Number of iterations per run",
    )
    bench_parser.add_argument("--seed", type=int, default=2003, help="Random seed")
    bench_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (for GD, SGES, ASEBO)",
    )
    bench_parser.add_argument(
        "--sigma", type=float, default=1e-4, help="Smoothing bandwidth"
    )
    bench_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for CSV results",
    )
    bench_parser.add_argument(
        "--plot",
        default=None,
        help="Save comparison bar chart to this file path",
    )
    bench_parser.add_argument(
        "--plot-convergence",
        default=None,
        dest="plot_conv",
        help="Save convergence grid plot to this file path",
    )
    bench_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop if no improvement for N iterations",
    )
    bench_parser.add_argument(
        "--ftol",
        type=float,
        default=None,
        help="Tolerance on f(x) change for stagnation",
    )
    bench_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    bench_parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-run output"
    )

    # ---- stats command ----
    stats_parser = subparsers.add_parser(
        "stats",
        help="Run multiple trials and compute convergence statistics",
    )
    stats_parser.add_argument("--function", required=True, help="Test function name")
    stats_parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGORITHMS),
        default=None,
        help="Algorithms to include (default: all)",
    )
    stats_parser.add_argument("--dim", type=int, default=100, help="Problem dimension")
    stats_parser.add_argument(
        "--iter",
        type=int,
        default=1000,
        dest="max_iter",
        help="Number of iterations per run",
    )
    stats_parser.add_argument(
        "--runs", type=int, default=30, help="Number of independent repetitions"
    )
    stats_parser.add_argument(
        "--seed",
        type=int,
        default=2003,
        help="Base random seed (each run uses seed + i)",
    )
    stats_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (for GD, SGES, ASEBO)",
    )
    stats_parser.add_argument(
        "--sigma", type=float, default=1e-4, help="Smoothing bandwidth"
    )
    stats_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for pickled results",
    )
    stats_parser.add_argument(
        "--plot",
        default=None,
        help="If set, save convergence plot to this file path",
    )
    stats_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    stats_parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-run output"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Check if a subcommand was provided
    if args.command is None:
        parser.print_help()
        return 1

    quiet = getattr(args, "quiet", False)
    if not quiet:
        configure_logging(level=logging.INFO)

    if args.command == "list":
        funcs = list_functions()
        print("Available test functions:")
        for name in funcs:
            print(f"  {name}")
        return 0

    if args.command in ("run", "compare"):
        f = get_function(args.function)

        if args.command == "run":
            algo_cls = ALGORITHMS[args.algo]
            algo_kwargs: dict = {"seed": args.seed}
            if args.algo in ("gd", "sges", "asebo"):
                algo_kwargs["lr"] = args.lr
                algo_kwargs["sigma"] = args.sigma
            elif args.algo == "ashgf-ng":
                algo_kwargs["n_jobs"] = 1
            algo = algo_cls(**algo_kwargs)

            best_vals, all_vals = algo.optimize(
                f,
                dim=args.dim,
                max_iter=args.max_iter,
                debug=not quiet,
                patience=args.patience,
                ftol=args.ftol,
            )
            print(f"Best value: {best_vals[-1][1]:.6e}")
            print(f"Iterations: {len(all_vals)}")
        else:
            for algo_name in args.algos:
                algo_cls = ALGORITHMS[algo_name]
                algo_kwargs = {"seed": args.seed}
                if algo_name in ("gd", "sges", "asebo"):
                    algo_kwargs["lr"] = args.lr
                    algo_kwargs["sigma"] = args.sigma
                elif algo_name == "ashgf-ng":
                    algo_kwargs["n_jobs"] = 1
                algo = algo_cls(**algo_kwargs)
                _, all_vals = algo.optimize(
                    f,
                    dim=args.dim,
                    max_iter=args.max_iter,
                    debug=False,
                    patience=args.patience,
                    ftol=args.ftol,
                )
                print(
                    f"{algo_name:>6}: final={all_vals[-1]:.6e}, "
                    f"best={min(all_vals):.6e}"
                )

        return 0

    if args.command == "benchmark":
        algos = args.algos
        if algos is not None:
            algos = [a.upper() for a in algos]

        # Determine whether single-dim or multi-dim
        output_dir = args.output or "results"

        if args.dims is not None:
            # Multi-dimension benchmark: one dim at a time,
            # with per-dimension plots produced immediately
            dims = _parse_dims(args.dims)
            all_results: dict[int, dict] = {}

            for dim in dims:
                dim_dir = os.path.join(output_dir, f"dim_{dim}")
                csv_dir = os.path.join(dim_dir, "csv")
                dim_results = benchmark(
                    algorithms=algos,
                    dim=dim,
                    max_iter=args.max_iter,
                    seed=args.seed,
                    lr=args.lr,
                    sigma=args.sigma,
                    output_dir=csv_dir,
                    debug=not quiet,
                    pattern=args.pattern,
                    patience=args.patience,
                    ftol=args.ftol,
                    n_jobs=args.jobs,
                )
                all_results[dim] = dim_results
                print_benchmark_summary(dim_results)

                wrapped = {dim: dim_results}

                # Per-dimension per-function plots
                per_func_dir = os.path.join(dim_dir, "per_function")
                saved = (
                    _safe_plot(plot_per_function, wrapped, output_dir=per_func_dir)
                    or []
                )

                # Per-dimension comparison bars
                bar_path = os.path.join(dim_dir, "comparison_bars.png")
                _safe_plot(
                    plot_benchmark_comparison, wrapped, output_path=bar_path, show=False
                )

                # Per-dimension convergence grid
                grid_path = os.path.join(dim_dir, "convergence_grid.png")
                _safe_plot(
                    plot_convergence_grid, wrapped, output_path=grid_path, show=False
                )

                print(f"  -> {len(saved)} plots + bars + grid saved in {dim_dir}/\n")

            # Cross-dimension summary
            print_benchmark_multi_summary(all_results)

            # Cross-dimension plots (only after all dims complete)
            bar_path = os.path.join(output_dir, "comparison_bars.png")
            _safe_plot(
                plot_benchmark_comparison, all_results, output_path=bar_path, show=False
            )

            grid_path = os.path.join(output_dir, "convergence_grid.png")
            _safe_plot(
                plot_convergence_grid, all_results, output_path=grid_path, show=False
            )

            print(f"\nCross-dimension plots saved in {output_dir}/")

            if args.plot:
                _safe_plot(
                    plot_benchmark_comparison,
                    all_results,
                    output_path=args.plot,
                    show=False,
                )
            if args.plot_conv:
                _safe_plot(
                    plot_convergence_grid,
                    all_results,
                    output_path=args.plot_conv,
                    show=False,
                )
        else:
            # Single-dimension benchmark
            dim = args.dim if args.dim is not None else 100
            results = benchmark(
                algorithms=algos,
                dim=dim,
                max_iter=args.max_iter,
                seed=args.seed,
                lr=args.lr,
                sigma=args.sigma,
                output_dir=output_dir,
                debug=not quiet,
                pattern=args.pattern,
                patience=args.patience,
                ftol=args.ftol,
                n_jobs=args.jobs,
            )
            print_benchmark_summary(results)

            # Wrap for plotting functions
            wrapped = {dim: results}

            # Auto-save comparison bar chart
            bar_path = os.path.join(output_dir, "comparison_bars.png")
            _safe_plot(
                plot_benchmark_comparison, wrapped, output_path=bar_path, show=False
            )

            # Auto-save one PNG PER FUNCTION (detailed convergence: dims x algos)
            per_func_dir = os.path.join(output_dir, "per_function")
            saved = (
                _safe_plot(plot_per_function, wrapped, output_dir=per_func_dir) or []
            )
            print(f"\nGenerated {len(saved)} per-function plots in {per_func_dir}/")

            if args.plot:
                _safe_plot(
                    plot_benchmark_comparison,
                    wrapped,
                    output_path=args.plot,
                    show=False,
                )

        return 0

    if args.command == "stats":
        algos = args.algos
        if algos is not None:
            algos = [a.upper() for a in algos]

        st = statistics(
            function=args.function,
            algorithms=algos,
            dim=args.dim,
            max_iter=args.max_iter,
            n_runs=args.runs,
            seed=args.seed,
            lr=args.lr,
            sigma=args.sigma,
            output_dir=args.output,
            debug=not quiet,
            patience=args.patience,
            ftol=args.ftol,
            n_jobs=args.jobs,
        )
        print_statistics_summary(st, args.function)

        if args.plot:
            _safe_plot(
                plot_statistics, st, args.function, output_path=args.plot, show=False
            )

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
