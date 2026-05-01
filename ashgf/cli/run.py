"""CLI entry point for running ASHGF algorithms."""

from __future__ import annotations

import argparse
import logging
import sys

from ashgf import __version__
from ashgf.algorithms import ASEBO, ASGF, ASHGF, GD, SGES
from ashgf.benchmark import (
    benchmark,
    plot_statistics,
    print_benchmark_summary,
    print_statistics_summary,
    statistics,
)
from ashgf.functions import get_function, list_functions
from ashgf.utils.logging import configure_logging

logger = logging.getLogger(__name__)

ALGORITHMS = {
    "gd": GD,
    "sges": SGES,
    "asgf": ASGF,
    "ashgf": ASHGF,
    "asebo": ASEBO,
}


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
    bench_parser.add_argument("--dim", type=int, default=100, help="Problem dimension")
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
            algo = algo_cls(**algo_kwargs)

            best_vals, all_vals = algo.optimize(
                f, dim=args.dim, max_iter=args.max_iter, debug=not quiet
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
                algo = algo_cls(**algo_kwargs)
                _, all_vals = algo.optimize(
                    f, dim=args.dim, max_iter=args.max_iter, debug=False
                )
                print(
                    f"{algo_name:>6}: final={all_vals[-1]:.6e}, "
                    f"best={min(all_vals):.6e}"
                )

        return 0

    if args.command == "benchmark":
        algos = args.algos  # may be None → default to all
        if algos is not None:
            # Convert CLI keys (gd, sges, ...) to benchmark keys (GD, SGES, ...)
            algos = [a.upper() for a in algos]

        results = benchmark(
            algorithms=algos,
            dim=args.dim,
            max_iter=args.max_iter,
            seed=args.seed,
            lr=args.lr,
            sigma=args.sigma,
            output_dir=args.output,
            debug=not quiet,
            pattern=args.pattern,
        )
        print_benchmark_summary(results)
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
        )
        print_statistics_summary(st, args.function)

        if args.plot:
            plot_statistics(st, args.function, output_path=args.plot, show=False)

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
