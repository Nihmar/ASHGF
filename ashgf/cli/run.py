"""CLI entry point for running ASHGF algorithms."""

from __future__ import annotations

import argparse
import logging
import sys

from ashgf import __version__
from ashgf.algorithms import ASEBO, ASGF, ASHGF, GD, SGES
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
