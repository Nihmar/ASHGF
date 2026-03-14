"""
Statistical plotting script for optimization algorithm comparison.

This script loads results from Parquet files and generates convergence plots.
"""

import argparse
import os
import warnings
from os import path
from typing import Dict, List, Optional

# Ignore the specific deprecation warning from multiprocessing.forkserver
warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FUNCTIONS: List[str] = ["sphere", "levy", "rastrigin", "ackley"]
DEFAULT_DIM: int = 100
DEFAULT_ALGORITHMS: List[str] = ["GD", "SGES", "ASGF", "ASHGF", "ASEBO"]


def get_results_path(dim: int) -> str:
    """Get path to results Parquet file for a given dimension."""
    return os.path.join("results", "profiles", f"dim={dim}", "results.parquet")


def load_results(dim: int) -> pd.DataFrame:
    """
    Load results for a given dimension from Parquet.

    Args:
        dim: Dimension to load.

    Returns:
        DataFrame with columns: function, algorithm, run, values
    """
    results_path = get_results_path(dim)
    if not path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    return pd.read_parquet(results_path)


def filter_results(
    df: pd.DataFrame,
    functions: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter results by functions and algorithms."""
    result = df
    if functions:
        result = result[result["function"].isin(functions)]
    if algorithms:
        result = result[result["algorithm"].isin(algorithms)]
    return result


def compute_statistics(values_list: List[List[float]]) -> Dict[str, np.ndarray]:
    """
    Compute statistics across multiple runs.

    Args:
        values_list: List of convergence curves.

    Returns:
        Dictionary with min, max, mean, std arrays.
    """
    min_len = max(len(v) for v in values_list) if values_list else 0

    padded = []
    for v in values_list:
        if len(v) < min_len:
            v = v + [np.nan] * (min_len - len(v))
        padded.append(v)

    arr = np.array(padded)

    return {
        "min": np.nanmin(arr, axis=0),
        "max": np.nanmax(arr, axis=0),
        "mean": np.nanmean(arr, axis=0),
        "std": np.nanstd(arr, axis=0),
    }


def plot_convergence_all_runs(
    df: pd.DataFrame,
    function: str,
    algorithm: str,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot individual convergence curves for all runs."""
    subset = df[(df["function"] == function) & (df["algorithm"] == algorithm)]

    if subset.empty:
        print(f"No data for {function} - {algorithm}")
        return

    plt.figure(figsize=(10, 6))
    for _, row in subset.iterrows():
        plt.plot(row["values"], alpha=0.5, label=f"run_{row['run']}")

    plt.yscale("log")
    plt.title(f"{function} - {algorithm}")
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"$f(x_t)$")
    plt.savefig(output_path, dpi=600)
    if show:
        plt.show()
    plt.close()
    print(f"Saved {output_path}")


def plot_convergence_with_stats(
    df: pd.DataFrame,
    function: str,
    algorithm: str,
    output_path: str,
    show: bool = False,
) -> None:
    """Plot mean convergence curve with min/max bounds and std shading."""
    subset = df[(df["function"] == function) & (df["algorithm"] == algorithm)]

    if subset.empty:
        print(f"No data for {function} - {algorithm}")
        return

    values_list = subset["values"].tolist()
    stats = compute_statistics(values_list)

    x = np.arange(len(stats["mean"]))

    plt.figure(figsize=(10, 6))
    plt.plot(x, stats["min"], linestyle="--", alpha=0.7, label="min")
    plt.plot(x, stats["max"], linestyle="--", alpha=0.7, label="max")
    plt.plot(x, stats["mean"], linewidth=2, label="mean")
    plt.fill_between(
        x,
        np.maximum(stats["min"], stats["mean"] - stats["std"]),
        np.minimum(stats["max"], stats["mean"] + stats["std"]),
        alpha=0.3,
        label="mean ± std",
    )
    plt.yscale("log")
    plt.legend()
    plt.title(f"{function} - {algorithm}")
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"$f(x_t)$")
    plt.savefig(output_path, dpi=600)
    if show:
        plt.show()
    plt.close()
    print(f"Saved {output_path}")


def plot_all_algorithms(
    df: pd.DataFrame,
    function: str,
    output_dir: str,
    show: bool = False,
) -> None:
    """Plot comparison of all algorithms for a given function."""
    algorithms = df[df["function"] == function]["algorithm"].unique()

    plt.figure(figsize=(12, 8))

    for algorithm in algorithms:
        subset = df[(df["function"] == function) & (df["algorithm"] == algorithm)]
        values_list = subset["values"].tolist()
        if values_list:
            stats = compute_statistics(values_list)
            plt.plot(stats["mean"], label=algorithm, linewidth=2)

    plt.yscale("log")
    plt.legend()
    plt.title(f"{function} - All Algorithms")
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"$f(x_t)$")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{function}_comparison.png")
    plt.savefig(output_path, dpi=600)
    if show:
        plt.show()
    plt.close()
    print(f"Saved {output_path}")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table."""
    summary = []

    for (func, alg), group in df.groupby(["function", "algorithm"]):
        values_list = group["values"].tolist()
        if not values_list:
            continue

        final_values = [v[-1] if len(v) > 0 else np.nan for v in values_list]
        best_values = [min(v) if len(v) > 0 else np.nan for v in values_list]

        summary.append(
            {
                "function": func,
                "algorithm": alg,
                "n_runs": len(values_list),
                "mean_final": np.mean(final_values),
                "std_final": np.std(final_values),
                "min_final": np.min(final_values),
                "mean_best": np.mean(best_values),
            }
        )

    return pd.DataFrame(summary)


def main() -> None:
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Generate convergence plots from Parquet results."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"Dimension (default: {DEFAULT_DIM})",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        default=None,
        help="Benchmark functions (default: all functions in data)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help=f"Algorithms to plot (default: {' '.join(DEFAULT_ALGORITHMS)})",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--plot-comparison",
        action="store_true",
        help="Plot all algorithms comparison",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary table")

    args = parser.parse_args()

    try:
        df = load_results(args.dim)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Run profiles.py first to generate results.")
        return

    # Get all functions from data if not specified
    all_functions = sorted(df["function"].unique().tolist())
    if args.functions is None:
        args.functions = all_functions
        print(
            f"No functions specified, using all {len(all_functions)} functions from data"
        )
    else:
        # Filter to only existing functions
        args.functions = [f for f in args.functions if f in all_functions]
        if not args.functions:
            print(f"No valid functions found. Available: {all_functions}")
            return

    df = filter_results(df, args.functions, args.algorithms)

    if df.empty:
        print("No data found matching criteria.")
        return

    print(f"Loaded {len(df)} records for dim={args.dim}")
    print(f"Functions: {args.functions}")
    print(f"Algorithms: {args.algorithms}")

    base_output_dir = os.path.join("results", "plots", f"dim={args.dim}")

    # Generate plots for each function in its own folder
    for function in args.functions:
        func_dir = os.path.join(base_output_dir, function)
        os.makedirs(func_dir, exist_ok=True)

        func_df = df[df["function"] == function]
        print(f"\nProcessing: {function} ({len(func_df)} records)")

        # Plot each algorithm
        for algorithm in args.algorithms:
            alg_df = func_df[func_df["algorithm"] == algorithm]
            if alg_df.empty:
                print(f"  No data for {algorithm}")
                continue

            conv_path = os.path.join(func_dir, f"{algorithm}_convergence.png")
            if not path.exists(conv_path):
                plot_convergence_with_stats(df, function, algorithm, conv_path, False)
                print(f"  Saved: {algorithm}_convergence.png")

        # Plot algorithm comparison
        if args.plot_comparison:
            comp_path = os.path.join(func_dir, "comparison.png")
            if not path.exists(comp_path):
                plot_all_algorithms(df, function, func_dir, False)
                print(f"  Saved: comparison.png")

    # Generate summary table
    if args.summary:
        summary = generate_summary_table(df)
        summary.to_csv(os.path.join(base_output_dir, "summary.csv"), index=False)
        print(f"\nSaved summary to {base_output_dir}/summary.csv")

    print(f"\nAll plots saved to: {base_output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
