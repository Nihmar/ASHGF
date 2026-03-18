"""
Performance profile runner for optimization algorithms.

Results are stored in Parquet format for efficiency.

Usage:
    python profiles.py --n-runs 10 --workers 4
"""

import argparse
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from os import path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from optimizers.gd import GD
from optimizers.sges import SGES
from optimizers.asebo import ASEBO
from optimizers.asgf import ASGF
from optimizers.ashgf import ASHGF


# Determine project root (parent of src/)
_SCRIPT_DIR = path.dirname(__file__)
if path.basename(_SCRIPT_DIR) == "src":
    PROJECT_ROOT = path.dirname(_SCRIPT_DIR)
else:
    PROJECT_ROOT = _SCRIPT_DIR

warnings.filterwarnings("default")
# Ignore the specific deprecation warning from multiprocessing.forkserver
warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")

RESULTS_DIR = path.join(PROJECT_ROOT, "results", "profiles")
LOG_DIR = path.join(PROJECT_ROOT, "results", "logs")


def setup_logging(log_file: str = "experiments.log") -> logging.Logger:
    """Setup logging to file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = path.join(LOG_DIR, log_file)

    logger = logging.getLogger("experiments")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


debug_it = 100
debug = False
tau = 10 ** (-3)
it = 10000
dims = [10, 100, 1000]
algorithms: Dict[str, type] = {
    "GD": GD,
    "SGES": SGES,
    "ASGF": ASGF,
    "ASHGF": ASHGF,
    "ASEBO": ASEBO,
}


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    function: str
    algorithm: str
    run: int
    values: List[float]
    status: str
    error_msg: Optional[str] = None
    warnings: Optional[str] = None


def get_functions() -> List[str]:
    """Get list of function names from functions.txt."""
    file_path = path.join("src", "functions.txt")
    if not path.exists(file_path):
        file_path = "functions.txt"

    with open(file_path) as f:
        lines = f.readlines()

    fs = []
    for line in lines:
        if "*" in line:
            fs.append(line.replace("*", "").strip())

    return fs


def get_parquet_path(dim: int) -> str:
    """Get path to the Parquet file for a given dimension."""
    return path.join(RESULTS_DIR, f"dim={dim}", "results.parquet")


def load_results(dim: int) -> pd.DataFrame:
    """Load results for a given dimension from Parquet."""
    parquet_path = get_parquet_path(dim)
    if path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame(columns=["function", "algorithm", "run", "values"])


def save_results(df: pd.DataFrame, dim: int) -> None:
    """Save results DataFrame to Parquet file."""
    output_path = get_parquet_path(dim)
    os.makedirs(path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def create_folders() -> None:
    """Create necessary folders for storing results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_experiment(
    function: str,
    algorithm_name: str,
    dim: int,
    run: int,
    seed: int,
) -> ExperimentResult:
    """
    Run a single experiment. This function runs in a separate process.

    Args:
        function: Name of the function.
        algorithm_name: Name of the algorithm.
        dim: Problem dimension.
        run: Run number.
        seed: Random seed.

    Returns:
        ExperimentResult with values or error info.
    """
    np.seterr(divide="ignore", over="ignore", invalid="ignore")

    try:
        algorithm = algorithms[algorithm_name]
        x_0 = np.random.default_rng(seed).standard_normal(dim)

        if algorithm_name not in ["ASGF", "ASHGF"]:
            alg = algorithm(1e-4, 1e-4)
        else:
            alg = algorithm()

        warning_msgs = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg_best, alg_all = alg.optimize(function, dim, it, x_0, debug, debug_it)

            if w:
                warning_msgs = [
                    f"{warning.category.__name__}: {warning.message}" for warning in w
                ]

        return ExperimentResult(
            function=function,
            algorithm=algorithm_name,
            run=run,
            values=alg_all,
            status="success",
            warnings="\n".join(warning_msgs) if warning_msgs else None,
        )

    except Exception as e:
        return ExperimentResult(
            function=function,
            algorithm=algorithm_name,
            run=run,
            values=[],
            status="failed",
            error_msg=str(e),
        )


def get_tasks(
    dim: int,
    n_runs: int,
    seed: int,
    overwrite: bool,
    algorithms_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, int, int, int]]:
    """Generate list of experiments to run."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    fs = get_functions()
    results = load_results(dim)

    # Filter algorithms if specified
    alg_names = algorithms.keys() if algorithms_filter is None else algorithms_filter

    tasks = []
    for name in alg_names:
        if name not in algorithms:
            print(f"Warning: Unknown algorithm '{name}', skipping")
            continue
        for function in fs:
            existing = results[
                (results["function"] == function) & (results["algorithm"] == name)
            ]
            existing_runs = set(existing["run"].tolist() if len(existing) > 0 else [])

            for run in range(n_runs):
                if not overwrite and run in existing_runs:
                    continue

                task_seed = int(rng.integers(0, 10000))
                tasks.append((function, name, dim, run, task_seed))

    return tasks


def run_experiments(
    dim: int,
    n_runs: int = 10,
    seed: int = 0,
    overwrite: bool = False,
    n_workers: int = 4,
    verbose: bool = True,
    batch_size: int = 20,
    logger: Optional[logging.Logger] = None,
    algorithms_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run experiments for a single dimension using parallel processing.

    Args:
        dim: Problem dimension.
        n_runs: Number of runs per algorithm/function.
        seed: Random seed.
        overwrite: Whether to overwrite existing results.
        n_workers: Number of parallel workers.
        verbose: Print progress.
        batch_size: Save results every N experiments.
        algorithms_filter: List of algorithms to run (None = all).

    Returns:
        DataFrame with all results.
    """
    tasks = get_tasks(dim, n_runs, seed, overwrite, algorithms_filter)

    if not tasks:
        print(f"All experiments already completed for dim={dim}")
        return load_results(dim)

    print(f"Running {len(tasks)} experiments for dim={dim} with {n_workers} workers...")

    results = load_results(dim)
    completed = 0
    failed = 0
    pending_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, *task): task for task in tasks
        }

        with tqdm(total=len(tasks), desc=f"dim={dim}", disable=not verbose) as pbar:
            for future in as_completed(futures):
                task = futures[future]
                function, algorithm_name, _, run, _ = task

                try:
                    result = future.result()

                    if result.status == "success":
                        pending_results.append(
                            {
                                "function": result.function,
                                "algorithm": result.algorithm,
                                "run": result.run,
                                "values": result.values,
                                "warnings": result.warnings,
                            }
                        )

                        msg = f"[{algorithm_name}] {function} run {run}: OK"
                        if result.warnings:
                            msg += f" ({len(result.warnings.split(chr(10)))} warnings)"
                            if logger:
                                logger.warning(
                                    f"dim={dim} - {msg}\n  {result.warnings}"
                                )
                        if verbose:
                            pbar.write(msg)
                    else:
                        failed += 1
                        msg = f"[{algorithm_name}] {function} run {run}: FAILED - {result.error_msg}"
                        if verbose:
                            pbar.write(msg)
                        if logger:
                            logger.error(f"dim={dim} - {msg}")

                except Exception as e:
                    failed += 1
                    msg = f"[{algorithm_name}] {function} run {run}: ERROR - {e}"
                    if verbose:
                        pbar.write(msg)
                    if logger:
                        logger.error(f"dim={dim} - {msg}")

                completed += 1
                pbar.update(1)

                if len(pending_results) >= batch_size:
                    results = pd.concat(
                        [results, pd.DataFrame(pending_results)], ignore_index=True
                    )
                    save_results(results, dim)
                    pending_results = []

    if pending_results:
        results = pd.concat([results, pd.DataFrame(pending_results)], ignore_index=True)

    save_results(results, dim)
    print(f"Completed: {completed - failed} success, {failed} failed")
    return results


def run_all_experiments(
    n_runs: int = 10,
    seed: int = 0,
    overwrite: bool = False,
    dims_to_run: Optional[List[int]] = None,
    n_workers: int = 4,
    logger: Optional[logging.Logger] = None,
    algorithms_filter: Optional[List[str]] = None,
) -> None:
    """Run all experiments across all dimensions."""
    create_folders()

    if dims_to_run is None:
        dims_to_run = dims

    for dim in dims_to_run:
        print(f"\n{'=' * 50}")
        print(f"Running experiments for dim={dim}")
        alg_msg = f" ({algorithms_filter})" if algorithms_filter else ""
        print(f"{'=' * 50}{alg_msg}\n")
        run_experiments(dim, n_runs, seed, overwrite, n_workers, logger=logger, algorithms_filter=algorithms_filter)


def analyze_results(dim: int) -> pd.DataFrame:
    """Analyze results for a given dimension."""
    df = load_results(dim)

    if df.empty:
        print(f"No results found for dim={dim}")
        return df

    summary = []
    for (func, alg), group in df.groupby(["function", "algorithm"]):
        all_values = group["values"].tolist()
        final_values = [v[-1] if len(v) > 0 else np.nan for v in all_values]
        min_values = [np.min(v) if len(v) > 0 else np.nan for v in all_values]

        summary.append(
            {
                "function": func,
                "algorithm": alg,
                "n_runs": len(all_values),
                "mean_final": np.mean(final_values),
                "std_final": np.std(final_values),
                "min_final": np.min(final_values),
                "max_final": np.max(final_values),
                "mean_best": np.mean(min_values),
            }
        )

    return pd.DataFrame(summary)


def main() -> None:
    """Main function to run performance profiles."""
    parser = argparse.ArgumentParser(
        description="Run performance profiles with parallel execution"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs per experiment (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=None,
        help="Specific dimensions to run (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze and print summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Specific algorithms to run (default: all)",
    )

    args = parser.parse_args()

    logger = (
        setup_logging(f"dim={args.dims[0] if args.dims else 'all'}.log")
        if not args.analyze
        else None
    )

    if args.analyze:
        for dim in dims:
            print(f"\n{'=' * 50}")
            print(f"Summary for dim={dim}")
            print(f"{'=' * 50}")
            summary = analyze_results(dim)
            if not summary.empty:
                print(summary.to_string(index=False))
    else:
        run_all_experiments(
            n_runs=args.n_runs,
            seed=args.seed,
            overwrite=args.overwrite,
            dims_to_run=args.dims,
            n_workers=args.workers,
            logger=logger,
            algorithms_filter=args.algorithms,
        )


if __name__ == "__main__":
    main()
