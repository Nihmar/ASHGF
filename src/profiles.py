import os
from os import path
from typing import Dict, List

import pandas as pd

from optimizers.gd import GD
from optimizers.sges import SGES
from optimizers.asebo import ASEBO
from optimizers.asgf import ASGF
from optimizers.ashgf import ASHGF

# Note: The original code referenced 'code/results' relative to the script location.
# Since this script is in 'src/', we adjust the path to be relative to the project root.
if path.exists(path.join('..', 'results')):
    main_folder = path.join('..', 'results', 'profiles')
else:
    main_folder = path.join('results', 'profiles')
debug_it = 100
debug = False
tau = 10**(-3)
it = 10000
dims = [10, 100, 1000]
algorithms: Dict[str, type] = {
    "GD": GD,
    "SGES": SGES,
    "ASGF": ASGF,
    "ASHGF": ASHGF,
    "ASEBO": ASEBO
}


def get_functions() -> List[str]:
    """
    Get list of function names from functions.txt.

    Returns:
        List of function names.
    """
    lines = []
    # Adjust path to be relative to the project root
    file_path = path.join('..', 'functions.txt')
    if not path.exists(file_path):
        # Fallback to src directory
        file_path = path.join('src', 'functions.txt')
    if not path.exists(file_path):
        # Fallback to current directory
        file_path = 'functions.txt'
    
    with open(file_path) as f:
        lines = f.readlines()

    fs = []
    for line in lines:
        if '*' in line:
            fs.append(line.replace('*', '').strip())

    return fs


def create_folders() -> None:
    """
    Create necessary folders for storing results.
    """
    functions = get_functions()

    if not path.exists(main_folder):
        os.makedirs(main_folder)

    for dim in dims:
        dim_folder = path.join(main_folder, str(dim))
        if not path.exists(dim_folder):
            os.mkdir(dim_folder)

        for function in functions:
            func_folder = path.join(dim_folder, function)
            if not path.exists(func_folder):
                os.mkdir(func_folder)

            for algorithm in algorithms.keys():
                algo_folder = path.join(func_folder, algorithm)
                if not path.exists(algo_folder):
                    os.mkdir(algo_folder)


def evaluations(it: int, dim: int, name: str) -> int:
    """
    Calculate the number of function evaluations for an algorithm.

    Args:
        it: Number of iterations.
        dim: Dimension of the problem.
        name: Name of the algorithm.

    Returns:
        Number of function evaluations.
    """
    correction = 1

    if name in ["ASGF", "ASHGF"]:
        return it + it * dim * 4
    else:
        return (it + it * dim * 2) * correction


def iterations(evs: int, dim: int, name: str) -> int:
    """
    Calculate the number of iterations from function evaluations.

    Args:
        evs: Number of function evaluations.
        dim: Dimension of the problem.
        name: Name of the algorithm.

    Returns:
        Number of iterations.
    """
    if name in ["ASGF", "ASHGF"]:
        return int(evs / (1 + dim * 4))
    else:
        return int(evs / (1 + dim * 2))


def execute_algorithm(function: str, dim: int, name: str, algorithm: type):
    """
    Execute an optimization algorithm on a function.

    Args:
        function: Name of the function to optimize.
        dim: Dimension of the problem.
        name: Name of the algorithm.
        algorithm: Algorithm class.

    Returns:
        Tuple of (best_values, all_values).
    """
    if name not in ["ASGF", "ASHGF"]:
        alg = algorithm(1e-4, 1e-4)
    else:
        alg = algorithm()
    alg_best, alg_all = alg.optimize(function, dim, it, None, debug, debug_it)

    return alg_best, alg_all


def save_data(function: str, dim: int, name: str, algorithm: type) -> None:
    """
    Save optimization results to CSV.

    Args:
        function: Name of the function.
        dim: Dimension of the problem.
        name: Name of the algorithm.
        algorithm: Algorithm class.
    """
    alg_best, alg_all = execute_algorithm(function, dim, name, algorithm)

    series = pd.Series(alg_all, name="descent")
    results_path = path.join("results", "profiles", str(dim), function, name, "descent.csv")
    series.to_csv(results_path, header=True, index=False)


def main() -> None:
    """
    Main function to run performance profiles.
    """
    fs = get_functions()
    create_folders()

    for dim in dims:
        for name, algorithm in algorithms.items():
            for function in fs:
                # Check if results file already exists
                result_file = path.join(main_folder, str(dim), function, name, "descent.csv")
                if not path.exists(result_file):
                    save_data(function, dim, name, algorithm)

                print(f"Done with: {name} - {function} - {dim}")


if __name__ == "__main__":
    main()