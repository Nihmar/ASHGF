import os
import pickle
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from optimizers.gd import GD
from optimizers.sges import SGES
from optimizers.asebo import ASEBO
from optimizers.asgf import ASGF
from optimizers.ashgf import ASHGF

# Note: The original code referenced 'code/results' relative to the script location.
# Since this script is in 'src/', we adjust the path to be relative to the project root.
np.random.seed(0)

it = 10000
debug_it = 100
debug = True
dim = 100
x_0 = np.random.randn(dim)

functions: List[str] = ["sphere", "levy", "rastrigin", "ackley"]

algorithms: Dict[str, type] = {
    "GD": GD,
    "SGES": SGES,
    "ASGF": ASGF,
    "ASHGF": ASHGF,
    "ASEBO": ASEBO
}

for algorithm in algorithms:
    for function in functions:
        bests = []

        # Adjust path to be relative to the project root
        results_dir = os.path.join("..", "results", "stats", algorithm)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        pkl_file = os.path.join(results_dir, f"descents_{function}.pkl")

        if not os.path.exists(pkl_file):
            for i in range(100):
                print(i)
                r_seed = np.random.randint(0, 10000)
                alg = algorithms[algorithm](seed=r_seed)
                alg_best, alg_all = alg.optimize(function, dim, it, x_0, debug, debug_it)
                bests.append(alg_all)

            with open(pkl_file, "wb") as output:
                pickle.dump(bests, output)

        print(f"Finished {function} for algorithm {algorithm}")


for algorithm in algorithms:
    for function in functions:
        # Adjust path to be relative to the project root
        results_dir = os.path.join("..", "results", "stats", algorithm)
        pkl_file = os.path.join(results_dir, f"descents_{function}.pkl")
        
        convergence_plot = os.path.join(results_dir, f"{function}_convergence.png")
        convergence_mean_plot = os.path.join(results_dir, f"{function}_convergence_mean.png")

        if not os.path.exists(convergence_plot):
            with open(pkl_file, "rb") as f:
                bests = pickle.load(f)

            for i, best in enumerate(bests):
                plt.plot(best, label=f"{algorithm}_{i}")

            plt.yscale("log")
            plt.title(function)
            plt.xlabel(r"Iterations $t$")
            plt.ylabel(r"$f(x_t)$")
            plt.savefig(convergence_plot, dpi=600)
            plt.show()

            min_descent = []
            max_descent = []
            mean_descent = []
            std_plus_descent = []
            std_minus_descent = []

            min_num_it = int(np.max([len(best) for best in bests]))

            for i in range(min_num_it):
                values = [best[i] if i < len(best) else np.nan for best in bests]
                mean_value = np.nanmean(values)
                min_value = np.nanmin(values)
                max_value = np.nanmax(values)
                std_value = np.nanstd(values)

                min_descent.append(min_value)
                max_descent.append(max_value)
                mean_descent.append(mean_value)
                std_plus_descent.append(mean_value + std_value)
                std_minus_descent.append(max([min_value, mean_value - std_value]))

            plt.figure()
            plt.plot(min_descent, label="min")
            plt.plot(max_descent, label="max")
            plt.plot(mean_descent, label="mean")
            plt.fill_between(range(min_num_it),
                            std_minus_descent,
                            std_plus_descent,
                            alpha=0.5)
            plt.yscale("log")
            plt.legend()
            plt.title(function)
            plt.xlabel(r"Iterations $t$")
            plt.ylabel(r"$f(x_t)$")
            plt.savefig(convergence_mean_plot, dpi=600)
            plt.show()

        print(f"Saved images for {function}")
