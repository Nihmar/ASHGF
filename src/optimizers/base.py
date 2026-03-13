import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from functions import Function


class BaseOptimizer:
    """
    Base class for optimization algorithms.

    Attributes:
        kind (str): Name of the algorithm.
        seed (int): Random seed for reproducibility.
        eps (float): Convergence threshold.
    """

    kind: str = "Base Optimizer"

    def __init__(self, seed: int = 2003, eps: float = 1e-8):
        self.seed = seed
        self.eps = eps

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[np.ndarray] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        """
        Optimize the given function.

        Args:
            function: Name of the function to optimize.
            dim: Dimension of the domain.
            it: Number of iterations.
            x_init: Initial point.
            debug: Whether to print debug information.
            itprint: Frequency of debug prints.

        Returns:
            Tuple of (best_values, all_values).
        """
        raise NotImplementedError

    def plot_results(self, values: List, function: str) -> None:
        """
        Plot the optimization results.

        Args:
            values: List of values to plot.
            function: Name of the function.
        """
        plt.plot(values, "r", label=self.kind)
        plt.yscale("log")
        plt.legend()
        plt.title(function)
        plt.show()
