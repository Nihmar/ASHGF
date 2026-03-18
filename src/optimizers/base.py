import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union


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

    def _validate_x_init(
        self, x_init: Optional[Union[np.ndarray, List[float]]], dim: int
    ) -> np.ndarray:
        """
        Validate and convert x_init to a 1D numpy array of shape (dim,).

        Args:
            x_init: Initial point (can be None, list, or array).
            dim: Expected dimension.

        Returns:
            1D numpy array of shape (dim,).

        Raises:
            TypeError: If x_init cannot be converted to array.
            ValueError: If x_init has wrong dimension.
        """
        if x_init is None:
            return np.random.randn(dim)

        try:
            x = np.asarray(x_init, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"x_init must be array-like, got {type(x_init).__name__}") from e

        x = np.atleast_1d(x).ravel()

        if x.shape[0] != dim:
            raise ValueError(
                f"x_init has shape {x.shape}, expected ({dim},)"
            )

        return x

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
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
