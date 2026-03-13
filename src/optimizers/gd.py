import numpy as np
import numpy.linalg as la

from functions import Function
from optimizers.base import BaseOptimizer


class GD(BaseOptimizer):
    kind = "Vanilla Gradient Descent"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        seed: int = 2003,
        eps: float = 1e-8,
    ):
        super().__init__(seed, eps)
        if lr < 0:
            raise ValueError("Error: learning rate < 0")
        if sigma < 0:
            raise ValueError("Error: sigma < 0")

        self.lr = lr
        self.sigma = sigma

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: np.ndarray = None,
        debug: bool = True,
        itprint: int = 25,
    ):
        """
        Principal method of the class.
        It optimizes the given function and returns the sequence of values found by the algorithm.

        Args:
            function: Name of the function to optimize.
            dim: Dimension of the domain of the function.
            it: Number of iterations of the algorithm.
            x_init: Initial point of the algorithm.
            debug: True if debug prints are wanted.
            itprint: Iterations to wait before mid-execution print.

        Returns:
            Tuple of (best_values, all_values).
            best_values: list containing the best values found during the execution, in descending order.
            all_values: list containing all the values found during the execution.
        """
        np.random.seed(self.seed)

        f = Function(function)

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        if debug:
            print("algorithm:", "gd", "function:", function, "dimension:", len(x), "initial value:", steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, "th iteration - value:", steps[i - 1][1], "last best value:", best_value)

                grad = self.grad_estimator(x, f)

                x = x - self.lr * grad
                steps[i] = [x, f.evaluate(x)]
                if steps[i][1] < best_value:
                    best_value = steps[i][1]
                    best_values.append([steps[i][0], best_value])

                if la.norm(x - steps[i - 1][0]) < self.eps:
                    break

            except Exception as e:
                print("Something has gone wrong!")
                print(e)
                break

        if debug:
            print()
            try:
                print("last evaluation:", steps[i][1], "last_iterate:", i, "best evaluation:", best_value)
                print()
            except:
                print("last evaluation:", steps[i - 1][1], "last_iterate:", i - 1, "best evaluation:", best_value)
                print()

        return best_values, [steps[j][1] for j in range(i)]

    def grad_estimator(self, x: np.ndarray, f: Function) -> np.ndarray:
        """
        Performs the Central Gaussian Smoothing around x for the function f.

        Args:
            x: Point on which the operation is performed.
            f: Function object over which the smoothing is performed.

        Returns:
            The estimated gradient.
        """
        dim = len(x)

        grad = np.zeros(dim, )
        directions = self.compute_directions(dim, dim)

        evaluations = []

        for i in range(dim):
            d = directions[i].reshape(x.shape)
            dir_plus = x + self.sigma * d
            dir_minus = x - self.sigma * d

            evaluations_plus = f.evaluate(dir_plus)
            evaluations_minus = f.evaluate(dir_minus)

            evaluations.append(evaluations_plus)
            evaluations.append(evaluations_minus)

            grad = grad + (evaluations_plus - evaluations_minus) * d.reshape(grad.shape)

        grad = grad / (2 * self.sigma * dim)

        return grad

    def compute_directions(self, dim_1: int, dim_2: int) -> np.ndarray:
        """
        Compute a matrix of random directions.

        Args:
            dim_1: Number of rows.
            dim_2: Number of columns.

        Returns:
            Matrix of directions.
        """
        return np.random.randn(dim_1, dim_2)
