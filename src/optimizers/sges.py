import numpy as np
import numpy.linalg as la

from functions import Function
from optimizers.base import BaseOptimizer


class SGES(BaseOptimizer):
    kind = "Self-Guided Evolution Strategies"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        k: int = 50,
        k1: float = 0.9,
        k2: float = 0.1,
        alpha: float = 0.5,
        delta: float = 1.1,
        t: int = 50,
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
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.delta = delta
        self.t = t

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
        alpha = self.alpha

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        G = []

        if debug:
            print("algorithm:", "sges", "function:", function, "dimension:", len(x), "initial value:", steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, "th iteration - value:", steps[i - 1][1], "last best value:", best_value)

                if i < self.t:
                    grad = self.grad_estimator(x, f)
                    G.append(grad)
                else:
                    grad, evaluations, M = self.grad_estimator(x, f, G, True, alpha)
                    G.append(grad)
                    G = G[1:]

                    try:
                        r = (1 / M) * np.sum([min([evaluations[2 * j], evaluations[2 * j + 1]]) for j in range(M)])
                    except:
                        r = False

                    try:
                        r_hat = (1 / (dim - M)) * np.sum([min([evaluations[2 * j], evaluations[2 * j + 1]]) for j in range(M, dim)])
                    except:
                        r_hat = False

                    if not r or r < r_hat:
                        alpha = min([self.delta * alpha, self.k1])
                    elif not r_hat or r >= r_hat:
                        alpha = max([(1 / self.delta) * alpha, self.k2])
                    else:
                        pass

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

    def grad_estimator(self, x: np.ndarray, f: Function, G: np.ndarray = None, sges: bool = False, alpha: float = 0) -> np.ndarray:
        """
        Performs the Central Gaussian Smoothing around x for the function f.

        Args:
            x: Point on which the operation is performed.
            f: Function object over which the smoothing is performed.
            G: Buffer of gradients, 2-D array.
            sges: True if to compute special directions of SGES.
            alpha: Probability of sampling from gradients.

        Returns:
            The estimated gradient.
        """
        np.random.seed(self.seed)

        dim = len(x)

        grad = np.zeros(dim, )

        if sges:
            directions, M = self.compute_directions_sges(dim, G, alpha)
        directions = self.compute_directions(dim)

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

        if sges:
            return grad, evaluations, M
        else:
            return grad

    def compute_directions(self, dim: int) -> np.ndarray:
        """
        Compute a matrix of random directions.

        Args:
            dim: Number of rows and columns.

        Returns:
            Matrix of directions.
        """
        return np.random.randn(dim, dim)

    def compute_directions_sges(self, dim: int, G: np.ndarray, alpha: float) -> tuple:
        """
        Compute a matrix of random directions depending on gradient information.

        Args:
            dim: Number of rows and columns.
            G: Buffer of gradients, 2-D array.
            alpha: Probability of sampling from gradients.

        Returns:
            Tuple of (matrix of directions, number of directions sampled from gradients).
        """
        G = np.array(G)
        
        G_clean = G[~np.isnan(G).any(axis=1)]
        if len(G_clean) < 2:
            cov_L_G = np.eye(dim)
        else:
            cov_L_G = np.cov(G_clean.T)
            cov_L_G = (cov_L_G + cov_L_G.T) / 2
            eigvals = np.linalg.eigvalsh(cov_L_G)
            if eigvals.min() < 0:
                cov_L_G = cov_L_G - eigvals.min() * np.eye(dim)

        choices = 0

        for i in range(dim):
            choices += int(np.random.choice([0, 1], p=[alpha, 1 - alpha]))

        try:
            dirs_L_G = np.random.multivariate_normal(np.zeros(dim), cov_L_G, choices)
            for i in range(choices):
                dirs_L_G[i] = dirs_L_G[i] / np.std(dirs_L_G[i])
        except:
            dirs_L_G = np.zeros((0, dim))

        dirs_L_G_T = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim - choices)

        dirs = np.concatenate((dirs_L_G, dirs_L_G_T))

        return dirs / np.linalg.norm(dirs, axis=-1)[:, np.newaxis], choices
