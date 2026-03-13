import math
import sys

import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.linalg import orth
from scipy.stats import special_ortho_group

from functions import Function
from optimizers.base import BaseOptimizer


class ASHGF(BaseOptimizer):
    kind = "Adaptive Stochastic Historical Gradient-Free"

    data = {
        "m": 5,
        "A": 0.1,
        "B": 0.9,
        "A_minus": 0.95,
        "A_plus": 1.02,
        "B_minus": 0.98,
        "B_plus": 1.01,
        "gamma_L": 0.9,
        "gamma_sigma": 0.9,
        "gamma_sigma_plus": 1 / 0.9,
        "gamma_sigma_minus": 0.9,
        "r": 10,
        "ro": 0.01,
        "threshold": 10**(-6),
        "sigma_zero": 0.01,
    }

    def __init__(
        self,
        k1: float = 0.9,
        k2: float = 0.1,
        alpha: float = 0.5,
        delta: float = 1.1,
        t: int = 50,
        seed: int = 2003,
        eps: float = 1e-8,
    ):
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.delta = delta
        self.t = t
        super().__init__(seed, eps)

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

        norm = np.linalg.norm(x)
        ASHGF.data["sigma_zero"] = norm / 10
        sigma = ASHGF.data["sigma_zero"]
        A = ASHGF.data["A"]
        B = ASHGF.data["B"]
        r = ASHGF.data["r"]
        L_nabla = 0
        M = dim
        lipschitz_coefficients = np.ones((dim,))
        basis = special_ortho_group.rvs(dim)

        G = []

        if debug:
            print("algorithm:", "ashgf", "function:", function, "dimension:", len(x), "initial value:", steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, "th iteration - value:", steps[i - 1][1], "last best value:", best_value)

                grad, lipschitz_coefficients, lr, derivatives, L_nabla, evaluations = self.grad_estimator(
                    x, ASHGF.data["m"], sigma, len(x), lipschitz_coefficients, basis, f, L_nabla, M, steps[i - 1][1]
                )

                if np.isfinite(grad).all() and np.isfinite(lr):
                    G.append(grad)
                    if len(G) > self.t:
                        G = G[1:]
                    x = x - lr * grad
                else:
                    if debug:
                        print(f"Warning: non-finite gradient or learning rate at iteration {i}")
                    break

                steps[i] = [x, f.evaluate(x)]
                if steps[i][1] < best_value:
                    best_value = steps[i][1]
                    best_values.append([steps[i][0], best_value])

                if la.norm(x - steps[i - 1][0]) < self.eps:
                    break
                else:
                    if i < self.t:
                        historical = False

                    else:
                        if i >= self.t + 1:
                            try:
                                r_ = (1 / M) * np.sum([min(evaluations[j]) for j in range(M)])
                            except:
                                r_ = False

                            try:
                                r_hat = (1 / (dim - M)) * np.sum([min(evaluations[j]) for j in range(M, dim)])
                            except:
                                r_hat = False

                            if not r_ or r_ < r_hat:
                                alpha = min([self.delta * alpha, self.k1])
                            elif not r_hat or r_ >= r_hat:
                                alpha = max([(1 / self.delta) * alpha, self.k2])
                            else:
                                pass

                        historical = True

                    sigma, basis, A, B, r, M = self.subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B, r, G, alpha, historical
                    )

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

    def grad_estimator(
        self,
        x: np.ndarray,
        m: int,
        sigma: float,
        dim: int,
        lipschitz_coefficients: np.ndarray,
        basis: np.ndarray,
        f: Function,
        L_nabla: float,
        M: int,
        value: float,
    ) -> tuple:
        """
        Estimates the gradient and the learning rate.

        Args:
            x: Point on which the operation is performed.
            m: Number of points in the numerical approximation.
            sigma: Smoothing parameter.
            dim: Dimension of the domain.
            lipschitz_coefficients: List of lipschitz coefficients.
            basis: Orthonormal basis for the domain.
            f: Function object over which the smoothing is performed.
            L_nabla: Needed to compute learning rate.
            M: Number of directions sampled from gradient subspace.
            value: f(x).

        Returns:
            Tuple of (estimated gradient, updated lipschitz coefficients, learning rate, array of derivatives, updated L_nabla, evaluations).
        """
        evaluations = {}

        points = {}
        derivatives = []
        p_5, w_5 = np.polynomial.hermite.hermgauss(m)
        p_w_5 = p_5 * w_5
        sigma_p_5 = sigma * p_5
        norm_factor = 2 / (sigma * np.sqrt(math.pi))

        buffer = []
        n = m

        for i in range(n):
            for j in range(n):
                if [i, j] or [j, i] not in buffer:
                    if np.abs(i - int(m / 2)) != np.abs(j - int(m / 2)):
                        buffer.append([i, j])

        for i in range(dim):
            temp = []

            for k in range(m):
                try:
                    if int(m / 2) == k:
                        evaluation = value
                    else:
                        evaluation = f.evaluate(x + sigma_p_5[k] * basis[i])
                    temp.append(evaluation)
                except:
                    print(x.shape, basis.shape)
                    sys.exit(1)

            new_estimate = norm_factor * np.sum(p_w_5 * np.array(temp))

            points[i] = p_5
            evaluations[i] = temp
            derivatives.append(new_estimate)

        grad = np.zeros(x.shape)

        for i in range(len(x)):
            grad = grad + derivatives[i] * basis[i]

        for i in range(len(grad)):
            temp = 0
            for couple in buffer:
                denom = sigma * (points[i][couple[0]] - points[i][couple[1]])
                if abs(denom) > 1e-12:
                    value = np.abs((evaluations[i][couple[0]] - evaluations[i][couple[1]]) / denom)
                    if value > temp:
                        temp = value

            lipschitz_coefficients[i] = temp

        try:
            if M > 0 and len(lipschitz_coefficients) > 0:
                L_nabla = (1 - ASHGF.data["gamma_L"]) * np.max(lipschitz_coefficients[:M]) + ASHGF.data["gamma_L"] * L_nabla
            else:
                L_nabla = max(lipschitz_coefficients) if len(lipschitz_coefficients) > 0 else 1.0
        except Exception:
            L_nabla = max(lipschitz_coefficients) if len(lipschitz_coefficients) > 0 else 1.0

        lr = sigma / L_nabla

        return grad, lipschitz_coefficients, lr, np.array(derivatives), L_nabla, evaluations

    def subroutine(
        self,
        sigma: float,
        grad: np.ndarray,
        derivatives: np.ndarray,
        lipschitz_coefficients: np.ndarray,
        A: float,
        B: float,
        r: int,
        G: np.ndarray,
        alpha: float,
        historical: bool,
    ) -> tuple:
        """
        Subroutine that updates the parameters of the algorithm to adapt it to the function.

        Args:
            sigma: Smoothing parameter.
            grad: Estimated gradient.
            derivatives: Estimated directional derivative.
            lipschitz_coefficients: List of lipschitz coefficients.
            A, B, r: Parameters of the algorithm.
            G: Buffer of gradients, 2-D array.
            alpha: Probability of sampling from gradient subspace.
            historical: If true, use G.

        Returns:
            Tuple of (sigma, basis, A, B, r, M).
        """
        if r > 0 and sigma < ASHGF.data["ro"] * ASHGF.data["sigma_zero"]:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            sigma = ASHGF.data["sigma_zero"]
            A = ASHGF.data["A"]
            B = ASHGF.data["B"]
            r = r - 1
            M = int(len(grad) / 2)
            return sigma, basis, A, B, r, M

        else:
            if historical:
                basis, M = self.compute_directions_sges(len(grad), G, alpha)
                basis = orth(basis)
            else:
                M = int(len(grad) / 2)
                basis = sp.stats.special_ortho_group.rvs(len(grad))

            while basis.shape != (len(grad), len(grad)):
                v = np.random.randn(len(grad), len(grad) - basis.shape[1])
                basis = np.concatenate((basis.T, v.T))
                basis = orth(basis)

            value = np.max(np.abs(derivatives / lipschitz_coefficients))

            if value < A:
                sigma = sigma * ASHGF.data["gamma_sigma_minus"]
                A = A * ASHGF.data["A_minus"]

                return sigma, basis, A, B, r, M

            elif value > B:
                sigma = sigma * ASHGF.data["gamma_sigma_plus"]
                B = B * ASHGF.data["B_plus"]

                return sigma, basis, A, B, r, M

            else:
                A = A * ASHGF.data["A_plus"]
                B = B * ASHGF.data["B_minus"]

                return sigma, basis, A, B, r, M

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
