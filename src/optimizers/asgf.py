import math
import sys

import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.linalg import orth
from scipy.stats import special_ortho_group

from functions import Function
from optimizers.base import BaseOptimizer


class ASGF(BaseOptimizer):
    kind = "Adaptive Stochastic Gradient-Free"

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
        "r": 2,
        "ro": 0.01,
        "epsilon_m": 0.1,
        "threshold": 10**(-6),
        "sigma_zero": 0.01,
    }

    def __init__(self, seed: int = 2003, eps: float = 1e-8):
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

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = x_init

        steps = {}
        steps[0] = [x, f.evaluate(x)]

        best_value = steps[0][1]
        best_values = [[x, best_value]]

        norm = np.linalg.norm(x)
        ASGF.data["sigma_zero"] = norm / 10
        sigma = ASGF.data["sigma_zero"]
        A = ASGF.data["A"]
        B = ASGF.data["B"]
        r = ASGF.data["r"]
        L_nabla = 0
        lipschitz_coefficients = np.ones((dim,))
        basis = special_ortho_group.rvs(dim)

        if debug:
            print("algorithm:", "asgf", "function:", function, "dimension:", len(x), "initial value:", steps[0][1])

        for i in range(1, it + 1):
            try:
                if i % itprint == 0:
                    if debug:
                        print(i, "th iteration - value:", steps[i - 1][1], "last best value:", best_value)

                grad, lipschitz_coefficients, lr, derivatives, L_nabla = self.grad_estimator(
                    x, ASGF.data["m"], sigma, len(x), lipschitz_coefficients, basis, f, L_nabla, steps[i - 1][1]
                )

                if not np.isfinite(grad).all() or not np.isfinite(lr):
                    if debug:
                        print(f"Warning: non-finite gradient or learning rate at iteration {i}")
                    break

                x = x - lr * grad
                steps[i] = [x, f.evaluate(x)]
                if steps[i][1] < best_value:
                    best_value = steps[i][1]
                    best_values.append([steps[i][0], best_value])

                if la.norm(x - steps[i - 1][0]) < self.eps:
                    break
                else:
                    sigma, basis, A, B, r = self.subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B, r
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
            value: f(x).

        Returns:
            Tuple of (estimated gradient, updated lipschitz coefficients, learning rate, array of derivatives, updated L_nabla).
        """
        evaluations = {}

        points = {}
        derivatives = []
        p_5, w_5 = np.polynomial.hermite.hermgauss(m)
        p_w_5 = p_5 * w_5
        sigma_p_5 = sigma * p_5
        norm_factor = 2 / (sigma * np.sqrt(math.pi))

        for i in range(dim):
            temp = []

            for k in range(m):
                if int(m / 2) == k:
                    evaluation = value
                else:
                    evaluation = f.evaluate(x + sigma_p_5[k] * basis[i])
                temp.append(evaluation)

            new_estimate = norm_factor * np.sum(p_w_5 * np.array(temp))

            points[i] = p_5
            evaluations[i] = temp
            derivatives.append(new_estimate)

        grad = np.zeros(x.shape)

        for i in range(len(x)):
            grad = grad + derivatives[i] * basis[i]

        for i in range(len(grad)):
            temp = 0
            for k in range(len(points[i]) - 1):
                denom = sigma * (points[i][k + 1] - points[i][k])
                if abs(denom) > 1e-12:
                    value = np.abs((evaluations[i][k + 1] - evaluations[i][k]) / denom)
                    if value > temp:
                        temp = value

            lipschitz_coefficients[i] = temp

        L_nabla = (1 - ASGF.data["gamma_L"]) * lipschitz_coefficients[0] + ASGF.data["gamma_L"] * L_nabla

        lr = sigma / L_nabla

        return grad, lipschitz_coefficients, lr, np.array(derivatives), L_nabla

    def subroutine(
        self, sigma: float, grad: np.ndarray, derivatives: np.ndarray, lipschitz_coefficients: np.ndarray, A: float, B: float, r: int
    ) -> tuple:
        """
        Subroutine that updates the parameters of the algorithm to adapt it to the function.

        Args:
            sigma: Smoothing parameter.
            grad: Estimated gradient.
            derivatives: Estimated directional derivative.
            lipschitz_coefficients: List of lipschitz coefficients.
            A, B, r: Parameters of the algorithm.

        Returns:
            Tuple of (sigma, basis, A, B, r).
        """
        if r > 0 and sigma < ASGF.data["ro"] * ASGF.data["sigma_zero"]:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            sigma = ASGF.data["sigma_zero"]
            A = ASGF.data["A"]
            B = ASGF.data["B"]
            r = r - 1

            return sigma, basis, A, B, r

        else:
            basis = sp.stats.special_ortho_group.rvs(len(grad))
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                basis[0] = grad / grad_norm
            
            # Use QR decomposition to orthonormalize rows
            # basis.T has shape (len(grad), len(grad))
            # qr(basis.T) returns Q with orthonormal columns
            # We want orthonormal rows, so we take Q.T
            Q, R = np.linalg.qr(basis.T)
            basis = Q.T

            # Avoid division by zero in lipschitz_coefficients
            lipschitz_coefficients = np.maximum(lipschitz_coefficients, 1e-10)
            
            value = np.max(np.abs(derivatives / lipschitz_coefficients))

            if value < A:
                sigma = sigma * ASGF.data["gamma_sigma"]
                A = A * ASGF.data["A_minus"]

                return sigma, basis, A, B, r

            elif value > B:
                sigma = sigma / ASGF.data["gamma_sigma"]
                B = B * ASGF.data["B_plus"]

                return sigma, basis, A, B, r

            else:
                A = A * ASGF.data["A_plus"]
                B = B * ASGF.data["B_minus"]

                return sigma, basis, A, B, r
