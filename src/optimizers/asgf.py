import math

import numpy as np
import numpy.linalg as la
from scipy.stats import special_ortho_group
from typing import Optional, Union, List, Tuple

from functions import Function
from optimizers.base import BaseOptimizer


class ASGF(BaseOptimizer):
    """
    Adaptive Stochastic Gradient-Free (Algorithm 6 + 7 in the thesis).

    Uses Directional Gaussian Smoothing with Gauss-Hermite quadrature to
    estimate the gradient, plus adaptive learning rate and smoothing parameter.
    """

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
        "threshold": 1e-6,
        "sigma_zero": 0.01,
    }

    def __init__(self, seed: int = 2003, eps: float = 1e-8):
        super().__init__(seed, eps)

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        np.random.seed(self.seed)
        f = Function(function)

        x = self._validate_x_init(x_init, dim)

        current_val = f.evaluate(x)
        best_value = current_val
        best_values = [[x.copy(), best_value]]
        all_values = [current_val]

        norm = la.norm(x)
        ASGF.data["sigma_zero"] = norm / 10
        sigma = ASGF.data["sigma_zero"]
        A = ASGF.data["A"]
        B = ASGF.data["B"]
        r = ASGF.data["r"]
        L_nabla = 0.0
        lipschitz_coefficients = np.ones(dim)
        basis = special_ortho_group.rvs(dim)

        # Precompute Gauss-Hermite quadrature for standard normal:
        #   nodes_std = √2 * v_m, weights_std = w_m / √π
        #   so that ∑ weights_std * f(nodes_std) ≈ 𝔼_{z∼N(0,1)}[f(z)].
        m = ASGF.data["m"]
        v_m, w_m = np.polynomial.hermite.hermgauss(m)
        nodes_std = v_m * np.sqrt(2)
        weights_std = w_m / np.sqrt(np.pi)
        mid = m // 2

        # Set of index pairs I for Lipschitz estimation (eq. 3.1)
        pair_indices = [
            (a, b)
            for a in range(m)
            for b in range(a + 1, m)
            if abs(a - mid) != abs(b - mid)
        ]

        if debug:
            print(f"algorithm: asgf  function: {function}  dimension: {dim}  initial value: {current_val}")

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(f"{i}th iteration - value: {current_val}  last best value: {best_value}")

                grad, lipschitz_coefficients, lr, derivatives, L_nabla = self._grad_estimator(
                    x, m, sigma, dim, lipschitz_coefficients, basis, f,
                    L_nabla, current_val, nodes_std, weights_std, pair_indices
                )

                if not np.isfinite(grad).all() or not np.isfinite(lr):
                    if debug:
                        print(f"Warning: non-finite gradient or learning rate at iteration {i}")
                    break

                x_new = x - lr * grad
                new_val = f.evaluate(x_new)
                all_values.append(new_val)

                if new_val < best_value:
                    best_value = new_val
                    best_values.append([x_new.copy(), best_value])

                if la.norm(x_new - x) < self.eps:
                    break
                else:
                    sigma, basis, A, B, r = self._subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B, r
                    )

                x = x_new
                current_val = new_val

            except Exception as e:
                print("Something has gone wrong!")
                print(e)
                break

        if debug:
            print(f"\nlast evaluation: {all_values[-1]}  last_iterate: {len(all_values)-1}  best evaluation: {best_value}\n")

        return best_values, all_values

    def _grad_estimator(
        self, x, m, sigma, dim, lipschitz_coefficients, basis, f,
        L_nabla, value, nodes_std, weights_std, pair_indices
    ):
        """
        DGS gradient estimator with Gauss-Hermite quadrature (eq. 2.22).

        For each basis direction ξⱼ, evaluates F at m quadrature points along
        that direction and assembles the directional derivative estimate.
        Also estimates local Lipschitz constants per direction (eq. 3.1).
        """
        # Perturbation magnitudes: σ * nodes_std
        pert = sigma * nodes_std

        # Evaluate all quadrature points for all directions
        # evaluations[j, k] = F(x + pert[k] * basis[j])
        evaluations = np.empty((dim, m))
        derivatives = np.empty(dim)

        for j in range(dim):
            for k in range(m):
                if k == m // 2:
                    evaluations[j, k] = value
                else:
                    evaluations[j, k] = f.evaluate(x + pert[k] * basis[j])

            # DGS estimator: (1/σ) Σ weights_std * nodes_std * F
            derivatives[j] = (1.0 / sigma) * np.dot(weights_std * nodes_std, evaluations[j])

        # Assemble gradient: g = Σⱼ derivatives[j] * basis[j]
        grad = derivatives @ basis  # (dim,) @ (dim, dim) -> (dim,)

        # Estimate local Lipschitz constants (eq. 3.1) using all pairs in set I
        for j in range(dim):
            lip = 0.0
            for a, b in pair_indices:
                denom = sigma * (nodes_std[a] - nodes_std[b])
                if abs(denom) > 1e-12:
                    val = abs((evaluations[j, a] - evaluations[j, b]) / denom)
                    if val > lip:
                        lip = val
            lipschitz_coefficients[j] = lip

        # Update average Lipschitz constant and compute learning rate (eq. 3.2)
        # In ASGF they use the Lipschitz constant of the first direction (the gradient)
        L_nabla = (1 - ASGF.data["gamma_L"]) * lipschitz_coefficients[0] + ASGF.data["gamma_L"] * L_nabla
        lr = sigma / max(L_nabla, 1e-12)

        return grad, lipschitz_coefficients, lr, derivatives, L_nabla

    def _subroutine(self, sigma, grad, derivatives, lipschitz_coefficients, A, B, r):
        """
        Parameter update (Algorithm 7).
        Adapts sigma and builds a new orthonormal basis with the gradient
        as first direction.
        """
        dim = len(grad)

        # Reset if sigma became too small and resets left
        if r > 0 and sigma < ASGF.data["ro"] * ASGF.data["sigma_zero"]:
            basis = special_ortho_group.rvs(dim)
            sigma = ASGF.data["sigma_zero"]
            A = ASGF.data["A"]
            B = ASGF.data["B"]
            r -= 1
            return sigma, basis, A, B, r

        # Build basis with grad as first direction, then orthonormalize
        basis = special_ortho_group.rvs(dim)
        grad_norm = la.norm(grad)
        if grad_norm > 1e-10:
            basis[0] = grad / grad_norm

        Q, _ = la.qr(basis.T)
        basis = Q.T

        # Adapt sigma based on max(|derivative| / Lipschitz)
        lipschitz_coefficients = np.maximum(lipschitz_coefficients, 1e-10)
        ratio = np.max(np.abs(derivatives / lipschitz_coefficients))

        if ratio < A:
            sigma *= ASGF.data["gamma_sigma"]
            A *= ASGF.data["A_minus"]
        elif ratio > B:
            sigma /= ASGF.data["gamma_sigma"]
            B *= ASGF.data["B_plus"]
        else:
            A *= ASGF.data["A_plus"]
            B *= ASGF.data["B_minus"]

        return sigma, basis, A, B, r