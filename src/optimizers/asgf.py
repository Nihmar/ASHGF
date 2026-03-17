import math

import numpy as np
import numpy.linalg as la
from scipy.stats import special_ortho_group

from functions import Function
from optimizers.base import BaseOptimizer


class ASGF(BaseOptimizer):
    """
    Adaptive Stochastic Gradient-Free (Algorithm 6 + 7 in the thesis).

    Uses Directional Gaussian Smoothing with Gauss-Hermite quadrature to
    estimate the gradient, plus adaptive learning rate and smoothing parameter.

    BUGFIX vs original code:
      Lipschitz constant estimation used only consecutive quadrature node
      pairs (k, k+1), but the thesis (eq. 3.1) specifies all pairs (i,k)
      in set I where |i - ⌊m/2⌋ - 1| ≠ |k - ⌊m/2⌋ - 1|.  Using only
      adjacent pairs underestimates the Lipschitz constant, causing the
      adaptive learning rate to be too large.
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
        x_init: np.ndarray = None,
        debug: bool = True,
        itprint: int = 25,
    ):
        np.random.seed(self.seed)
        f = Function(function)

        x = np.random.randn(dim) if x_init is None else x_init.copy()

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

        # Precompute the Gauss-Hermite nodes, weights, and the valid pair
        # index set I (eq. 3.1) — these are constant across iterations.
        m = ASGF.data["m"]
        p_nodes, w_nodes = np.polynomial.hermite.hermgauss(m)
        p_w = p_nodes * w_nodes
        mid = m // 2
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
                    L_nabla, current_val, p_nodes, p_w, pair_indices
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
        L_nabla, value, p_nodes, p_w, pair_indices
    ):
        """
        DGS gradient estimator with Gauss-Hermite quadrature (eq. 2.22).

        For each basis direction ξⱼ, evaluates F at m quadrature points along
        that direction and assembles the directional derivative estimate.
        Also estimates local Lipschitz constants per direction (eq. 3.1).
        """
        norm_factor = 2.0 / (sigma * math.sqrt(math.pi))
        sigma_p = sigma * p_nodes  # precomputed perturbation magnitudes

        # Evaluate all quadrature points for all directions
        # evaluations[j][k] = F(x + sigma * p_nodes[k] * basis[j])
        evaluations = np.empty((dim, m))
        derivatives = np.empty(dim)

        for j in range(dim):
            for k in range(m):
                if k == m // 2:
                    evaluations[j, k] = value
                else:
                    evaluations[j, k] = f.evaluate(x + sigma_p[k] * basis[j])

            derivatives[j] = norm_factor * np.dot(p_w, evaluations[j])

        # Assemble gradient: g = Σⱼ derivatives[j] * basis[j]
        grad = derivatives @ basis  # (dim,) @ (dim, dim) -> (dim,)

        # Estimate local Lipschitz constants (eq. 3.1)
        # FIX: use all pairs in set I, not just adjacent pairs
        for j in range(dim):
            lip = 0.0
            for a, b in pair_indices:
                denom = sigma * (p_nodes[a] - p_nodes[b])
                if abs(denom) > 1e-12:
                    val = abs((evaluations[j, a] - evaluations[j, b]) / denom)
                    if val > lip:
                        lip = val
            lipschitz_coefficients[j] = lip

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
