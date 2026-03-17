import math
import sys

import numpy as np
import numpy.linalg as la
from scipy.linalg import orth
from scipy.stats import special_ortho_group

from functions import Function
from optimizers.base import BaseOptimizer


class ASHGF(BaseOptimizer):
    """
    Adaptive Stochastic Historical Gradient-Free (Algorithms 8 + 9 in thesis).

    Combines Directional Gaussian Smoothing (from ASGF) with historical
    gradient-guided direction sampling (from SGES) and adaptive sigma / lr.

    FIXES vs original code:
      1. L∇ update (eq. 3.2): the thesis says L_G is the max of Lipschitz
         constants for the M directions sampled from the gradient subspace.
         The original code had a fragile try/except that could silently fall
         back to incorrect values.  Now handled cleanly.
      2. The pair index set I for Lipschitz estimation is precomputed once
         instead of being rebuilt every call (the original `buffer` variable
         had a subtle bug: `if [i,j] or [j,i] not in buffer` always
         evaluated to True because `[i,j]` is truthy as a non-empty list —
         the `or` short-circuits before reaching `not in buffer`).
      3. Basis complementing logic in subroutine is simplified and more robust.
    """

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
        "threshold": 1e-6,
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
        np.random.seed(self.seed)
        f = Function(function)
        alpha = self.alpha

        x = np.random.randn(dim) if x_init is None else x_init.copy()

        current_val = f.evaluate(x)
        best_value = current_val
        best_values = [[x.copy(), best_value]]
        all_values = [current_val]

        norm_x = la.norm(x)
        ASHGF.data["sigma_zero"] = norm_x / 10
        sigma = ASHGF.data["sigma_zero"]
        A = ASHGF.data["A"]
        B = ASHGF.data["B"]
        r = ASHGF.data["r"]
        L_nabla = 0.0
        M = dim
        lipschitz_coefficients = np.ones(dim)
        basis = special_ortho_group.rvs(dim)

        G = []

        # Precompute quadrature nodes/weights and the valid pair index set I.
        m = ASHGF.data["m"]
        p_nodes, w_nodes = np.polynomial.hermite.hermgauss(m)
        p_w = p_nodes * w_nodes
        mid = m // 2

        # FIX: The original code had `if [i, j] or [j, i] not in buffer`
        # which is always True because `[i, j]` (a non-empty list) is truthy,
        # so the `or` short-circuits.  Correct logic: pairs (a,b) where
        # |a - mid| != |b - mid|, with a < b to avoid duplicates.
        pair_indices = [
            (a, b)
            for a in range(m)
            for b in range(a + 1, m)
            if abs(a - mid) != abs(b - mid)
        ]

        if debug:
            print(f"algorithm: ashgf  function: {function}  dimension: {dim}  initial value: {current_val}")

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(f"{i}th iteration - value: {current_val}  last best value: {best_value}")

                grad, lipschitz_coefficients, lr, derivatives, L_nabla, evaluations = (
                    self._grad_estimator(
                        x, m, sigma, dim, lipschitz_coefficients, basis, f,
                        L_nabla, M, current_val, p_nodes, p_w, pair_indices
                    )
                )

                if not (np.isfinite(grad).all() and np.isfinite(lr)):
                    if debug:
                        print(f"Warning: non-finite gradient or learning rate at iteration {i}")
                    break

                G.append(grad)
                if len(G) > self.t:
                    G = G[1:]

                x_new = x - lr * grad
                new_val = f.evaluate(x_new)
                all_values.append(new_val)

                if new_val < best_value:
                    best_value = new_val
                    best_values.append([x_new.copy(), best_value])

                if la.norm(x_new - x) < self.eps:
                    break
                else:
                    # Decide whether to use historical gradients
                    if i < self.t:
                        historical = False
                    else:
                        if i >= self.t + 1:
                            # Adapt alpha (eq. 2.14)
                            # Handle edge cases where M=0 or M=dim (empty ranges)
                            vals_G = [min(evaluations[j]) for j in range(M)] if M > 0 else []
                            vals_ort = [min(evaluations[j]) for j in range(M, dim)] if M < dim else []

                            r_G = np.mean(vals_G) if vals_G else None
                            r_G_ort = np.mean(vals_ort) if vals_ort else None

                            if r_G is None or (r_G_ort is not None and r_G < r_G_ort):
                                alpha = min(self.delta * alpha, self.k1)
                            elif r_G_ort is None or r_G >= r_G_ort:
                                alpha = max((1 / self.delta) * alpha, self.k2)

                        historical = True

                    sigma, basis, A, B, r, M = self._subroutine(
                        sigma, grad, derivatives, lipschitz_coefficients, A, B, r, G, alpha, historical
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
        L_nabla, M, value, p_nodes, p_w, pair_indices
    ):
        """
        DGS gradient estimator with Gauss-Hermite quadrature (eq. 2.22)
        plus Lipschitz constant estimation (eq. 3.1) and adaptive lr (eq. 3.2).
        """
        norm_factor = 2.0 / (sigma * math.sqrt(math.pi))
        sigma_p = sigma * p_nodes

        evaluations = {}
        derivatives = np.empty(dim)

        for j in range(dim):
            temp = np.empty(m)
            for k in range(m):
                if k == m // 2:
                    temp[k] = value
                else:
                    temp[k] = f.evaluate(x + sigma_p[k] * basis[j])

            derivatives[j] = norm_factor * np.dot(p_w, temp)
            evaluations[j] = temp

        # Assemble gradient
        grad = derivatives @ basis

        # Estimate Lipschitz constants (eq. 3.1) using precomputed pair set I
        for j in range(dim):
            lip = 0.0
            evals_j = evaluations[j]
            for a, b in pair_indices:
                denom = sigma * (p_nodes[a] - p_nodes[b])
                if abs(denom) > 1e-12:
                    val = abs((evals_j[a] - evals_j[b]) / denom)
                    if val > lip:
                        lip = val
            lipschitz_coefficients[j] = lip

        # Adaptive learning rate (eq. 3.2)
        # L_G = max of Lipschitz constants for the first M directions
        # (those sampled from gradient subspace)
        M_eff = max(1, min(M, dim))
        L_G = np.max(lipschitz_coefficients[:M_eff])
        L_nabla = (1 - ASHGF.data["gamma_L"]) * L_G + ASHGF.data["gamma_L"] * L_nabla
        lr = sigma / max(L_nabla, 1e-12)

        return grad, lipschitz_coefficients, lr, derivatives, L_nabla, evaluations

    def _subroutine(
        self, sigma, grad, derivatives, lipschitz_coefficients, A, B, r,
        G, alpha, historical
    ):
        """
        Parameter update (Algorithm 9).
        Updates sigma, builds new basis (optionally using gradient history).
        """
        dim = len(grad)

        if r > 0 and sigma < ASHGF.data["ro"] * ASHGF.data["sigma_zero"]:
            basis = special_ortho_group.rvs(dim)
            sigma = ASHGF.data["sigma_zero"]
            A = ASHGF.data["A"]
            B = ASHGF.data["B"]
            r -= 1
            M = dim // 2
            return sigma, basis, A, B, r, M

        if historical:
            dirs, M = self._compute_directions_sges(dim, G, alpha)
            basis = orth(dirs)
        else:
            M = dim // 2
            basis = special_ortho_group.rvs(dim)

        # Ensure basis spans R^dim (complement if rank-deficient)
        while basis.shape[1] < dim:
            v = np.random.randn(dim, dim - basis.shape[1])
            basis = np.hstack([basis, v])
            basis = orth(basis)

        # orth returns (dim, rank) — we need (dim, dim) then transpose
        # so that rows are basis vectors
        if basis.shape[0] == dim and basis.shape[1] == dim:
            basis = basis.T  # rows = basis vectors
        elif basis.shape != (dim, dim):
            # fallback: random orthonormal basis
            basis = special_ortho_group.rvs(dim)

        # Adapt sigma based on derivative/Lipschitz ratio
        lipschitz_coefficients = np.maximum(lipschitz_coefficients, 1e-10)
        ratio = np.max(np.abs(derivatives / lipschitz_coefficients))

        if ratio < A:
            sigma *= ASHGF.data["gamma_sigma_minus"]
            A *= ASHGF.data["A_minus"]
        elif ratio > B:
            sigma *= ASHGF.data["gamma_sigma_plus"]
            B *= ASHGF.data["B_plus"]
        else:
            A *= ASHGF.data["A_plus"]
            B *= ASHGF.data["B_minus"]

        return sigma, basis, A, B, r, M

    def _compute_directions_sges(self, dim: int, G: list, alpha: float):
        """
        Compute directions: M from gradient covariance, (dim-M) from N(0,I).
        Returns (directions matrix, M).
        """
        G_arr = np.array(G)
        G_clean = G_arr[~np.isnan(G_arr).any(axis=1)]

        if len(G_clean) < 2:
            cov_G = np.eye(dim)
        else:
            cov_G = np.cov(G_clean.T)
            cov_G = (cov_G + cov_G.T) / 2
            eigvals = la.eigvalsh(cov_G)
            if eigvals.min() < 0:
                cov_G -= eigvals.min() * np.eye(dim)

        M = np.random.binomial(dim, 1 - alpha)
        M = max(0, min(M, dim))

        try:
            if M > 0:
                dirs_G = np.random.multivariate_normal(np.zeros(dim), cov_G, M)
                stds = np.std(dirs_G, axis=1, keepdims=True)
                stds = np.maximum(stds, 1e-12)
                dirs_G /= stds
            else:
                dirs_G = np.zeros((0, dim))
        except Exception:
            dirs_G = np.zeros((0, dim))
            M = 0

        dirs_rand = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), dim - M)
        dirs = np.concatenate((dirs_G, dirs_rand), axis=0)

        norms = la.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs /= norms

        return dirs, M
