import math
import sys

import numpy as np
import numpy.linalg as la
from scipy.linalg import orth
from scipy.stats import special_ortho_group
from numpy.polynomial.hermite import hermgauss   # Hermite (fisici) -> poi convertiamo
from typing import Optional, Union, List, Tuple

from functions import Function
from optimizers.base import BaseOptimizer


class ASHGF(BaseOptimizer):
    """
    Adaptive Stochastic Historical Gradient-Free (Algorithms 8 + 9 in thesis).

    Combines Directional Gaussian Smoothing (from ASGF) with historical
    gradient‑guided direction sampling (from SGES) and adaptive sigma / lr.

    CORRECTED VERSION:
      1. Uses Hermite quadrature (physical) converted to standard normal.
      2. Fixes probability in direction sampling (now α for subspace).
      3. Removes erroneous normalization of sampled directions.
      4. Handles M properly: when historical=False, M = dim (all directions
         used for Lipschitz estimation).
      5. Cleaner alpha adaptation.
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
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        np.random.seed(self.seed)
        f = Function(function)
        alpha = self.alpha

        x = self._validate_x_init(x_init, dim)

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
        M = dim                     # initial: no historical, so all directions
        lipschitz_coefficients = np.ones(dim)
        basis = special_ortho_group.rvs(dim)

        G = []

        # Precompute quadrature nodes/weights (physical Hermite -> convert to standard normal)
        m = ASHGF.data["m"]
        nodes_phys, weights_phys = hermgauss(m)
        nodes = nodes_phys * np.sqrt(2)            # nodes for standard normal
        weights = weights_phys / np.sqrt(np.pi)    # weights for standard normal
        mid = m // 2

        # Precompute pair indices for Lipschitz estimation
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
                        L_nabla, M, current_val, nodes, weights, pair_indices
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
                        # When not using historical, we treat all directions equally
                        # for Lipschitz estimation (M = dim)
                        M = dim
                    else:
                        if i >= self.t + 1:
                            # Adapt alpha (eq. 2.14)
                            # r_G = mean of best evaluations among subspace directions
                            # r_G_ort = mean among orthogonal complement
                            # We use the minimum evaluation along each direction
                            # (the evaluations[j] array contains the m quadrature points)
                            vals_G = [min(evaluations[j]) for j in range(M)] if M > 0 else []
                            vals_ort = [min(evaluations[j]) for j in range(M, dim)] if M < dim else []

                            r_G = np.mean(vals_G) if vals_G else None
                            r_G_ort = np.mean(vals_ort) if vals_ort else None

                            # Update alpha based on which region gives better (lower) values
                            if r_G is not None and r_G_ort is not None:
                                if r_G < r_G_ort:
                                    alpha = min(self.delta * alpha, self.k1)
                                else:
                                    alpha = max(alpha / self.delta, self.k2)
                            # If one region is empty, leave alpha unchanged

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
        L_nabla, M, value, nodes, weights, pair_indices
    ):
        """
        DGS gradient estimator with Gauss-Hermite quadrature (eq. 2.22)
        using standard normal nodes/weights (converted from physical Hermite).
        Also estimates Lipschitz constants (eq. 3.1) and adaptive learning rate.
        """
        norm_factor = 1.0 / sigma
        sigma_nodes = sigma * nodes

        evaluations = {}
        derivatives = np.empty(dim)

        for j in range(dim):
            temp = np.empty(m)
            for k in range(m):
                if k == m // 2:
                    temp[k] = value
                else:
                    temp[k] = f.evaluate(x + sigma_nodes[k] * basis[j])

            # DGS estimator: (1/σ) Σ w_i * F(x+σ v_i ξ) * v_i
            derivatives[j] = norm_factor * np.dot(weights * nodes, temp)
            evaluations[j] = temp

        # Assemble gradient: grad = Σ (derivative along ξ) * ξ
        grad = derivatives @ basis

        # Estimate Lipschitz constants (eq. 3.1) using precomputed pair set I
        for j in range(dim):
            lip = 0.0
            evals_j = evaluations[j]
            for a, b in pair_indices:
                denom = sigma * (nodes[a] - nodes[b])
                if abs(denom) > 1e-12:
                    val = abs((evals_j[a] - evals_j[b]) / denom)
                    if val > lip:
                        lip = val
            lipschitz_coefficients[j] = lip

        # Adaptive learning rate (eq. 3.2)
        # L_G = max of Lipschitz constants for the first M directions
        # (those sampled from gradient subspace, if any; when M=dim, it's all)
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

        # Reset if sigma too small and resets left
        if r > 0 and sigma < ASHGF.data["ro"] * ASHGF.data["sigma_zero"]:
            basis = special_ortho_group.rvs(dim)
            sigma = ASHGF.data["sigma_zero"]
            A = ASHGF.data["A"]
            B = ASHGF.data["B"]
            r -= 1
            # After reset, we will go through normal loop; M will be set later
            # (in the main loop based on i and historical flag)
            # We return M = dim as default (since historical will likely be False first)
            M = dim
            return sigma, basis, A, B, r, M

        if historical:
            dirs, M = self._compute_directions_sges(dim, G, alpha)
            # Orthonormalize the directions (rows are basis vectors)
            basis = orth(dirs.T).T   # orth returns columns orthonormal, so transpose
        else:
            M = dim    # when not using historical, all directions are "subspace" for L_G
            basis = special_ortho_group.rvs(dim)

        # Ensure basis spans R^dim (if rank-deficient, complement)
        if basis.shape[0] != dim or basis.shape[1] != dim:
            # orth might return fewer vectors if rank deficient; we complete
            U, s, _ = la.svd(basis, full_matrices=True)
            rank = np.sum(s > 1e-10)
            if rank < dim:
                # Generate random orthonormal completion
                Q = orth(np.random.randn(dim, dim - rank))
                basis = np.hstack([U[:, :rank], Q])
            else:
                basis = U
            # Ensure shape is (dim, dim) and rows are orthonormal
            basis = basis.T

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
            # Ensure symmetry and positive semi-definiteness
            cov_G = (cov_G + cov_G.T) / 2
            eigvals = la.eigvalsh(cov_G)
            if eigvals.min() < 0:
                cov_G -= eigvals.min() * np.eye(dim)

        # Number of directions from the gradient subspace (exploitation)
        M = np.random.binomial(dim, alpha)   # FIXED: now alpha is probability for subspace
        M = max(0, min(M, dim))

        try:
            if M > 0:
                dirs_G = np.random.multivariate_normal(np.zeros(dim), cov_G, M)
                # No normalization here – raw samples are fine
            else:
                dirs_G = np.zeros((0, dim))
        except Exception:
            dirs_G = np.zeros((0, dim))
            M = 0

        dirs_rand = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), dim - M)
        dirs = np.concatenate((dirs_G, dirs_rand), axis=0)

        # Normalize each direction to unit length (as required for basis)
        norms = la.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs /= norms

        return dirs, M