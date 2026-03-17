import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal

from functions import Function
from optimizers.base import BaseOptimizer


class ASEBO(BaseOptimizer):
    """
    Adaptive ES-Active Subspaces (Algorithm 3 in the thesis).

    Uses PCA on gradient history to discover an active subspace, then
    samples directions from a mixture of the subspace and the full space.

    BUGFIX vs original code:
      The gradient was divided by (2σ) but NOT by n_samples.  The thesis
      (Algorithm 3, line 11) specifies: ∇ = 1/(2n_t σ) Σ (F⁺-F⁻) gⱼ.
      Missing the 1/n_t factor makes the gradient estimate scale with the
      number of samples, which breaks the learning rate tuning.
    """

    kind = "Adaptive ES-Active Subspaces"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        k: int = 50,
        lambd: float = 0.1,
        thresh: float = 1e-4,
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
        self.lambd = lambd
        self.thresh = thresh

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

        alpha = 1.0
        G = []

        if debug:
            print(f"algorithm: asebo  function: {function}  dimension: {dim}  initial value: {current_val}")

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(f"{i}th iteration - value: {current_val}  last best value: {best_value}")

                grad, alpha = self._grad_estimator(x, G, i, alpha, f, dim)

                if not np.isfinite(grad).all():
                    if debug:
                        print(f"Warning: non-finite gradient at iteration {i}")
                    break

                if i == 1:
                    G = grad.reshape(1, -1)
                else:
                    G *= 0.99  # decay
                    G = np.vstack([G, grad.reshape(1, -1)])

                x_new = x - self.lr * grad
                new_val = f.evaluate(x_new)
                all_values.append(new_val)

                if new_val < best_value:
                    best_value = new_val
                    best_values.append([x_new.copy(), best_value])

                if la.norm(x_new - x) < self.eps:
                    break

                x = x_new
                current_val = new_val

            except Exception as e:
                print("Something has gone wrong!")
                print(e)
                break

        if debug:
            print(f"\nlast evaluation: {all_values[-1]}  last_iterate: {len(all_values)-1}  best evaluation: {best_value}\n")

        return best_values, all_values

    def _grad_estimator(self, x: np.ndarray, G, i: int, alpha: float,
                         f: Function, dim: int):
        """
        ASEBO gradient estimator.

        For i < k: uses n_samples=100 random directions from N(0, I).
        For i >= k: uses PCA on gradient history to build a covariance
        matrix that biases sampling toward the active subspace.

        FIX: gradient is now divided by (2 * sigma * n_samples) instead of
        just (2 * sigma).
        """
        n_samples = 100
        UUT = np.zeros((dim, dim))
        UUT_ort = np.zeros((dim, dim))

        if i >= self.k and isinstance(G, np.ndarray) and len(G) >= 2:
            G_clean = G[~np.isnan(G).any(axis=1)]
            if len(G_clean) >= 2:
                pca = PCA()
                pca.fit(G_clean)
                var_exp = np.cumsum(pca.explained_variance_ratio_)
                n_samples = max(10, np.argmax(var_exp > self.thresh) + 1)

                U = pca.components_[:n_samples]
                UUT = U.T @ U
                U_ort = pca.components_[n_samples:]
                UUT_ort = U_ort.T @ U_ort

                if i == self.k:
                    n_samples = 100
            else:
                alpha = 1.0
        else:
            alpha = 1.0

        # Build covariance and sample directions
        cov = (alpha / dim) * np.eye(dim) + ((1 - alpha) / max(n_samples, 1)) * UUT
        cov *= self.sigma

        A = np.zeros((n_samples, dim))
        try:
            L = cholesky(cov, check_finite=False, overwrite_a=True)
            z = standard_normal((n_samples, dim))
            A = z @ L.T  # equivalent to L @ z_i for each row
        except LinAlgError:
            A = np.random.randn(n_samples, dim)

        # Normalize to unit row norms
        norms = la.norm(A, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        A /= norms

        # Evaluate all directions (vectorized outer loop)
        points_plus = x + self.sigma * A   # (n_samples, dim)
        points_minus = x - self.sigma * A  # (n_samples, dim)

        evals_plus = np.array([f.evaluate(points_plus[j]) for j in range(n_samples)])
        evals_minus = np.array([f.evaluate(points_minus[j]) for j in range(n_samples)])

        diffs = evals_plus - evals_minus  # (n_samples,)

        # FIX: divide by n_samples as well (thesis eq: 1/(2 n_t σ))
        grad = diffs @ A / (2 * self.sigma * n_samples)

        # Update alpha based on gradient projections
        if i >= self.k and UUT.any() and UUT_ort.any():
            norm_ort = la.norm(grad @ UUT_ort)
            norm_act = la.norm(grad @ UUT)
            if norm_act > 1e-12:
                alpha = norm_ort / norm_act
            else:
                alpha = 1.0

        return grad, alpha
