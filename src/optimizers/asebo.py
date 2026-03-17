import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA
from scipy.stats import chi2
from functions import Function
from optimizers.base import BaseOptimizer


class ASEBO(BaseOptimizer):
    """
    Adaptive ES-Active Subspaces (Algorithm 3 in the thesis).

    The algorithm uses PCA on the history of estimated gradients to discover
    an active subspace that captures most of the variance. Directions for
    gradient estimation are then sampled from a mixture of the active subspace
    and its orthogonal complement. The mixture probability p is adapted
    online using the estimated gradient components.
    """

    kind = "Adaptive ES-Active Subspaces"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        k: int = 50,  # buffer size / warm‑up iterations
        thresh: float = 0.995,  # variance threshold for PCA
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
        self.thresh = thresh

    def _sample_directions(self, n: int, dim: int, U_act=None, p: float = 0.5):
        """
        Sample `n` perturbation vectors g ∈ ℝᵈ.

        If U_act is None, sampling is isotropic: each g is a standard Gaussian
        renormalised to have χ(d) norm.
        If U_act is given (d×r matrix with orthonormal columns), sampling is
        a mixture:
          - with probability p: g lies in the active subspace (span of U_act)
          - with probability 1‑p: g lies in the orthogonal complement
        In both cases the direction is first made unit length and then scaled
        by an independent χ(d) random variable, so that ‖g‖₂ follows the
        correct distribution.
        """
        directions = np.zeros((n, dim))
        for i in range(n):
            # ----- raw Gaussian vector -----
            v = np.random.randn(dim)

            if U_act is None:
                # isotropic : keep as is
                raw = v
            else:
                # mixture
                proj = U_act @ (U_act.T @ v)  # projection onto active
                if np.random.rand() < p:
                    raw = proj  # active part
                else:
                    raw = v - proj  # orthogonal part

            # ----- normalise to unit direction -----
            norm_raw = la.norm(raw)
            if norm_raw < 1e-12:
                # fallback (extremely unlikely)
                raw = np.random.randn(dim)
                norm_raw = la.norm(raw)
            unit = raw / norm_raw

            # ----- scale to have χ(d) norm -----
            chi = np.sqrt(chi2.rvs(df=dim))
            directions[i] = chi * unit

        return directions

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

        # Buffer of past gradients (FIFO, max size = self.k)
        grad_buffer = []

        # Probability of sampling from the active subspace (initial guess)
        p = 0.5

        if debug:
            print(
                f"algorithm: asebo  function: {function}  dimension: {dim}  initial value: {current_val}"
            )

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(
                        f"{i}th iteration - value: {current_val}  last best value: {best_value}"
                    )

                # ----- gradient estimation -----
                if i <= self.k:
                    # Warm‑up: isotropic sampling, use d directions
                    n_samples = dim
                    U_act = None
                else:
                    # Build active subspace from the gradient buffer
                    if len(grad_buffer) >= 2:
                        G = np.array(grad_buffer)
                        pca = PCA()
                        pca.fit(G)
                        cum_var = np.cumsum(pca.explained_variance_ratio_)
                        # smallest r such that cumulative variance >= thresh
                        r = np.searchsorted(cum_var, self.thresh) + 1
                        # U_act : columns are the principal components
                        U_act = pca.components_[:r].T  # shape (dim, r)
                        n_samples = r
                    else:
                        # Not enough gradients yet – fall back to isotropic
                        n_samples = dim
                        U_act = None

                # Sample the directions
                directions = self._sample_directions(n_samples, dim, U_act, p)

                # Evaluate the black‑box function at perturbed points
                points_plus = x + self.sigma * directions
                points_minus = x - self.sigma * directions

                evals_plus = np.array(
                    [f.evaluate(points_plus[j]) for j in range(n_samples)]
                )
                evals_minus = np.array(
                    [f.evaluate(points_minus[j]) for j in range(n_samples)]
                )

                diffs = evals_plus - evals_minus  # (n_samples,)

                # Gradient estimate: correct scaling (including 1/n_samples)
                grad = diffs @ directions / (2 * self.sigma * n_samples)

                # ----- update gradient buffer (FIFO) -----
                grad_buffer.append(grad.copy())
                if len(grad_buffer) > self.k:
                    grad_buffer.pop(0)

                # ----- update probability p (using the current gradient estimate) -----
                if U_act is not None:
                    # Project gradient onto active and orthogonal subspaces
                    grad_act = U_act @ (U_act.T @ grad)
                    grad_perp = grad - grad_act
                    s_act = np.dot(grad_act, grad_act)
                    s_perp = np.dot(grad_perp, grad_perp)
                    d_act = r
                    d_perp = dim - r

                    # Optimal p from Theorem 3.2 (ignoring bias terms)
                    sqrt_act = np.sqrt(s_act * (d_act + 2)) if s_act > 0 else 0.0
                    sqrt_perp = np.sqrt(s_perp * (d_perp + 2)) if s_perp > 0 else 0.0
                    if sqrt_act + sqrt_perp > 0:
                        p_new = sqrt_act / (sqrt_act + sqrt_perp)
                    else:
                        p_new = 0.5
                    p = np.clip(p_new, 0.01, 0.99)  # avoid extremes

                # ----- parameter update -----
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
            print(
                f"\nlast evaluation: {all_values[-1]}  last_iterate: {len(all_values)-1}  best evaluation: {best_value}\n"
            )

        return best_values, all_values
