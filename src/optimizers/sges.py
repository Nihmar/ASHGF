import numpy as np
import numpy.linalg as la

from functions import Function
from optimizers.base import BaseOptimizer


class SGES(BaseOptimizer):
    """
    Self-Guided Evolution Strategies (Algorithm 5 in the thesis).

    After a warmup of t iterations, directions are sampled partly from a
    gradient-dependent subspace (exploitation) and partly from N(0,I)
    (exploration). The trade-off parameter alpha is adapted each iteration.

    BUGFIXES vs original code:
      1. The original grad_estimator computed SGES directions but then
         *immediately overwrote* them with random directions on the next line.
         The SGES directions were never actually used. Fixed by using an
         if/else branch.
      2. Removed np.random.seed(self.seed) inside grad_estimator — resetting
         the seed every call made all random directions identical across
         iterations, defeating the purpose of Monte Carlo sampling.
    """

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
        np.random.seed(self.seed)
        f = Function(function)
        alpha = self.alpha

        x = np.random.randn(dim) if x_init is None else x_init.copy()

        current_val = f.evaluate(x)
        best_value = current_val
        best_values = [[x.copy(), best_value]]
        all_values = [current_val]

        G = []

        if debug:
            print(f"algorithm: sges  function: {function}  dimension: {dim}  initial value: {current_val}")

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(f"{i}th iteration - value: {current_val}  last best value: {best_value}")

                if i < self.t:
                    grad, evaluations = self._grad_estimator(x, f, dim)
                    G.append(grad)
                else:
                    grad, evaluations, M = self._grad_estimator_sges(x, f, dim, G, alpha)
                    G.append(grad)
                    G = G[1:]  # sliding window of size k

                    # Adapt alpha per eq. 2.14
                    # Handle edge cases where M=0 or M=dim (empty ranges)
                    vals_G = [min(evaluations[2 * j], evaluations[2 * j + 1]) for j in range(M)] if M > 0 else []
                    vals_ort = [min(evaluations[2 * j], evaluations[2 * j + 1]) for j in range(M, dim)] if M < dim else []

                    r_G = np.mean(vals_G) if vals_G else None
                    r_G_ort = np.mean(vals_ort) if vals_ort else None

                    if r_G is None or (r_G_ort is not None and r_G < r_G_ort):
                        alpha = min(self.delta * alpha, self.k1)
                    elif r_G_ort is None or r_G >= r_G_ort:
                        alpha = max((1 / self.delta) * alpha, self.k2)

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

    def _grad_estimator(self, x: np.ndarray, f: Function, dim: int):
        """
        Standard central Gaussian smoothing (warmup phase).
        Vectorized: uses matrix operations instead of per-direction loop.
        """
        directions = np.random.randn(dim, dim)

        points_plus = x + self.sigma * directions
        points_minus = x - self.sigma * directions

        evals_plus = np.array([f.evaluate(points_plus[i]) for i in range(dim)])
        evals_minus = np.array([f.evaluate(points_minus[i]) for i in range(dim)])

        diffs = evals_plus - evals_minus
        grad = diffs @ directions / (2 * self.sigma * dim)

        # Interleave evaluations for compatibility with alpha adaptation
        evaluations = np.empty(2 * dim)
        evaluations[0::2] = evals_plus
        evaluations[1::2] = evals_minus

        return grad, evaluations

    def _grad_estimator_sges(self, x: np.ndarray, f: Function, dim: int,
                              G: list, alpha: float):
        """
        SGES gradient estimator (post-warmup phase).
        Directions are sampled partly from gradient subspace, partly from N(0,I).

        FIX: The original code computed SGES directions but then overwrote them
        with purely random directions. This version correctly uses the
        gradient-dependent directions.
        """
        directions, M = self._compute_directions_sges(dim, G, alpha)

        points_plus = x + self.sigma * directions
        points_minus = x - self.sigma * directions

        evals_plus = np.array([f.evaluate(points_plus[i]) for i in range(dim)])
        evals_minus = np.array([f.evaluate(points_minus[i]) for i in range(dim)])

        diffs = evals_plus - evals_minus
        grad = diffs @ directions / (2 * self.sigma * dim)

        evaluations = np.empty(2 * dim)
        evaluations[0::2] = evals_plus
        evaluations[1::2] = evals_minus

        return grad, evaluations, M

    def _compute_directions_sges(self, dim, G, alpha):
        """
        Compute directions: M directions from the gradient subspace (exploitation),
        (dim-M) from N(0,I) (exploration). Subspace directions are sampled as
        U @ z with z ~ N(0,I), where U is an orthonormal basis of the subspace.
        """
        G_arr = np.array(G)
        G_clean = G_arr[~np.isnan(G_arr).any(axis=1)]

        M = np.random.binomial(dim, alpha)
        M = max(0, min(M, dim))

        dirs_G = np.zeros((0, dim))   # default vuoto
        if M > 0 and len(G_clean) >= 2:
            # Costruisci la base del sottospazio via SVD
            U, s, _ = np.linalg.svd(G_clean.T, full_matrices=False)  # U: (dim, k)
            rank = np.sum(s > 1e-10)                                  # soglia per stabilità
            if rank > 0:
                U_sub = U[:, :rank]                                   # base ortonormale
                # Campiona M vettori nel sottospazio
                z = np.random.randn(rank, M)
                dirs_G = (U_sub @ z).T
            else:
                # Se il rango è zero (improbabile), usa N(0,I) come ripiego
                dirs_G = np.random.randn(M, dim)
        elif M > 0:
            # Non ci sono abbastanza gradienti validi → usa N(0,I)
            dirs_G = np.random.randn(M, dim)

        # Direzioni esplorative (complemento ortogonale)
        dirs_rand = np.random.randn(dim - M, dim) if dim - M > 0 else np.zeros((0, dim))

        dirs = np.concatenate((dirs_G, dirs_rand), axis=0)

        # Nota: NON normalizzare! Le direzioni devono mantenere la loro distribuzione.
        return dirs, M