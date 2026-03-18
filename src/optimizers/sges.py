from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as la

from functions import Function
from optimizers.base import BaseOptimizer


class SGES(BaseOptimizer):
    """
    Self-Guided Evolution Strategies (Algorithm 5 in the thesis).

    Versione corretta:
      - Un unico RNG (nessun reset del seed).
      - Le direzioni SGES vengono effettivamente utilizzate.
      - Il sottospazio dei gradienti storici è costruito via SVD.
      - Adattamento di alpha secondo l'equazione (2.14).
    """

    kind = "Self-Guided Evolution Strategies (corrected)"

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
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Error: alpha must be in [0,1]")

        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.delta = delta
        self.t = t

        # RNG condiviso (inizializzato una volta sola)
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        f = Function(function)
        alpha = self.alpha

        x = self._validate_x_init(x_init, dim)

        current_val = f.evaluate(x)
        best_value = current_val
        best_values = [[x.copy(), best_value]]
        all_values = [current_val]

        G = []  # buffer dei gradienti (sliding window di dimensione k)

        if debug:
            print(
                f"algorithm: sges (corrected)  function: {function}  dimension: {dim}  initial value: {current_val}"
            )

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(
                        f"{i}th iteration - value: {current_val}  last best value: {best_value}"
                    )

                if i < self.t:
                    # Warmup: stima standard con smoothing gaussiano centrale
                    grad, evaluations = self._grad_estimator(x, f, dim)
                    G.append(grad)
                else:
                    # Fase SGES: usa il sottospazio dei gradienti recenti
                    grad, evaluations, M = self._grad_estimator_sges(
                        x, f, dim, G, alpha
                    )
                    G.append(grad)
                    if len(G) > self.k:
                        G.pop(0)

                    # Adattamento di alpha (eq. 2.14)
                    # I valori di evaluations sono interleaved: [f⁺₁, f⁻₁, f⁺₂, f⁻₂, ...]
                    # Calcola il minimo delle due valutazioni per ogni direzione
                    vals = np.array(evaluations).reshape(-1, 2).min(axis=1)
                    vals_G = vals[:M] if M > 0 else np.array([])
                    vals_ort = vals[M:] if M < dim else np.array([])

                    r_G = np.mean(vals_G) if len(vals_G) > 0 else None
                    r_G_ort = np.mean(vals_ort) if len(vals_ort) > 0 else None

                    # Logica di adattamento (identica a quella descritta nella tesi)
                    if r_G is None or (r_G_ort is not None and r_G < r_G_ort):
                        alpha = min(self.delta * alpha, self.k1)
                    elif r_G_ort is None or (r_G is not None and r_G >= r_G_ort):
                        alpha = max(alpha / self.delta, self.k2)

                # Passo di discesa
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
                f"\nlast evaluation: {all_values[-1]}  last_iterate: {len(all_values) - 1}  best evaluation: {best_value}\n"
            )

        return best_values, all_values

    def _grad_estimator(self, x: np.ndarray, f: Function, dim: int):
        """
        Stima del gradiente con smoothing gaussiano centrale (fase di warmup).
        Versione vettorizzata.
        """
        directions = self._rng.normal(size=(dim, dim))

        points_plus = x + self.sigma * directions
        points_minus = x - self.sigma * directions

        evals_plus = np.array([f.evaluate(points_plus[i]) for i in range(dim)])
        evals_minus = np.array([f.evaluate(points_minus[i]) for i in range(dim)])

        diffs = evals_plus - evals_minus
        grad = diffs @ directions / (2 * self.sigma * dim)

        # Interleave: [f⁺₁, f⁻₁, f⁺₂, f⁻₂, ...]
        evaluations = np.empty(2 * dim)
        evaluations[0::2] = evals_plus
        evaluations[1::2] = evals_minus

        return grad, evaluations

    def _grad_estimator_sges(
        self, x: np.ndarray, f: Function, dim: int, G: list, alpha: float
    ):
        """
        Stima del gradiente con SGES (post-warmup).
        Genera direzioni miste: sfruttamento (sottospazio) ed esplorazione (casuali).
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

    def _compute_directions_sges(self, dim: int, G: list, alpha: float):
        """
        Costruzione delle direzioni di ricerca:
        - M ~ Binomiale(dim, alpha) dal sottospazio dei gradienti recenti (SVD)
        - (dim - M) da N(0,I) (esplorazione)

        Restituisce una matrice (dim, dim) con le direzioni (righe) e il numero M.
        """
        # Campiona M
        M = self._rng.binomial(dim, alpha)
        M = max(0, min(M, dim))

        # Filtra i gradienti che contengono NaN/inf
        G_arr = np.array(G)
        G_clean = G_arr[np.isfinite(G_arr).all(axis=1)]

        dirs_G = np.zeros((0, dim))

        if M > 0 and len(G_clean) >= 2:
            # Costruisci la base ortonormale del sottospazio via SVD di G_clean.T
            # G_clean: (n_grads, dim) → G_clean.T: (dim, n_grads)
            U, s, _ = np.linalg.svd(G_clean.T, full_matrices=False)
            rank = np.sum(s > 1e-10)
            if rank > 0:
                U_sub = U[:, :int(rank)]  # (dim, rank)
                z = self._rng.normal(size=(int(rank), M))  # (rank, M)
                dirs_G = (U_sub @ z).T  # (M, dim)
            else:
                # rango zero (improbabile) → fallback a N(0,I)
                dirs_G = self._rng.normal(size=(M, dim))
        elif M > 0:
            # Non ci sono abbastanza gradienti validi → N(0,I)
            dirs_G = self._rng.normal(size=(M, dim))

        # Direzioni esplorative (casuali)
        dirs_rand = (
            self._rng.normal(size=(dim - M, dim)) if dim - M > 0 else np.zeros((0, dim))
        )

        # Concatena: prime M dal sottospazio, poi le esplorative
        directions = np.vstack((dirs_G, dirs_rand))

        return directions, M
