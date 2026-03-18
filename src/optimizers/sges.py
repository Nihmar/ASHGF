from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as la

from functions import Function
from optimizers.base import BaseOptimizer


class SGES(BaseOptimizer):
    """
    Self-Guided Evolution Strategies (Algorithm 5 in the thesis).
    Versione corretta: usa SVD per il sottospazio, finestra di dimensione k,
    e adatta alpha come descritto nell'equazione (2.14).

    Bugfix rispetto al codice originale:
      1. Le direzioni SGES non venivano mai usate (erano sovrascritte).
      2. La finestra dei gradienti era mantenuta con dimensione t invece di k.
      3. La generazione delle direzioni nel sottospazio era errata (mancava SVD).
      4. Il seed veniva resettato a ogni iterazione, rendendo le direzioni identiche.
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

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        np.random.seed(self.seed)  # seed globale per l'intera ottimizzazione
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
                    # Fase di warmup: stima standard con direzioni casuali
                    grad, evaluations = self._grad_estimator(x, f, dim)
                    G.append(grad)
                else:
                    # Fase SGES: usa il sottospazio dei gradienti recenti
                    grad, evaluations, M = self._grad_estimator_sges(
                        x, f, dim, G, alpha
                    )
                    G.append(grad)
                    if len(G) > self.k:
                        G.pop(0)  # mantiene solo gli ultimi k gradienti

                    # Adattamento di alpha (eq. 2.14)
                    # Calcola i minimi delle coppie per le direzioni nel sottospazio (M)
                    # e per quelle esplorative (dim-M)
                    if M > 0:
                        vals_G = [
                            min(evaluations[2 * j], evaluations[2 * j + 1])
                            for j in range(M)
                        ]
                        r_G = np.mean(vals_G)
                    else:
                        r_G = None

                    if M < dim:
                        vals_ort = [
                            min(evaluations[2 * j], evaluations[2 * j + 1])
                            for j in range(M, dim)
                        ]
                        r_G_ort = np.mean(vals_ort)
                    else:
                        r_G_ort = None

                    # Logica di adattamento
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
        directions = np.random.randn(dim, dim)

        points_plus = x + self.sigma * directions
        points_minus = x - self.sigma * directions

        evals_plus = np.array([f.evaluate(points_plus[i]) for i in range(dim)])
        evals_minus = np.array([f.evaluate(points_minus[i]) for i in range(dim)])

        diffs = evals_plus - evals_minus
        grad = diffs @ directions / (2 * self.sigma * dim)

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
        M = np.random.binomial(dim, alpha)
        M = max(0, min(M, dim))

        # Filtra i gradienti che contengono NaN
        G_arr = np.array(G)
        G_clean = G_arr[~np.isnan(G_arr).any(axis=1)]

        dirs_G = np.zeros((0, dim))  # inizialmente vuoto

        if M > 0:
            if len(G_clean) >= 2:
                # Costruisci la base ortonormale del sottospazio via SVD di G.T
                # G_clean ha dimensione (n_grads, dim) → G_clean.T è (dim, n_grads)
                U, s, _ = np.linalg.svd(
                    G_clean.T, full_matrices=False
                )  # U: (dim, n_grads)
                rank = np.sum(s > 1e-10)
                if rank > 0:
                    U_sub = U[:, :rank]  # base ortonormale (dim, rank)
                    z = np.random.randn(rank, M)  # coefficienti casuali (rank, M)
                    dirs_G = (U_sub @ z).T  # direzioni nel sottospazio (M, dim)
                else:
                    # rango zero (improbabile) → fallback a N(0,I)
                    dirs_G = np.random.randn(M, dim)
            else:
                # Non ci sono abbastanza gradienti validi → N(0,I)
                dirs_G = np.random.randn(M, dim)

        # Direzioni esplorative
        dirs_rand = np.random.randn(dim - M, dim) if dim - M > 0 else np.zeros((0, dim))

        # Concatena: prime M dal sottospazio, poi le esplorative
        directions = np.concatenate((dirs_G, dirs_rand), axis=0)

        # Nota: NON normalizzare! Le direzioni devono mantenere la loro distribuzione.
        return directions, M
