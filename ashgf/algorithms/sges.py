"""Self-Guided Evolution Strategies (SGES) optimizer."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import _parallel_eval, gaussian_smoothing
from ashgf.gradient.sampling import compute_directions, compute_directions_sges

logger = logging.getLogger(__name__)

__all__ = ["SGES"]


class SGES(BaseOptimizer):
    """Self-Guided Evolution Strategies.

    Extends Gaussian smoothing by adaptively mixing random directions
    with directions sampled from the gradient-history subspace.

    .. note::

       **Semantica di ``alpha``.**  In questa implementazione ``alpha``
       rappresenta la probabilità di campionare una direzione **casuale**
       (isotropica).  Il valore ``1 - alpha`` è quindi la probabilità di
       campionare dal sottospazio dei gradienti storici.  Questa scelta è
       coerente con ``compute_directions_sges``, dove
       ``choices = binomial(dim, 1 - alpha)``.

       Nella tesi, il parametro :math:`\\alpha` ha il significato opposto
       (probabilità di campionare dal sottospazio gradiente).  Di
       conseguenza, la regola di aggiornamento è **invertita** rispetto
       alla formulazione della tesi: quando il sottospazio gradiente è
       migliore (:math:`r < \\hat{r}`), ``alpha`` viene **diminuita** per
       favorire lo sfruttamento.

    Parameters
    ----------
    lr : float
        Fixed learning rate.
    sigma : float
        Smoothing bandwidth.
    k : int
        Parameter controlling buffer management.
    k1 : float
        Upper bound for alpha.
    k2 : float
        Lower bound for alpha.
    alpha : float
        Initial probability of sampling a **random** (isotropic) direction.
        ``1 - alpha`` is the probability of sampling from the gradient
        subspace.
    delta : float
        Multiplicative factor for alpha update.
    t : int
        Number of pure-random warm-up iterations.
    seed : int
        Random seed.
    eps : float
        Convergence threshold.
    """

    kind = "SGES"

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
    ) -> None:
        super().__init__(seed=seed, eps=eps)
        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {lr}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.delta = delta
        self.t = t

        # Internal state — circular buffer for gradient history
        self._G_buffer: np.ndarray | None = None  # (t, dim) pre-allocated
        self._G_count: int = 0  # number of gradients stored so far
        self._G_idx: int = 0  # next write position
        self._current_alpha = alpha

    def _get_step_size(self) -> float:
        return self.lr

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        self._G_buffer = np.zeros((self.t, dim))
        self._G_count = 0
        self._G_idx = 0
        self._current_alpha = self.alpha

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        """Update alpha based on relative performance of gradient vs random directions.

        .. note::

           **Corretto (bug #1).**  Nella tesi, quando
           :math:`\\hat{r}_{\\mathbf{G}} < \\hat{r}_{\\mathbf{G}}^{\\perp}`
           (il sottospazio gradiente produce minimi migliori), :math:`\\alpha`
           viene **aumentato** per favorire lo sfruttamento.  In questo codice
           ``alpha`` è la probabilità di direzioni **casuali**, quindi quando
           ``r < r_hat`` ``alpha`` viene **diminuita** (effetto equivalente:
           più direzioni dal gradiente).
        """
        if iteration < self.t:
            return

        # The evaluations were stored by the gradient estimator
        if not hasattr(self, "_last_evaluations") or not hasattr(self, "_last_M"):
            return

        evaluations = self._last_evaluations  # (2*dim,) array
        M = self._last_M
        dim = len(x)

        if M <= 0 or M >= dim:
            return

        # Vectorised: reshape to (dim, 2) and take min per row
        evals_pairs = evaluations.reshape(-1, 2)  # (dim, 2)
        min_per_dir = np.min(evals_pairs, axis=1)  # (dim,)

        # r = mean of minima over gradient-subspace directions (0:M)
        r = float(np.mean(min_per_dir[:M]))
        # r_hat = mean of minima over random directions (M:dim)
        r_hat = float(np.mean(min_per_dir[M:]))

        # FIXED (bug #1): alpha is probability of RANDOM directions,
        # so when gradient subspace is better (r < r_hat) we DECREASE alpha
        # (favour gradient subspace).  When random is better, INCREASE alpha.
        if r < r_hat:
            # Gradient subspace is more promising → decrease alpha (less random)
            self._current_alpha = max(self._current_alpha / self.delta, self.k2)
        else:
            # Random subspace is more promising (or tie) → increase alpha (more exploration)
            self._current_alpha = min(self.delta * self._current_alpha, self.k1)

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Estimate gradient using SGES direction sampling.

        FIXED: No longer seeds RNG inside this method (bug 1.5.3).
        FIXED: Correctly uses sges directions when buffer is available (bug 1.5.1).
        """
        dim = len(x)

        # Use SGES directions once we have at least t-1 gradients in the buffer
        # (matching original code behavior: at iteration t, G has t-1 entries)
        if self._G_count >= self.t - 1:
            # Use SGES directions mixing gradients + random
            directions, M = compute_directions_sges(
                dim, self._G_buffer[: self._G_count], self._current_alpha
            )
            self._last_M = M

            # Collect evaluations for alpha update
            sigma_dirs = self.sigma * directions  # (dim, dim) pre-scaled
            points: list[np.ndarray] = []
            for i in range(dim):
                d = sigma_dirs[i]
                points.append(x + d)
                points.append(x - d)

            # Evaluate (parallel if ASHGF_N_JOBS > 1)
            results = _parallel_eval(f, points)
            evaluations = np.asarray(results)

            # ---- Vectorised gradient assembly ----
            # diff[i] = f(x + σ·d_i) - f(x - σ·d_i)
            diff = evaluations[0::2] - evaluations[1::2]  # (dim,)
            grad = np.dot(diff, directions) / (2.0 * self.sigma * dim)

            self._last_evaluations = evaluations
        else:
            # Pure random directions during warm-up
            directions = compute_directions(dim)
            grad = gaussian_smoothing(x, f, self.sigma, directions)
            self._last_M = 0
            self._last_evaluations = np.array([])

        # Append to circular gradient buffer
        self._G_buffer[self._G_idx] = grad
        self._G_idx = (self._G_idx + 1) % self.t
        self._G_count = min(self._G_count + 1, self.t)

        return grad
