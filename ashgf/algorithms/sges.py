"""Self-Guided Evolution Strategies (SGES) optimizer."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import gaussian_smoothing
from ashgf.gradient.sampling import compute_directions, compute_directions_sges

logger = logging.getLogger(__name__)

__all__ = ["SGES"]


class SGES(BaseOptimizer):
    """Self-Guided Evolution Strategies.

    Extends Gaussian smoothing by adaptively mixing random directions
    with directions sampled from the gradient-history subspace.

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
        Initial probability of sampling from gradient subspace.
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

        # Internal state
        self._G: list[np.ndarray] = []
        self._current_alpha = alpha

    def _get_step_size(self) -> float:
        return self.lr

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        self._G = []
        self._current_alpha = self.alpha

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        """Update alpha based on relative performance of gradient vs random directions."""
        if iteration < self.t:
            return

        # The evaluations were stored by the gradient estimator
        if not hasattr(self, "_last_evaluations") or not hasattr(self, "_last_M"):
            return

        evaluations = self._last_evaluations
        M = self._last_M
        dim = len(x)

        # r = average min(f_plus, f_minus) over gradient-subspace directions
        r_vals = [min(evaluations[2 * j], evaluations[2 * j + 1]) for j in range(M)]
        r = np.mean(r_vals) if r_vals else None

        # r_hat = same over random directions
        r_hat_vals = [
            min(evaluations[2 * j], evaluations[2 * j + 1]) for j in range(M, dim)
        ]
        r_hat = np.mean(r_hat_vals) if r_hat_vals else None

        if r is not None and r_hat is not None:
            if r < r_hat:
                self._current_alpha = min(self.delta * self._current_alpha, self.k1)
            elif r >= r_hat:
                self._current_alpha = max(
                    (1.0 / self.delta) * self._current_alpha, self.k2
                )

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
        if len(self._G) >= self.t - 1:
            # Use SGES directions mixing gradients + random
            directions, M = compute_directions_sges(dim, self._G, self._current_alpha)
            self._last_M = M

            # Collect evaluations for alpha update
            evaluations: list[float] = []
            grad = np.zeros(dim)
            for i in range(dim):
                d = directions[i].reshape(x.shape)
                f_plus = f(x + self.sigma * d)
                f_minus = f(x - self.sigma * d)
                evaluations.append(f_plus)
                evaluations.append(f_minus)
                grad += (f_plus - f_minus) * d.reshape(grad.shape)
            grad /= 2 * self.sigma * dim
            self._last_evaluations = evaluations
        else:
            # Pure random directions during warm-up
            directions = compute_directions(dim)
            grad = gaussian_smoothing(x, f, self.sigma, directions)
            self._last_M = 0
            self._last_evaluations = []

        # Append to gradient buffer
        self._G.append(grad.copy())
        if len(self._G) > self.t:
            self._G = self._G[1:]

        return grad
