"""ASGF-2SMA: 2S with adaptive momentum.

Uses gradient-direction consistency to modulate the momentum coefficient.
When consecutive gradients point in the same direction (high cosine
similarity), momentum is reduced to avoid overshoot on smooth functions.
When gradients are noisy (low similarity), momentum is increased to
smooth through rugged terrain.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SMA"]


class ASGF2SMA(ASGF):
    """2S with adaptive momentum based on gradient consistency.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    beta_min : float
        Minimum momentum when gradients are consistent.
        Default ``0.2``.
    beta_max : float
        Maximum momentum when gradients are noisy.
        Default ``0.8``.
    consistency_ema : float
        Exponential smoothing factor for the gradient-consistency
        tracker.  Default ``0.9``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SMA"

    def __init__(
        self,
        warmup: int = 3,
        beta_min: float = 0.2,
        beta_max: float = 0.8,
        consistency_ema: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._consistency_ema = consistency_ema
        self._velocity: np.ndarray | None = None
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None
        self._prev_grad: np.ndarray | None = None
        self._consistency: float = 0.0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._velocity = np.zeros(dim)
        self._improve_streak = 0
        self._prev_f_base = None
        self._prev_grad = None
        self._consistency = 0.0

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        assert self._velocity is not None

        # --- adaptive momentum coefficient -------------------------------
        if self._prev_grad is not None:
            cos_sim = np.dot(self._prev_grad, grad) / max(
                np.linalg.norm(self._prev_grad) * np.linalg.norm(grad), 1e-12,
            )
            self._consistency = (
                self._consistency_ema * self._consistency
                + (1.0 - self._consistency_ema) * max(cos_sim, 0.0)
            )

        # high consistency → low beta  (avoid overshoot on smooth problems)
        # low  consistency → high beta (smooth through rugged terrain)
        beta = self._beta_min + (self._beta_max - self._beta_min) * (
            1.0 - self._consistency
        )

        self._velocity = beta * self._velocity + (1.0 - beta) * grad
        self._prev_grad = grad.copy()

        step_size = self._get_step_size()
        direction = self._velocity if maximize else -self._velocity

        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
