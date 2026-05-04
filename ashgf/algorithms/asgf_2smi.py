"""ASGF-2SMI: 2S with momentum and improvement-weighted streak.

The streak counter is weighted by the relative magnitude of each
improvement.  Large improvements build confidence faster than tiny ones,
and large regressions erode it faster.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SMI"]


class ASGF2SMI(ASGF):
    """2S with momentum and improvement-weighted streak.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    beta : float
        Momentum coefficient.  Default ``0.8``.
    max_streak : float
        Upper cap on the streak value to prevent runaway confidence.
        Default ``10.0``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SMI"

    def __init__(
        self,
        warmup: int = 3,
        beta: float = 0.8,
        max_streak: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self.beta = beta
        self._max_streak = max_streak
        self._velocity: np.ndarray | None = None
        self._improve_streak: float = 0.0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._velocity = np.zeros(dim)
        self._improve_streak = 0.0
        self._prev_f_base = None

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        assert self._velocity is not None

        self._velocity = self.beta * self._velocity + (1.0 - self.beta) * grad

        step_size = self._get_step_size()
        direction = self._velocity if maximize else -self._velocity

        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        if self._prev_f_base is not None:
            denom = max(abs(self._prev_f_base), 1e-12)
            if f_base < self._prev_f_base:
                rel_imp = min((self._prev_f_base - f_base) / denom, 2.0)
                self._improve_streak += 1.0 + rel_imp
            elif f_base > self._prev_f_base:
                rel_loss = min((f_base - self._prev_f_base) / denom, 2.0)
                self._improve_streak = max(
                    0.0, self._improve_streak - (1.0 + rel_loss)
                )
        self._prev_f_base = f_base

        self._improve_streak = min(self._improve_streak, self._max_streak)

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
