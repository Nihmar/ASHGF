"""ASGF-2SMC: 2S with momentum and step-norm clipping.

Combines gradient momentum with step-norm clipping as additional safety,
preventing excessively large steps in high-noise regimes.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SMC"]


class ASGF2SMC(ASGF):
    """2S with momentum and step-norm clipping.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    beta : float
        Momentum coefficient.  Default ``0.8``.
    max_step : float or None
        Maximum allowed Euclidean norm of the displacement.  If
        ``None`` (default), set to ``max(1.0, ||x0||)`` at setup.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SMC"

    def __init__(
        self,
        warmup: int = 3,
        beta: float = 0.8,
        max_step: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self.beta = beta
        self._max_step = max_step
        self._clip_value: float | None = None
        self._velocity: np.ndarray | None = None
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._velocity = np.zeros(dim)
        self._improve_streak = 0
        self._prev_f_base = None
        if self._max_step is None:
            self._clip_value = max(1.0, float(np.linalg.norm(x)))
        else:
            self._clip_value = self._max_step

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

        displacement = step_size * direction
        norm = float(np.linalg.norm(displacement))
        if self._clip_value is not None and norm > self._clip_value:
            displacement *= self._clip_value / norm

        x_base = x + displacement
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
            displacement_big = k * displacement
            norm_big = float(np.linalg.norm(displacement_big))
            if self._clip_value is not None and norm_big > self._clip_value * 2.0:
                displacement_big *= (self._clip_value * 2.0) / norm_big

            x_big = x + displacement_big
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
