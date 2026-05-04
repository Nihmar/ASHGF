"""ASGF-2J: progressive multiplier + EMA improvement tracking.

Combines sigma-proportional multiplier (2P) with exponential-moving-
average improvement tracking instead of a hard binary streak (2E).
More robust to noise-driven single-iteration stalls.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2J"]


class ASGF2J(ASGF):
    """Progressive multiplier + EMA improvement gating.

    Parameters
    ----------
    ema_beta : float
        EMA decay for the improvement tracker.  Higher = more inertia.
        Default ``0.7``.
    ema_threshold : float
        EMA value above which the boost is attempted.
        Default ``0.5`` (at least 50% of recent steps improved).
    cooldown : int
        Cooldown after rejection.  Default ``5``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2J"

    def __init__(
        self,
        ema_beta: float = 0.7,
        ema_threshold: float = 0.5,
        cooldown: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ema_beta = ema_beta
        self._ema_threshold = ema_threshold
        self._cooldown = cooldown
        self._improve_ema: float = 0.0
        self._skip_until: int = 0
        self._prev_f: float | None = None
        self._iter_count: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_ema = 0.0
        self._skip_until = 0
        self._prev_f = None
        self._iter_count = 0

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad

        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        # EMA of improvement: 1 = improved, 0 = did not
        improved = 1.0 if (self._prev_f is not None and f_base < self._prev_f) else 0.0
        self._improve_ema = (self._ema_beta * self._improve_ema
                             + (1.0 - self._ema_beta) * improved)
        self._prev_f = f_base
        self._iter_count += 1

        if self._improve_ema >= self._ema_threshold and self._iter_count > self._skip_until:
            k = 1.0 + self._sigma / max(self._sigma_zero, 1e-15)
            if k > 1.01:
                x_big = x + k * step_size * direction
                f_big = f(x_big)
                if np.isfinite(f_big) and f_big < f_base:
                    return x_big, f_big
            self._skip_until = self._iter_count + self._cooldown

        return x_base, f_base
