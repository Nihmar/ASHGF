"""ASGF-2SAW: 2S with adaptive warmup.

The warmup parameter (streak length for full 2x boost) is adjusted online
based on the success rate of 2x attempts.  If most attempts succeed the
optimizer becomes more aggressive (shorter warmup); if many fail it
becomes more conservative (longer warmup).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SAW"]


class ASGF2SAW(ASGF):
    """2S with online warmup adaptation via 2x success-rate tracking.

    Parameters
    ----------
    warmup_init : int
        Starting warmup length.  Default ``3``.
    warmup_min : int
        Minimum allowed warmup.  Default ``1``.
    warmup_max : int
        Maximum allowed warmup.  Default ``10``.
    warmup_interval : int
        How often (in 2x-attempt iterations) to update warmup.
        Default ``10``.
    success_threshold_up : float
        Success rate above which warmup is *decreased* (more aggressive).
        Default ``0.7``.
    success_threshold_down : float
        Success rate below which warmup is *increased* (more conservative).
        Default ``0.3``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SAW"

    def __init__(
        self,
        warmup_init: int = 3,
        warmup_min: int = 1,
        warmup_max: int = 10,
        warmup_interval: int = 10,
        success_threshold_up: float = 0.7,
        success_threshold_down: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup_init
        self._warmup_min = warmup_min
        self._warmup_max = warmup_max
        self._warmup_interval = warmup_interval
        self._threshold_up = success_threshold_up
        self._threshold_down = success_threshold_down

        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

        self._2x_attempts: int = 0
        self._2x_successes: int = 0
        self._iter_since_update: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None
        self._2x_attempts = 0
        self._2x_successes = 0
        self._iter_since_update = 0

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

            accepted = (
                np.isfinite(f_big) and f_big < f_base and f_big < f_cur
            )

            self._2x_attempts += 1
            if accepted:
                self._2x_successes += 1
            self._iter_since_update += 1

            if self._iter_since_update >= self._warmup_interval:
                if self._2x_attempts > 0:
                    success_rate = self._2x_successes / self._2x_attempts
                    if success_rate >= self._threshold_up:
                        self._warmup = max(self._warmup_min, self._warmup - 1)
                    elif success_rate <= self._threshold_down:
                        self._warmup = min(self._warmup_max, self._warmup + 1)
                self._2x_attempts = 0
                self._2x_successes = 0
                self._iter_since_update = 0

            if accepted:
                return x_big, f_big

        return x_base, f_base
