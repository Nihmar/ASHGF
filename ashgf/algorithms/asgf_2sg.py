"""ASGF-2SG: 2S with relaxed safety gate at high confidence.

When confidence is high (streak >= warmup), the safety gate is relaxed:
the 2x step is accepted if it beats the current point and is not
*substantially* worse than the base step.  This increases the 2x
acceptance rate when the direction is clearly good.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SG"]


class ASGF2SG(ASGF):
    """2S with relaxed safety gate at high confidence.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    relax_threshold : float
        Minimum confidence to relax the safety gate.  Default ``0.8``.
    relax_margin : float
        Maximum relative excess over ``f_base`` allowed at confidence=1.0.
        Default ``0.01``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SG"

    def __init__(
        self,
        warmup: int = 3,
        relax_threshold: float = 0.8,
        relax_margin: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._relax_threshold = relax_threshold
        self._relax_margin = relax_margin
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None

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

            if not np.isfinite(f_big):
                return x_base, f_base

            if confidence < self._relax_threshold:
                if f_big < f_base and f_big < f_cur:
                    return x_big, f_big
            else:
                margin = (
                    self._relax_margin
                    * (confidence - self._relax_threshold)
                    / (1.0 - self._relax_threshold)
                )
                if f_big < f_cur and f_big < f_base * (1.0 + margin):
                    return x_big, f_big

        return x_base, f_base
