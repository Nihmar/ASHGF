"""ASGF-2SN: 2S with sigmoid confidence mapping.

Instead of a linear confidence = streak / warmup, uses a sigmoid that
keeps confidence near zero until the streak approaches warmup, then
rises sharply.  This avoids premature 2x attempts at low confidence
while still reaching full boost at high confidence.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SN"]


class ASGF2SN(ASGF):
    """2S with sigmoid confidence (non-linear k boost).

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    sigmoid_slope : float
        Slope of the sigmoid.  Higher values make the transition sharper.
        Default ``8.0``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SN"

    def __init__(
        self,
        warmup: int = 3,
        sigmoid_slope: float = 8.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._sigmoid_slope = sigmoid_slope
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

        x_norm = (self._improve_streak / self._warmup - 0.5) * self._sigmoid_slope
        confidence = 1.0 / (1.0 + np.exp(-x_norm))
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
