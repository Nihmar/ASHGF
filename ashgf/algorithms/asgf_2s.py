"""ASGF-2S: frequency-gated step boost with safety and smooth blending.

Two fixes over ASGF-2F:
1. **Safety gate**: 2x is only accepted when ``f(2x) < f(x_current)``,
   preventing escape from already-good points.
2. **Smooth multiplier**: blends from 1.0 to 2.0 proportionally to the
   improvement streak, avoiding the zig-zag of binary on/off.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2S"]


class ASGF2S(ASGF):
    """2F with safety gate and smooth blending.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.
        Default ``3``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2S"

    def __init__(self, warmup: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
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

        # Update improvement streak (smooth)
        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        # Smooth confidence-based multiplier
        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0  # ranges from 1.0 to 2.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)

            # Safety: must beat BOTH base AND current point.
            # f(x) is already available from grad_estimator (stored as _f_at_x).
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
