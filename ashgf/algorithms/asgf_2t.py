"""ASGF-2T: symmetric two-way step adaptation.

Extends 2S with a symmetric strategy:
- High confidence (streak rising): boost up to 2x, same as 2S.
- Zero confidence (streak exhausted): try a half-step (0.5x),
  which helps when the base step is too aggressive.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2T"]


class ASGF2T(ASGF):
    """Symmetric two-way step adaptation.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2T"

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

        # Update improvement streak
        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)

        # ---- HIGH confidence: try larger step ----
        if confidence > 0.0:
            k_big = 1.0 + confidence  # 1.0 → 2.0
            if k_big > 1.01:
                x_big = x + k_big * step_size * direction
                f_big = f(x_big)
                # f_at_x is already computed by grad_estimator
                f_cur = getattr(self, "_f_at_x", float("inf"))
                if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                    return x_big, f_big

        # ---- ZERO confidence: try smaller step ONLY if base step went uphill ----
        if confidence == 0.0:
            f_cur = getattr(self, "_f_at_x", float("inf"))
            if f_base > f_cur:  # base step is harmful, try smaller
                x_small = x + 0.5 * step_size * direction
                f_small = f(x_small)
                if np.isfinite(f_small) and f_small < f_cur:
                    return x_small, f_small

        return x_base, f_base
