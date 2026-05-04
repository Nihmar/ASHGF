"""ASGF-2F: frequency-adaptive Try-2x.

Only attempts the 2x step when recent iterations show consistent
descent, indicating a reliable gradient direction.  When progress
stalls the extra evaluation is skipped entirely to save cost and
avoid interfering with sigma/basis adaptation.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2F"]


class ASGF2F(ASGF):
    """ASGF with frequency-adaptive Try-2x.

    Parameters
    ----------
    warmup : int
        Number of consecutive improvements required before 2x is
        attempted.  Default ``3``.
    cooldown : int
        After a rejected 2x attempt, skip 2x for this many iterations.
        Default ``5``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2F"

    def __init__(self, warmup: int = 3, cooldown: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._cooldown = cooldown
        self._improve_streak: int = 0
        self._skip_until: int = 0
        self._prev_f: float | None = None
        self._iter_count: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
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

        # Track improvement streak
        if self._prev_f is not None and f_base < self._prev_f:
            self._improve_streak += 1
        else:
            self._improve_streak = 0
        self._prev_f = f_base
        self._iter_count += 1

        # Only try 2x when confident
        if self._improve_streak >= self._warmup and self._iter_count > self._skip_until:
            x_big = x + 2.0 * step_size * direction
            f_big = f(x_big)
            if np.isfinite(f_big) and f_big < f_base:
                return x_big, f_big
            # Rejected — cooldown
            self._skip_until = self._iter_count + self._cooldown

        return x_base, f_base
