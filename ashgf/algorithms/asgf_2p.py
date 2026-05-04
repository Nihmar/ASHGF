"""ASGF-2P: frequency-gated step boost with progressive multiplier.

The multiplier decays with sigma: k = 1 + sigma/sigma_0.
Large sigma (early exploration) → k ≈ 2.0.
Small sigma (fine convergence) → k ≈ 1.0.
No extra parameters.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2P"]


class ASGF2P(ASGF):
    """Frequency-gated with sigma-proportional multiplier.

    Parameters
    ----------
    warmup : int
        Consecutive improvements before boost attempt.  Default ``3``.
    cooldown : int
        Cooldown after rejection.  Default ``5``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2P"

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

        if self._prev_f is not None and f_base < self._prev_f:
            self._improve_streak += 1
        else:
            self._improve_streak = 0
        self._prev_f = f_base
        self._iter_count += 1

        if self._improve_streak >= self._warmup and self._iter_count > self._skip_until:
            # Progressive multiplier: decays with sigma
            k = 1.0 + self._sigma / max(self._sigma_zero, 1e-15)
            if k > 1.01:
                x_big = x + k * step_size * direction
                f_big = f(x_big)
                if np.isfinite(f_big) and f_big < f_base:
                    return x_big, f_big
            self._skip_until = self._iter_count + self._cooldown

        return x_base, f_base
