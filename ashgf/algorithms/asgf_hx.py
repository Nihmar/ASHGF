"""ASGF-HX (Hybrid eXploration): combines Try-2x and conservative 0.5x.

Evaluates 3 step sizes per iteration: base (1x), double (2x), and
half (0.5x).  The 2x step is always accepted if better than base;
the 0.5x step is only accepted when it gives a *substantial*
improvement, preventing greedy myopia on flat plateaus.
Total: 2 extra f-evaluations per iteration.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFHX"]


class ASGFHX(ASGF):
    """Hybrid: Try-2x + conservative Try-0.5x.

    Parameters
    ----------
    half_threshold : float
        The 0.5x step is only accepted when its f-value is at least
        ``half_threshold * max(abs(f_base), 1)`` below the base step.
        Default ``0.7`` (must be 70% better than base).
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFHX"

    def __init__(self, half_threshold: float = 0.7, **kwargs) -> None:
        super().__init__(**kwargs)
        self._half_threshold = half_threshold

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad

        # 1. Base step (always evaluated, always fallback)
        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        best_x = x_base
        best_f = f_base

        # 2. Try 2x — accept if strictly better (low bar, safe).
        x_big = x + 2.0 * step_size * direction
        f_big = f(x_big)
        if np.isfinite(f_big) and f_big < best_f:
            best_x, best_f = x_big, f_big

        # 3. Try 0.5x — accept only if SUBSTANTIALLY better.
        #    High bar prevents greedy myopia on flat plateaus (ackley, levy).
        threshold = self._half_threshold * max(abs(f_base), 1.0)
        x_half = x + 0.5 * step_size * direction
        f_half = f(x_half)
        if np.isfinite(f_half) and f_half < best_f - threshold:
            best_x, best_f = x_half, f_half

        return best_x, best_f
