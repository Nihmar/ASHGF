"""ASGF-2SQ: 2S with quadratic interpolation for optimal step size.

Fits a quadratic through the three available points (current, base,
and 2x-candidate) and evaluates the interpolated minimum.  This can
find an intermediate step size that is better than either endpoint.
Cost: 1 extra function evaluation per successful 2x attempt.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SQ"]


class ASGF2SQ(ASGF):
    """2S with quadratic interpolation for the optimal boost factor.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    min_k_step : float
        Minimum difference between candidate k values for interpolation
        to be meaningful.  Default ``0.05``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SQ"

    def __init__(
        self,
        warmup: int = 3,
        min_k_step: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._min_k_step = min_k_step
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None

    def _quadratic_min(self, k_vals, f_vals):
        k0, k1, k2 = k_vals
        f0, f1, f2 = f_vals

        denom = k2 * (k2 - k1) * k1
        if abs(denom) < 1e-14:
            return None

        a = (
            k1 * f2
            - k2 * f1
            + (k2 - k1) * f0
        ) / denom
        b = (
            -(k1**2) * f2
            + k2**2 * f1
            - (k2**2 - k1**2) * f0
        ) / denom

        if a <= 0.0:
            return None

        k_opt = -b / (2.0 * a)
        return k_opt

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

            if not (f_big < f_base and f_big < f_cur):
                return x_base, f_base

            best_x, best_f = x_big, f_big

            if k - 1.0 >= self._min_k_step:
                k_opt = self._quadratic_min(
                    (0.0, 1.0, k), (f_cur, f_base, f_big)
                )
                if (
                    k_opt is not None
                    and 1.01 < k_opt < k - 0.01
                ):
                    x_opt = x + k_opt * step_size * direction
                    f_opt = f(x_opt)
                    if np.isfinite(f_opt) and f_opt < best_f:
                        return x_opt, f_opt

            return best_x, best_f

        return x_base, f_base
