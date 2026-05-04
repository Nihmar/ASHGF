"""ASGF-2SA: 2S with Adam-style per-coordinate step scaling.

Maintains an EMA of squared gradients and uses it to scale the step
size *per dimension*, like RMSprop without momentum.  This handles
ill-conditioned functions (where gradient magnitudes vary wildly
across dimensions) without accumulating directional bias.

The 2S boost + safety gate remain unchanged.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SA"]


class ASGF2SA(ASGF):
    """2S with per-coordinate adaptive step sizes.

    Parameters
    ----------
    warmup : int
        Streak length for full 2x boost.  Default ``3``.
    beta2 : float
        EMA factor for squared-gradient accumulator.
        Default ``0.999``.
    eps_a : float
        Regularisation for the per-coordinate denominator.
        Default ``1e-8``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SA"

    def __init__(
        self,
        warmup: int = 3,
        beta2: float = 0.999,
        eps_a: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._beta2 = beta2
        self._eps_a = eps_a

        self._v: np.ndarray | None = None
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._v = np.zeros(dim)
        self._improve_streak = 0
        self._prev_f_base = None

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        assert self._v is not None

        # Per-coordinate squared-gradient EMA
        self._v = self._beta2 * self._v + (1.0 - self._beta2) * grad**2

        step_size = self._get_step_size()

        # Per-coordinate adaptive step: alpha_i = step_size / sqrt(v_i + eps)
        adaptive_step = step_size / (np.sqrt(self._v) + self._eps_a)
        direction = (grad if maximize else -grad) * adaptive_step

        x_base = x + direction
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
            x_big = x + k * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
