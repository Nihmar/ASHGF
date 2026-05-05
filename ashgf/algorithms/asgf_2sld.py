"""ASGF-2SLD: Lipschitz-weighted acceleration + multiplicative streak decay.

Combines the two best-performing improvements over ASGF-2S:
1. Per-direction step weighting via Lipschitz constants (from 2SL).
2. Multiplicative streak decay instead of subtractive (from 2SD).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLD"]


class ASGF2SLD(ASGF):
    """2S with Lipschitz-weighted steps and decay streak.

    Parameters
    ----------
    warmup : int
        Streak length at which full 2x boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    decay_factor : float
        Multiplicative factor applied to the streak on a regression.
        Default ``0.7``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SLD"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        decay_factor: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._decay_factor = decay_factor
        self._improve_streak: float = 0.0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0.0
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
            self._improve_streak += 1.0
        else:
            self._improve_streak *= self._decay_factor
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)

        if confidence > 0.0:
            lipschitz = self._lipschitz
            use_uniform = (
                lipschitz is None
                or np.all(lipschitz == 0.0)
                or float(np.mean(lipschitz)) < 1e-12
            )

            if use_uniform:
                k_vec = confidence * np.ones_like(direction)
            else:
                l_mean = float(np.mean(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_vec = confidence / ratio

            max_k = float(np.max(k_vec))
            if max_k < 0.01:
                return x_base, f_base

            x_big = x + (1.0 + k_vec) * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if (
                np.isfinite(f_big)
                and f_big < f_base
                and f_big < f_cur
            ):
                return x_big, f_big

        return x_base, f_base
