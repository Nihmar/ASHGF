"""ASGF-2SLA: 2SL with adaptive Lipschitz clipping.

The ``lip_clip`` parameter is adjusted online based on the success rate
of 2x attempts.  When attempts succeed frequently, the clip is widened
(more aggressive anisotropy); when they fail, it is tightened (more
conservative, closer to isotropic).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLA"]


class ASGF2SLA(ASGF):
    """2SL with adaptive Lipschitz clip bound.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip_min : float
        Minimum clip value (most conservative).  Default ``1.5``.
    lip_clip_max : float
        Maximum clip value (most aggressive).  Default ``10.0``.
    lip_clip_interval : int
        How often (in 2x-attempt iterations) to update the clip.
        Default ``10``.
    success_threshold_up : float
        Success rate above which the clip is *widened*.
        Default ``0.7``.
    success_threshold_down : float
        Success rate below which the clip is *tightened*.
        Default ``0.3``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SLA"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip_min: float = 1.5,
        lip_clip_max: float = 10.0,
        lip_clip_interval: int = 10,
        success_threshold_up: float = 0.7,
        success_threshold_down: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip_min
        self._lip_clip_min = lip_clip_min
        self._lip_clip_max = lip_clip_max
        self._lip_clip_interval = lip_clip_interval
        self._threshold_up = success_threshold_up
        self._threshold_down = success_threshold_down

        self._improve_streak: int = 0
        self._prev_f_base: float | None = None
        self._2x_attempts: int = 0
        self._2x_successes: int = 0
        self._iter_since_update: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None
        self._lip_clip = self._lip_clip_min
        self._2x_attempts = 0
        self._2x_successes = 0
        self._iter_since_update = 0

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

            accepted = (
                np.isfinite(f_big) and f_big < f_base and f_big < f_cur
            )
            self._2x_attempts += 1
            if accepted:
                self._2x_successes += 1
            self._iter_since_update += 1

            if self._iter_since_update >= self._lip_clip_interval:
                if self._2x_attempts > 0:
                    success_rate = self._2x_successes / self._2x_attempts
                    if success_rate >= self._threshold_up:
                        self._lip_clip = min(
                            self._lip_clip_max, self._lip_clip * 1.2
                        )
                    elif success_rate <= self._threshold_down:
                        self._lip_clip = max(
                            self._lip_clip_min, self._lip_clip * 0.8
                        )
                self._2x_attempts = 0
                self._2x_successes = 0
                self._iter_since_update = 0

            if accepted:
                return x_big, f_big

        return x_base, f_base
