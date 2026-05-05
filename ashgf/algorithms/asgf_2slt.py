"""ASGF-2SLT: two-stage step — uniform first, Lipschitz-anisotropic on failure.

First attempts the uniform 2x step.  If it fails the safety gate AND the
Lipschitz spread indicates high anisotropy, retries with per-direction
Lipschitz-weighted steps.  Cost: 0 extra evaluations when uniform
succeeds (common on isotropic problems); 1 extra when it falls back.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLT"]


class ASGF2SLT(ASGF):
    """Two-stage step: uniform first, Lipschitz-weighted on failure.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    spread_min : float
        Minimum Lipschitz spread to attempt the anisotropic fallback.
        Below this, the function is considered isotropic and we never
        pay for a second evaluation.  Default ``2.0``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SLT"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        spread_min: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._spread_min = spread_min
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

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            f_cur = getattr(self, "_f_at_x", f(x))

            # Stage 1: uniform big step (same as 2S)
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            if (
                np.isfinite(f_big)
                and f_big < f_base
                and f_big < f_cur
            ):
                return x_big, f_big

            # Stage 2: Lipschitz-weighted fallback
            lipschitz = self._lipschitz
            can_use_aniso = (
                lipschitz is not None
                and not np.all(lipschitz == 0.0)
                and float(np.mean(lipschitz)) >= 1e-12
            )
            if can_use_aniso:
                l_mean = float(np.mean(lipschitz))
                l_max = float(np.max(lipschitz))
                spread = l_max / max(l_mean, 1e-12)
                if spread >= self._spread_min:
                    ratio = np.clip(
                        lipschitz / l_mean, 1e-12, self._lip_clip
                    )
                    k_aniso = confidence / ratio
                    x_big2 = x + (1.0 + k_aniso) * step_size * direction
                    f_big2 = f(x_big2)
                    if (
                        np.isfinite(f_big2)
                        and f_big2 < f_base
                        and f_big2 < f_cur
                    ):
                        return x_big2, f_big2

        return x_base, f_base
