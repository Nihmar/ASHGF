"""ASGF-2SLP: 2SL with Lipschitz-spread-gated anisotropy.

Instead of always applying per-direction step weighting, computes the
spread of the Lipschitz constants and blends between uniform and
anisotropic steps proportionally.  On isotropic functions (spread ≈ 1)
the step is fully uniform; only when the Lipschitz spread is high does
anisotropy kick in.  Zero extra function evaluations.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLP"]


class ASGF2SLP(ASGF):
    """2SL with spread-gated anisotropy.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    spread_ref : float
        Lipschitz spread above which full anisotropy is applied.
        Below 1.0 anisotropy is disabled entirely.  Default ``3.0``.
    spread_ema : float
        Exponential smoothing factor for the Lipschitz spread tracker.
        Default ``0.9``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SLP"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        spread_ref: float = 3.0,
        spread_ema: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._spread_ref = spread_ref
        self._spread_ema = spread_ema
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None
        self._spread_smooth: float = 1.0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None
        self._spread_smooth = 1.0

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

            k_uniform = confidence * np.ones_like(direction)

            if use_uniform:
                k_vec = k_uniform
            else:
                l_mean = float(np.mean(lipschitz))
                l_max = float(np.max(lipschitz))
                raw_spread = l_max / max(l_mean, 1e-12)
                self._spread_smooth = (
                    self._spread_ema * self._spread_smooth
                    + (1.0 - self._spread_ema) * raw_spread
                )
                beta = np.clip(
                    (self._spread_smooth - 1.0) / max(self._spread_ref - 1.0, 1e-6),
                    0.0,
                    1.0,
                )
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_aniso = confidence / ratio
                k_vec = (1.0 - beta) * k_uniform + beta * k_aniso

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
