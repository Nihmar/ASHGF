"""ASGF-2SLV2: 2SLV with triple candidate — uniform, full-aniso, and sqrt.

Adds a third candidate using ``confidence / sqrt(ratio)`` weighting.
This covers the intermediate case that fixes ``extended_white_and_holst``
while retaining the extremes.
Cost: 3 (vs 2) extra function evaluations per confident iteration.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLV2"]

class ASGF2SLV2(ASGF):
    kind = "ASGF2SLV2"

    def __init__(self, warmup: int = 3, lip_clip: float = 5.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None

    def _get_candidates(self, x, step_size, direction, confidence, f_base, f_cur, lipschitz, f):
        k = 1.0 + confidence * 1.0
        x_uni = x + k * step_size * direction
        f_uni = f(x_uni)
        cand = []
        if np.isfinite(f_uni) and f_uni < f_base and f_uni < f_cur:
            cand.append((x_uni, f_uni, "uni"))

        can_aniso = lipschitz is not None and not np.all(lipschitz == 0.0) and float(np.mean(lipschitz)) >= 1e-12
        if can_aniso:
            l_mean = float(np.mean(lipschitz))
            ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
            k_aniso = confidence / ratio
            x_ani = x + (1.0 + k_aniso) * step_size * direction
            f_ani = f(x_ani)
            if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                cand.append((x_ani, f_ani, "aniso"))

            k_sqrt = confidence / np.sqrt(ratio)
            x_sqrt = x + (1.0 + k_sqrt) * step_size * direction
            f_sqrt = f(x_sqrt)
            if np.isfinite(f_sqrt) and f_sqrt < f_base and f_sqrt < f_cur:
                cand.append((x_sqrt, f_sqrt, "sqrt"))

        return cand

    def _select(self, candidates):
        candidates.sort(key=lambda t: t[1])
        return candidates[0][0], candidates[0][1]

    def _compute_step(self, x, grad, f, maximize):
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
        if confidence > 0.0 and 1.0 + confidence * 1.0 > 1.01:
            f_cur = getattr(self, "_f_at_x", f(x))
            cand = self._get_candidates(x, step_size, direction, confidence, f_base, f_cur, self._lipschitz, f)
            if cand:
                return self._select(cand)

        return x_base, f_base
