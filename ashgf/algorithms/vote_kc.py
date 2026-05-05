"""VOTEK-C: candidate pruning — skips anisotropic evaluation when Lipschitz
spread is low, and uses the saved budget for an aggressive extra candidate.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote_k import VOTEK

logger = logging.getLogger(__name__)

__all__ = ["VOTEKC"]

_SPREAD_THRESHOLD = 1.3
_AGGRESSIVE_K = 2.5


class VOTEKC(VOTEK):
    kind = "VOTEKC"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        spread_threshold: float = _SPREAD_THRESHOLD,
        aggressive_k: float = _AGGRESSIVE_K,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._spread_threshold = spread_threshold
        self._aggressive_k = aggressive_k

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
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            f_cur = getattr(self, "_f_at_x", f(x))
            lipschitz = self._lipschitz
            can_aniso = (
                lipschitz is not None
                and not np.all(lipschitz == 0.0)
                and float(np.mean(lipschitz)) >= 1e-12
            )

            candidates: list[tuple[np.ndarray, float]] = []

            x_uni = x + k * step_size * direction
            f_uni = f(x_uni)
            if np.isfinite(f_uni) and f_uni < f_base and f_uni < f_cur:
                candidates.append((x_uni, f_uni))

            if can_aniso:
                l_mean = float(np.mean(lipschitz))
                l_max = float(np.max(lipschitz))
                spread = l_max / max(l_mean, 1e-12)

                if spread > self._spread_threshold:
                    ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                    k_aniso = confidence / ratio
                    x_ani = x + (1.0 + k_aniso) * step_size * direction
                    f_ani = f(x_ani)
                    if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                        candidates.append((x_ani, f_ani))
                else:
                    aggr_k = (1.0 + k) * 0.5 + self._aggressive_k * 0.5
                    x_aggr = x + aggr_k * step_size * direction
                    f_aggr = f(x_aggr)
                    if np.isfinite(f_aggr) and f_aggr < f_base and f_aggr < f_cur:
                        candidates.append((x_aggr, f_aggr))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                self._big_step_accepted = True
                return candidates[0]

        self._big_step_accepted = False
        return x_base, f_base
