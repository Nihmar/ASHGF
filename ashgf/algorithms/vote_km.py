"""VOTEK-M: multi-scale sigma voting — computes gradient at both sigma
and sigma/2, generates step candidates from both, picks the best.

Only activates when confidence is high and Lipschitz spread > 2.0.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote_k import VOTEK
from ashgf.gradient.estimators import gauss_hermite_derivative

logger = logging.getLogger(__name__)

__all__ = ["VOTEKM"]

_SPREAD_MIN = 2.0
_CONFIDENCE_MIN = 0.5


class VOTEKM(VOTEK):
    kind = "VOTEKM"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        spread_min: float = _SPREAD_MIN,
        confidence_min: float = _CONFIDENCE_MIN,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._spread_min = spread_min
        self._confidence_min = confidence_min

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
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                l_spread = float(np.max(lipschitz)) / max(l_mean, 1e-12)

                k_aniso = confidence / ratio
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    candidates.append((x_ani, f_ani))

                if (
                    confidence >= self._confidence_min
                    and l_spread > self._spread_min
                ):
                    sigma_half = self._sigma * 0.5
                    basis = self._basis
                    grad2, _, _, _ = gauss_hermite_derivative(
                        x, f, sigma_half, basis, self.m, f_cur
                    )
                    dir2 = grad2 if maximize else -grad2
                    step2 = sigma_half / max(self._L_nabla, 1e-12)

                    x_uni2 = x + k * step2 * dir2
                    f_uni2 = f(x_uni2)
                    if np.isfinite(f_uni2) and f_uni2 < f_base and f_uni2 < f_cur:
                        candidates.append((x_uni2, f_uni2))

                    ratio2 = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                    k_aniso2 = confidence / ratio2
                    x_ani2 = x + (1.0 + k_aniso2) * step2 * dir2
                    f_ani2 = f(x_ani2)
                    if np.isfinite(f_ani2) and f_ani2 < f_base and f_ani2 < f_cur:
                        candidates.append((x_ani2, f_ani2))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                self._big_step_accepted = True
                return candidates[0]

        self._big_step_accepted = False
        return x_base, f_base
