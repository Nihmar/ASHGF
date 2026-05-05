"""VOTEKM-L: light multi-scale — second gradient only when
confidence >= 0.7 AND Lipschitz spread > 2.0. Otherwise falls
back to single-gradient VOTEK voting.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote_k import VOTEK
from ashgf.gradient.estimators import gauss_hermite_derivative

logger = logging.getLogger(__name__)

__all__ = ["VOTEKML"]

_CONF_MIN = 0.7
_SPREAD_MIN = 2.0


class VOTEKML(VOTEK):
    kind = "VOTEKML"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        conf_min: float = _CONF_MIN, spread_min: float = _SPREAD_MIN,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs)
        self._conf_min = conf_min
        self._spread_min = spread_min

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

                if confidence >= self._conf_min and l_spread > self._spread_min:
                    sigma2 = self._sigma * 0.5
                    basis = self._basis
                    grad2, _, _, _ = gauss_hermite_derivative(x, f, sigma2, basis, self.m, f_cur)
                    dir2 = grad2 if maximize else -grad2
                    step2 = sigma2 / max(self._L_nabla, 1e-12)

                    x_u2 = x + k * step2 * dir2
                    f_u2 = f(x_u2)
                    if np.isfinite(f_u2) and f_u2 < f_base and f_u2 < f_cur:
                        candidates.append((x_u2, f_u2))

                    k_a2 = confidence / ratio
                    x_a2 = x + (1.0 + k_a2) * step2 * dir2
                    f_a2 = f(x_a2)
                    if np.isfinite(f_a2) and f_a2 < f_base and f_a2 < f_cur:
                        candidates.append((x_a2, f_a2))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                self._big_step_accepted = True
                return candidates[0]

        self._big_step_accepted = False
        return x_base, f_base
