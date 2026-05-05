"""VOTEKM-B: budget-aware ensemble — distributes extra evaluations among
candidates based on Lipschitz spread.

Low   spread (< 1.5): aggressive step replaces anisotropic (VOTEKC-like)
Med   spread (1.5-3): add sqrt candidate (VOTES-like)
High  spread (> 3.0): multi-scale second gradient (VOTEKM-like)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote_k import VOTEK
from ashgf.gradient.estimators import gauss_hermite_derivative

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMB"]

_SPREAD_LOW = 1.5
_SPREAD_MED = 3.0
_AGGRESSIVE_K = 2.5


class VOTEKMB(VOTEK):
    kind = "VOTEKMB"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        spread_low: float = _SPREAD_LOW,
        spread_med: float = _SPREAD_MED,
        aggressive_k: float = _AGGRESSIVE_K,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs)
        self._spread_low = spread_low
        self._spread_med = spread_med
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
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                l_spread = float(np.max(lipschitz)) / max(l_mean, 1e-12)

                k_aniso = confidence / ratio
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    candidates.append((x_ani, f_ani))

                if l_spread < self._spread_low and confidence > 0.5:
                    aggr_k = (1.0 + k) * 0.5 + self._aggressive_k * 0.5
                    x_aggr = x + aggr_k * step_size * direction
                    f_aggr = f(x_aggr)
                    if np.isfinite(f_aggr) and f_aggr < f_base and f_aggr < f_cur:
                        candidates.append((x_aggr, f_aggr))

                elif l_spread < self._spread_med and confidence > 0.3:
                    k_sqrt = confidence / np.sqrt(ratio)
                    x_sqrt = x + (1.0 + k_sqrt) * step_size * direction
                    f_sqrt = f(x_sqrt)
                    if np.isfinite(f_sqrt) and f_sqrt < f_base and f_sqrt < f_cur:
                        candidates.append((x_sqrt, f_sqrt))

                else:
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
