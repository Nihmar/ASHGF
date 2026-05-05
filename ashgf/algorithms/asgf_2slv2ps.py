"""ASGF-2SLV2PS: 2SLV2 + persistence + spread bonus.

The full combination: triple candidate (uniform + full-aniso + sqrt)
with trajectory persistence and spread-conditioned selection.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLV2PS"]

_PERSISTENCE_ALPHA = 0.3
_PERSISTENCE_EPS = 1e-12
_SPREAD_EMA = 0.9
_SPREAD_REF = 3.0
_SPREAD_EPS = 1e-12


class ASGF2SLV2PS(ASGF):
    kind = "ASGF2SLV2PS"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        persist_alpha: float = _PERSISTENCE_ALPHA,
        persist_eps: float = _PERSISTENCE_EPS,
        spread_ema: float = _SPREAD_EMA, spread_ref: float = _SPREAD_REF,
        spread_eps: float = _SPREAD_EPS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._persist_alpha = persist_alpha
        self._persist_eps = persist_eps
        self._spread_ema = spread_ema
        self._spread_ref = spread_ref
        self._spread_eps = spread_eps
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None
        self._persistence_score: float = 0.0
        self._spread_smooth: float = 1.0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None
        self._persistence_score = 0.0
        self._spread_smooth = 1.0

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

            raw_spread = float(np.max(lipschitz)) / max(l_mean, 1e-12)
            self._spread_smooth = self._spread_ema * self._spread_smooth + (1.0 - self._spread_ema) * raw_spread

        return cand

    def _spread_bonus(self):
        if self._spread_smooth > self._spread_ref:
            return np.log(self._spread_smooth / self._spread_ref) * self._spread_eps
        return 0.0

    def _select(self, candidates):
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            persist_bias = self._persistence_score * self._persist_eps
            s_bonus = self._spread_bonus()
            def key(t):
                bonus = (persist_bias + s_bonus) if t[2] != "uni" else -(persist_bias + s_bonus)
                return t[1] + bonus
            candidates.sort(key=key)
            chosen = candidates[0]

        self._persistence_score += (
            (1.0 if chosen[2] != "uni" else -1.0) - self._persistence_score
        ) * self._persist_alpha

        return chosen[0], chosen[1]

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
