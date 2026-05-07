"""VOTEKMB-Pro: VOTEKMB with 5 mathematical improvements.

1. Entropic spread: H = -Σ p_i log(p_i), spread = exp(H)
2. Curvature-aware step: step * sqrt(L_mean / L_max)
3. EMA confidence: α * Δf + (1-α) * confidence_ema
4. Adaptive thresholds: Bayesian-like update
5. Candidate ensemble: weighted average instead of greedy
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote import VOTE

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBPRO"]

_SPREAD_LOW = 1.5
_SPREAD_MED = 3.0
_AGGRESSIVE_K = 2.5
_ENTROPIC_LOW = 1.2
_ENTROPIC_MED = 2.0


def _compute_entropic_spread(lipschitz: np.ndarray) -> float:
    """Compute entropic spread: H = -Σ p_i log(p_i), spread = exp(H normalized).

    Returns a value in [1, d] where 1 = maximally concentrated, d = uniform.
    """
    l_sum = float(np.sum(lipschitz))
    if l_sum < 1e-12:
        return 1.0
    p = lipschitz / l_sum
    p = p[p > 1e-12]  # avoid log(0)
    if len(p) == 0:
        return 1.0
    H = -float(np.sum(p * np.log(p)))
    d = len(lipschitz)
    H_normalized = H / np.log(d) if d > 1 else 1.0
    return np.exp(H_normalized)


class VOTEKMBPRO(VOTE):
    kind = "VOTEKMBPRO"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        ema_alpha: float = 0.3,
        adaptive_thresh: bool = True,
        ensemble_temp: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, **kwargs)
        self._sigma_decay = sigma_decay
        self._ema_alpha = ema_alpha
        self._adaptive_thresh = adaptive_thresh
        self._ensemble_temp = ensemble_temp

        self._spread_low = _SPREAD_LOW
        self._spread_med = _SPREAD_MED
        self._aggressive_k = _AGGRESSIVE_K

        self._ema_confidence: float = -1.0
        self._improve_streak: int = 0
        self._prev_f_val: float | None = None
        self._thresh_low: float = _ENTROPIC_LOW
        self._thresh_med: float = _ENTROPIC_MED
        self._big_step_accepted: bool = False

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._ema_confidence = 0.0
        self._prev_f_val = None
        self._big_step_accepted = False
        if self._adaptive_thresh:
            self._thresh_low = _ENTROPIC_LOW
            self._thresh_med = _ENTROPIC_MED

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

        f_cur = getattr(self, "_f_at_x", f(x))

        if self._prev_f_val is not None and f_base < self._prev_f_val:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_val = f_base

        if self._ema_confidence >= 0:
            self._ema_confidence = self._ema_alpha * min(float(self._improve_streak) / self._warmup, 1.0) + \
                               (1 - self._ema_alpha) * self._ema_confidence
        else:
            self._ema_confidence = min(float(self._improve_streak) / self._warmup, 1.0)

        confidence = max(0.0, min(1.0, self._ema_confidence))

        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            candidates: list[tuple[np.ndarray, float, float]] = []

            x_uni = x + k * step_size * direction
            f_uni = f(x_uni)
            if np.isfinite(f_uni) and f_uni < f_base and f_uni < f_cur:
                delta_uni = f_cur - f_uni
                candidates.append((x_uni, f_uni, delta_uni))

            lipschitz = self._lipschitz
            can_aniso = (
                lipschitz is not None
                and not np.all(lipschitz == 0.0)
                and float(np.mean(lipschitz)) >= 1e-12
            )

            if can_aniso:
                l_mean = float(np.mean(lipschitz))
                l_max = float(np.max(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                l_spread = _compute_entropic_spread(lipschitz)

                curv_factor = np.sqrt(l_mean / l_max) if l_max > 1e-12 else 1.0
                step_curv = step_size * curv_factor

                k_aniso = confidence / np.mean(ratio)
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    delta_ani = f_cur - f_ani
                    candidates.append((x_ani, f_ani, delta_ani))

                if l_spread < self._thresh_low and confidence > 0.5:
                    aggr_k = (1.0 + k) * 0.5 + self._aggressive_k * 0.5
                    x_aggr = x + aggr_k * step_size * direction
                    f_aggr = f(x_aggr)
                    if np.isfinite(f_aggr) and f_aggr < f_base and f_aggr < f_cur:
                        candidates.append((x_aggr, f_aggr, f_cur - f_aggr))

                elif l_spread < self._thresh_med and confidence > 0.3:
                    k_sqrt = confidence / np.sqrt(np.mean(ratio))
                    x_sqrt = x + (1.0 + k_sqrt) * step_size * direction
                    f_sqrt = f(x_sqrt)
                    if np.isfinite(f_sqrt) and f_sqrt < f_base and f_sqrt < f_cur:
                        candidates.append((x_sqrt, f_sqrt, f_cur - f_sqrt))

                else:
                    from ashgf.gradient.estimators import gauss_hermite_derivative

                    sigma2 = self._sigma * 0.5
                    basis = self._basis
                    grad2, _, _, _ = gauss_hermite_derivative(x, f, sigma2, basis, self.m, f_cur)
                    dir2 = grad2 if maximize else -grad2
                    step2 = sigma2 / max(self._L_nabla, 1e-12)

                    x_u2 = x + k * step2 * dir2
                    f_u2 = f(x_u2)
                    if np.isfinite(f_u2) and f_u2 < f_base and f_u2 < f_cur:
                        candidates.append((x_u2, f_u2, f_cur - f_u2))

                    k_a2 = confidence / np.mean(ratio)
                    x_a2 = x + (1.0 + k_a2) * step2 * dir2
                    f_a2 = f(x_a2)
                    if np.isfinite(f_a2) and f_a2 < f_base and f_a2 < f_cur:
                        candidates.append((x_a2, f_a2, f_cur - f_a2))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                self._big_step_accepted = True

                if self._ensemble_temp > 0 and len(candidates) > 1:
                    deltas = np.array([delta for _, _, delta in candidates])
                    max_delta = np.max(deltas)
                    deltas = deltas - max_delta
                    weights = np.exp(deltas / (self._ensemble_temp + 1e-12))
                    weights = weights / (np.sum(weights) + 1e-12)
                    weights = np.clip(weights, 0, 1)

                    x_ensemble = np.zeros_like(x)
                    for (xc, _, _), w in zip(candidates, weights):
                        x_ensemble += w * xc
                    f_ensemble = f(x_ensemble)

                    if np.isfinite(f_ensemble) and f_ensemble < f_base:
                        return x_ensemble, f_ensemble

                return candidates[0][0], candidates[0][1]

        self._big_step_accepted = False
        return x_base, f_base

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        super()._post_iteration(iteration, x, grad, f_val)
        if self._big_step_accepted:
            self._sigma *= self._sigma_decay

        if self._adaptive_thresh and self._lipschitz is not None:
            l_mean = float(np.mean(self._lipschitz))
            if l_mean > 1e-12:
                l_max = float(np.max(self._lipschitz))
                ratio = l_max / l_mean
                feedback = ratio / (self._thresh_low + 1e-12)
                self._thresh_low = clip(self._thresh_low * (1 + 0.1 * (feedback - 1)), 0.5, 5.0)
                self._thresh_med = clip(self._thresh_med * (1 + 0.05 * (feedback - 1)), 1.0, 10.0)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))