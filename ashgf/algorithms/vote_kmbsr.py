"""VOTEKMB-SR: sigma floor + random direction on rugged landscapes.

Combines KMBS (sigma floor prevents premature convergence) and
KMBR (random-direction candidate on rugged/multimodal functions).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ashgf.algorithms.vote_kmb import VOTEKMB
from ashgf.gradient.estimators import gauss_hermite_derivative

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBSR"]

_WINDOW = 15
_CV_THRESHOLD = 0.7
_REJECT_RATE_THRESHOLD = 0.7
_SIGMA_FLOOR_FRAC = 0.01


class VOTEKMBSR(VOTEKMB):
    kind = "VOTEKMBSR"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        sigma_floor_frac: float = _SIGMA_FLOOR_FRAC,
        window: int = _WINDOW,
        cv_threshold: float = _CV_THRESHOLD,
        reject_threshold: float = _REJECT_RATE_THRESHOLD,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._sigma_floor_frac = sigma_floor_frac
        self._sigma_floor: float = 1e-8
        self._window = window
        self._cv_threshold = cv_threshold
        self._reject_threshold = reject_threshold
        self._f_window: deque[float] = deque(maxlen=window)
        self._vote_results: deque[bool] = deque(maxlen=window)
        self._iter_since_check: int = 0
        self._rugged: bool = False

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._sigma_floor = max(self._sigma_zero * self._sigma_floor_frac, 1e-8)
        self._f_window.clear()
        self._vote_results.clear()
        self._iter_since_check = 0
        self._rugged = False

    def _detect_rugged(self) -> bool:
        if len(self._f_window) < self._window:
            return False
        f_arr = np.array(list(self._f_window))
        f_mean = float(np.mean(f_arr))
        if abs(f_mean) < 1e-12:
            return False
        cv = float(np.std(f_arr)) / abs(f_mean)
        if cv < self._cv_threshold:
            return False
        if len(self._vote_results) < self._window:
            return False
        reject_rate = 1.0 - sum(self._vote_results) / len(self._vote_results)
        return reject_rate > self._reject_threshold

    def _post_iteration(self, iteration, x, grad, f_val):
        super()._post_iteration(iteration, x, grad, f_val)
        if self._sigma < self._sigma_floor:
            self._sigma = self._sigma_floor

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

        f_cur = getattr(self, "_f_at_x", f(x))
        self._f_window.append(f_cur)
        self._iter_since_check += 1

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            lipschitz = self._lipschitz
            can_aniso = (
                lipschitz is not None
                and not np.all(lipschitz == 0.0)
                and float(np.mean(lipschitz)) >= 1e-12
            )

            if self._iter_since_check >= self._window:
                self._rugged = self._detect_rugged()
                self._iter_since_check = 0

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

                if self._rugged:
                    dim = len(x)
                    rand_dir = self._rng.normal(size=dim)
                    rand_dir /= max(np.linalg.norm(rand_dir), 1e-12)
                    x_rand = x + k * step_size * rand_dir
                    f_rand = f(x_rand)
                    if np.isfinite(f_rand) and f_rand < f_base and f_rand < f_cur:
                        candidates.append((x_rand, f_rand))

            accepted = len(candidates) > 0
            self._vote_results.append(accepted)

            if candidates:
                candidates.sort(key=lambda t: t[1])
                return candidates[0]

        self._vote_results.append(False)
        return x_base, f_base
