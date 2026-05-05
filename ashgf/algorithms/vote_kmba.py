"""VOTEKMB-A: adaptive kappa from function variance — higher variance
= less decay (more exploration), lower variance = more decay (faster
convergence).
"""

from __future__ import annotations

import logging
from collections import deque

from ashgf.algorithms.vote_kmb import VOTEKMB

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBA"]

_ADAPT_WINDOW = 20
_KAPPA_MIN = 0.92
_KAPPA_MAX = 0.998


class VOTEKMBA(VOTEKMB):
    kind = "VOTEKMBA"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        adapt_window: int = _ADAPT_WINDOW,
        kappa_min: float = _KAPPA_MIN,
        kappa_max: float = _KAPPA_MAX,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._adapt_window = adapt_window
        self._kappa_min = kappa_min
        self._kappa_max = kappa_max
        self._f_window: deque[float] = deque(maxlen=adapt_window)
        self._iter_since_adapt: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._f_window.clear()
        self._iter_since_adapt = 0

    def _post_iteration(self, iteration, x, grad, f_val):
        super()._post_iteration(iteration, x, grad, f_val)
        self._f_window.append(f_val)
        self._iter_since_adapt += 1

        if self._iter_since_adapt >= self._adapt_window:
            f_list = list(self._f_window)
            if len(f_list) >= 2:
                f_std = float(__import__("numpy").std(f_list))
                f_mean = float(abs(__import__("numpy").mean(f_list)))
                cv = f_std / max(f_mean, 1e-12)
                # High CV = noisy function -> less decay (explore more)
                # Low CV = smooth function -> more decay (converge fast)
                normalized = min(cv / 10.0, 1.0)
                self._sigma_decay = self._kappa_min + normalized * (self._kappa_max - self._kappa_min)
            self._f_window.clear()
            self._iter_since_adapt = 0
