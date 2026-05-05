"""VOTEK-A: adaptive kappa — sigma decay rate adjusts based on vote success.

When big steps succeed frequently, kappa increases (faster convergence).
When they fail, kappa decreases (preserve exploration).
"""

from __future__ import annotations

import logging
from collections import deque

from ashgf.algorithms.vote_k import VOTEK

logger = logging.getLogger(__name__)

__all__ = ["VOTEKA"]

_KAPPA_INTERVAL = 20
_KAPPA_DELTA = 0.01


class VOTEKA(VOTEK):
    kind = "VOTEKA"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        kappa_interval: int = _KAPPA_INTERVAL,
        kappa_delta: float = _KAPPA_DELTA,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._kappa_interval = kappa_interval
        self._kappa_delta = kappa_delta
        self._vote_history: deque[bool] = deque(maxlen=kappa_interval)
        self._iters_since_kappa_update: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._vote_history.clear()
        self._iters_since_kappa_update = 0

    def _compute_step(self, x, grad, f, maximize):
        result = super()._compute_step(x, grad, f, maximize)
        self._vote_history.append(self._big_step_accepted)
        self._iters_since_kappa_update += 1
        if self._iters_since_kappa_update >= self._kappa_interval:
            if len(self._vote_history) > 0:
                sr = sum(self._vote_history) / len(self._vote_history)
                if sr > 0.7:
                    self._sigma_decay = max(0.90, self._sigma_decay - self._kappa_delta)
                elif sr < 0.3:
                    self._sigma_decay = min(0.99, self._sigma_decay + self._kappa_delta)
            self._vote_history.clear()
            self._iters_since_kappa_update = 0
        return result
