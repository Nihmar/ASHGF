"""VOTEKMB-T: stagnation-triggered sigma boost — when f stagnates,
sigma explodes to escape local minima.
"""

from __future__ import annotations

import logging
from collections import deque

from ashgf.algorithms.vote_kmb import VOTEKMB

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBT"]

_STAGNATION_WINDOW = 20
_STAGNATION_REL_TOL = 0.01
_SIGMA_BOOST = 1.5


class VOTEKMBT(VOTEKMB):
    kind = "VOTEKMBT"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        stagnation_window: int = _STAGNATION_WINDOW,
        stagnation_rel_tol: float = _STAGNATION_REL_TOL,
        sigma_boost: float = _SIGMA_BOOST,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._stag_window = stagnation_window
        self._stag_rel_tol = stagnation_rel_tol
        self._sigma_boost = sigma_boost
        self._f_history: deque[float] = deque(maxlen=stagnation_window)
        self._iter_since_check: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._f_history.clear()
        self._iter_since_check = 0

    def _post_iteration(self, iteration, x, grad, f_val):
        super()._post_iteration(iteration, x, grad, f_val)
        self._f_history.append(f_val)
        self._iter_since_check += 1

        if (
            self._iter_since_check >= self._stag_window
            and len(self._f_history) >= self._stag_window
        ):
            f_list = list(self._f_history)
            f_min = min(f_list)
            f_max = max(f_list)
            denom = max(abs(f_min), 1e-12)
            if denom > 0 and (f_max - f_min) / denom < self._stag_rel_tol:
                logger.debug(
                    "iter=%d stagnation detected (f range=%.4e) — boosting sigma",
                    iteration, f_max - f_min,
                )
                self._sigma *= self._sigma_boost
            self._f_history.clear()
            self._iter_since_check = 0
