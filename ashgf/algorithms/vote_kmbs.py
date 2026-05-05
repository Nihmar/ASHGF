"""VOTEKMB-S: sigma floor — prevents sigma from decaying below a minimum.
"""

from __future__ import annotations

import logging

from ashgf.algorithms.vote_kmb import VOTEKMB

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBS"]

_SIGMA_FLOOR_FRAC = 0.01


class VOTEKMBS(VOTEKMB):
    kind = "VOTEKMBS"

    def __init__(
        self, warmup: int = 3, lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        sigma_floor_frac: float = _SIGMA_FLOOR_FRAC,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._sigma_floor_frac = sigma_floor_frac
        self._sigma_floor: float = 1e-8

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._sigma_floor = max(self._sigma_zero * self._sigma_floor_frac, 1e-8)

    def _post_iteration(self, iteration, x, grad, f_val):
        super()._post_iteration(iteration, x, grad, f_val)
        if self._sigma < self._sigma_floor:
            self._sigma = self._sigma_floor
