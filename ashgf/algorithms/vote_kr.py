"""VOTEK-R: reset with memory — sigma reset decays with each reset.

Instead of restoring sigma to full sigma_zero on each reset, uses a
decaying fraction: sigma_zero * (decay_factor ** num_resets).  This
preserves convergence progress across resets.
"""

from __future__ import annotations

import logging

import numpy as np

from ashgf.algorithms.vote_k import VOTEK

logger = logging.getLogger(__name__)

__all__ = ["VOTEKR"]


class VOTEKR(VOTEK):
    kind = "VOTEKR"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        reset_decay: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._reset_decay = reset_decay
        self._reset_count: int = 0

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._reset_count = 0

    def _post_iteration(self, iteration, x, grad, f_val):
        dim = len(x)
        if self._last_derivatives is None:
            return
        derivatives = self._last_derivatives

        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            self._reset_count += 1
            factor = self._reset_decay ** self._reset_count
            logger.debug(
                "iter=%d reset #%d: sigma_zero factor=%.4f",
                iteration, self._reset_count, factor,
            )
            self._basis = np.linalg.qr(np.random.randn(dim, dim))[0].T
            self._sigma = self.sigma_zero_ref * factor
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            self._M = dim // 2
            return

        super()._post_iteration(iteration, x, grad, f_val)
