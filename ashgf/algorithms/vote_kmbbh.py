"""VOTEKMB-BH: basin hopping wrapper around VOTEKMB.

Runs multiple optimization phases, each restarting from a perturbed
version of the best point found so far.  Designed for multimodal
functions where a single run converges to a local minimum.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote_kmb import VOTEKMB

logger = logging.getLogger(__name__)

__all__ = ["VOTEKMBBH"]


class VOTEKMBBH(VOTEKMB):
    kind = "VOTEKMBBH"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = 0.98,
        n_restarts: int = 3,
        perturb_scale: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            warmup=warmup, lip_clip=lip_clip, sigma_decay=sigma_decay, **kwargs
        )
        self._n_restarts = n_restarts
        self._perturb_scale = perturb_scale  # None = auto: sqrt(dim) * 1.2

    def _get_perturb_scale(self, dim: int) -> float:
        if self._perturb_scale is not None:
            return self._perturb_scale
        return np.sqrt(dim) * 1.2

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        dim: int = 100,
        max_iter: int = 1000,
        x_init: np.ndarray | None = None,
        debug: bool = True,
        log_interval: int = 25,
        maximize: bool = False,
        patience: int | None = None,
        ftol: float | None = None,
        progress_cb: Callable[[int, float], None] | None = None,
    ) -> tuple[list[tuple[np.ndarray, float]], list[float]]:
        np.random.seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = np.copy(x_init)

        total_budget = max_iter
        iters_per_phase = max(total_budget // (self._n_restarts + 1), 10)

        all_values: list[float] = []
        best_global_x = x.copy()
        best_global_val = f(x)

        if debug:
            logger.info(
                "BasinHopping: %s restarts, %d iters/phase, dim=%d",
                self._n_restarts + 1,
                iters_per_phase,
                dim,
            )

        for phase in range(self._n_restarts + 1):
            if phase > 0:
                sigma_c = self._sigma if hasattr(self, "_sigma") else 1.0
                ps = self._get_perturb_scale(dim)
                noise = ps * sigma_c * self._rng.normal(size=dim)
                x = best_global_x + noise
                if debug:
                    logger.info(
                        "Phase %d/%d: perturbed x (sigma=%.4e, noise_norm=%.4e)",
                        phase + 1,
                        self._n_restarts + 1,
                        sigma_c,
                        float(np.linalg.norm(noise)),
                    )

            bv, av = super().optimize(
                f,
                dim=dim,
                max_iter=iters_per_phase,
                x_init=x,
                debug=debug and phase == 0,
                log_interval=log_interval,
                maximize=maximize,
                patience=patience,
                ftol=ftol,
                progress_cb=progress_cb,
            )

            all_values.extend(av)

            for xp, val in bv:
                if (maximize and val > best_global_val) or (
                    not maximize and val < best_global_val
                ):
                    best_global_val = val
                    best_global_x = xp.copy()

        best_values: list[tuple[np.ndarray, float]] = [
            (best_global_x.copy(), best_global_val)
        ]
        return best_values, all_values
