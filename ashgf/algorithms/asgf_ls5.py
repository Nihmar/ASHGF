"""ASGF-LS5: Portfolio / Ensemble.

Runs both ASGF (no line search) and ASGF-LS (greedy line search)
independently and returns the best overall result.  This doubles
the computation cost but guarantees that the result is never worse
than either constituent algorithm alone.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.algorithms.base import BaseOptimizer

logger = logging.getLogger(__name__)

__all__ = ["ASGFLS5"]


class ASGFLS5(BaseOptimizer):
    """Portfolio: runs ASGF and ASGF-LS independently, returns best result.

    Parameters
    ----------
    candidates : tuple of float
        Step-size multipliers for the LS instance.
        Default ``(0.25, 0.5, 1.0, 2.0)``.
    seed : int
        Random seed.  The LS instance uses ``seed`` and the ASGF
        instance uses ``seed + 1`` to ensure independent trajectories.
    eps : float
        Convergence threshold.
    **kwargs :
        Additional parameters forwarded to both inner optimizers.
    """

    kind = "ASGFLS5"

    def __init__(
        self,
        candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        seed: int = 2003,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        super().__init__(seed=seed, eps=eps)
        self._candidates = candidates
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Required abstract method — never called directly
    # ------------------------------------------------------------------

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        raise NotImplementedError("ASGFLS5 delegates to inner optimizers")

    # ------------------------------------------------------------------
    # Portfolio optimization
    # ------------------------------------------------------------------

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
    ) -> tuple[list[tuple[np.ndarray, float]], list[float]]:
        from ashgf.algorithms.asgf_ls import ASGFLS

        # Run ASGF with seed
        asgf = ASGF(
            seed=self.seed,
            eps=self.eps,
            **self._extra_kwargs,
        )
        best_asgf, all_asgf = asgf.optimize(
            f,
            dim=dim,
            max_iter=max_iter,
            x_init=x_init.copy() if x_init is not None else None,
            debug=debug,
            log_interval=log_interval,
            maximize=maximize,
            patience=patience,
            ftol=ftol,
        )

        # Run LS with seed+1 for independent trajectory
        ls = ASGFLS(
            candidates=self._candidates,
            seed=self.seed + 1,
            eps=self.eps,
            **self._extra_kwargs,
        )
        best_ls, all_ls = ls.optimize(
            f,
            dim=dim,
            max_iter=max_iter,
            x_init=x_init.copy() if x_init is not None else None,
            debug=debug,
            log_interval=log_interval,
            maximize=maximize,
            patience=patience,
            ftol=ftol,
        )

        asgf_final = best_asgf[-1][1] if best_asgf else float("inf")
        ls_final = best_ls[-1][1] if best_ls else float("inf")

        if debug:
            logger.info(
                "Portfolio: ASGF=%.4e  LS=%.4e → best=%s",
                asgf_final,
                ls_final,
                "ASGF" if asgf_final <= ls_final else "LS",
            )

        if (not maximize and asgf_final <= ls_final) or (maximize and asgf_final >= ls_final):
            return best_asgf, all_asgf
        else:
            return best_ls, all_ls
