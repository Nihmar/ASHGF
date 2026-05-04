"""ASGF-LS: ASGF with Line Search on the step size.

Instead of accepting a single step ``sigma / L_nabla``, this variant
evaluates several candidate step sizes and picks the one that gives
the lowest function value.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFLS"]


class ASGFLS(ASGF):
    """ASGF with line search on the step size.

    At each iteration, evaluates f at points obtained with
    ``candidates * step_size`` in both positive and negative
    gradient directions and selects the best one.

    Parameters
    ----------
    candidates : tuple of float
        Multiplicative factors for the step size to try.
        Default ``(0.25, 0.5, 1.0, 2.0)``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFLS"

    def __init__(
        self,
        candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ls_candidates = candidates

    # ------------------------------------------------------------------
    # Override step computation
    # ------------------------------------------------------------------

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()

        direction = grad if maximize else -grad
        best_x = None
        best_f = float("inf")

        for factor in self._ls_candidates:
            alpha = step_size * factor
            x_cand = x + alpha * direction
            f_cand = f(x_cand)
            if np.isfinite(f_cand) and f_cand < best_f:
                best_f = f_cand
                best_x = x_cand

        if best_x is None:
            best_x = x.copy()
            best_f = f(x)

        return best_x, best_f
