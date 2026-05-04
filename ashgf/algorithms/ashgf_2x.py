"""ASHGF-2X: ASHGF (gradient history) with Try-2x step boost.

Combines ASHGF's gradient-history basis and alpha adaptation with
the opportunistic 2x step from ASGF-2X.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2X"]


class ASHGF2X(ASHGF):
    """ASHGF with Try-2x step exploration.

    Inherits all of ASHGF's gradient-history machinery (basis, alpha,
    sigma adaptation) and adds the 2x step candidate from ASGF-2X.

    Parameters
    ----------
    **kwargs :
        Passed to :class:`ASHGF`.
    """

    kind = "ASHGF2X"

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

        # Try 2x — accept only if better
        x_big = x + 2.0 * step_size * direction
        f_big = f(x_big)
        if np.isfinite(f_big) and f_big < f_base:
            return x_big, f_big

        return x_base, f_base
