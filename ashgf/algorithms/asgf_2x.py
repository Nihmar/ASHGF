"""ASGF-2X: ASGF that tries a larger step before committing.

At each iteration, evaluates the function at both the standard step
and at twice the standard step.  Accepts the larger step only when it
produces a strictly lower function value.  One extra f-call per iter.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2X"]


class ASGF2X(ASGF):
    """ASGF that opportunistically tries a 2× step.

    Parameters
    ----------
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2X"

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad

        # Standard ASGF step
        x_base = x + step_size * direction
        f_base = f(x_base)

        if not np.isfinite(f_base):
            return x.copy(), f(x)

        # Try 2x step
        x_big = x + 2.0 * step_size * direction
        f_big = f(x_big)

        if np.isfinite(f_big) and f_big < f_base:
            return x_big, f_big

        return x_base, f_base
