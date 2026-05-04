"""ASGF-2G: Try-2x with a guard-rail.

Accepts the 2x step only when it reduces f by a meaningful amount
over the base step, preventing micro-optimisations near the minimum
that could destabilise convergence.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2G"]


class ASGF2G(ASGF):
    """ASGF with guarded Try-2x.

    Parameters
    ----------
    guard : float
        Fraction of ``|f_base|`` that the 2x step must improve upon
        to be accepted.  Default ``0.01`` (1% relative improvement).
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2G"

    def __init__(self, guard: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self._guard = guard

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

        x_big = x + 2.0 * step_size * direction
        f_big = f(x_big)

        # Accept 2x only when it gives a meaningful improvement
        threshold = self._guard * max(abs(f_base), 1.0)
        if np.isfinite(f_big) and f_big < f_base - threshold:
            return x_big, f_big

        return x_base, f_base
