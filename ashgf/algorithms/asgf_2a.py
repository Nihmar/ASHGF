"""ASGF-2A: dimension-adaptive step multiplier.

Uses ``k = 1 + alpha/sqrt(dim)`` so that the boost is aggressive at
low dimension and conservative at high dimension, where overshooting
is more dangerous.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2A"]


class ASGF2A(ASGF):
    """ASGF with dimension-adaptive step multiplier.

    Parameters
    ----------
    alpha : float
        Multiplier formula: ``k = 1 + alpha / sqrt(dim)``.
        Default ``2.0`` (gives k=1.63 at dim=10, k=1.20 at dim=100).
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2A"

    def __init__(self, alpha: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._alpha = alpha

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad
        dim = len(x)
        k = 1.0 + self._alpha / np.sqrt(dim)

        # Base step
        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        # Try k*x — only if k > 1.01 (meaningful boost)
        if k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            if np.isfinite(f_big) and f_big < f_base:
                return x_big, f_big

        return x_base, f_base
