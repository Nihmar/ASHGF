"""ASGF-C: ASGF with step-norm clipping.

Prevents the optimizer from taking excessively large steps by capping
the Euclidean norm of the displacement vector.  Zero extra f-calls.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFC"]


class ASGFC(ASGF):
    """ASGF with step-norm clipping.

    Parameters
    ----------
    max_step : float or None
        Maximum allowed Euclidean norm of the displacement.  If
        ``None`` (default), the max step is set to ``max(1.0, ||x0||)``
        at setup time.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFC"

    def __init__(self, max_step: float | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_step = max_step
        self._clip_value: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        if self._max_step is None:
            self._clip_value = max(1.0, float(np.linalg.norm(x)))
        else:
            self._clip_value = self._max_step

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad
        displacement = step_size * direction
        norm = float(np.linalg.norm(displacement))

        if self._clip_value is not None and norm > self._clip_value:
            displacement *= self._clip_value / norm

        x_new = x + displacement
        return x_new, f(x_new)
