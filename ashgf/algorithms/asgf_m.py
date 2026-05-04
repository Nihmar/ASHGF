"""ASGF-M: ASGF with Polyak heavy-ball momentum.

Accumulates a velocity vector across iterations, smoothing the
gradient estimates and helping to cross plateaus and narrow valleys.
Zero extra function evaluations.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFM"]


class ASGFM(ASGF):
    """ASGF with heavy-ball momentum.

    Parameters
    ----------
    beta : float
        Momentum coefficient.  ``v = beta*v + (1-beta)*grad``.
        Default ``0.8``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFM"

    def __init__(self, beta: float = 0.8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self._velocity: np.ndarray | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._velocity = np.zeros(dim)

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        assert self._velocity is not None

        # Update velocity with momentum
        self._velocity = self.beta * self._velocity + (1.0 - self.beta) * grad

        direction = self._velocity if maximize else -self._velocity
        x_new = x + step_size * direction
        return x_new, f(x_new)
