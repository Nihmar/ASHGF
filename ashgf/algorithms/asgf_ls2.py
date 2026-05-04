"""ASGF-LS2: Softmax-weighted step blending.

Instead of greedily picking the candidate with the lowest f, this
variant computes a softmax over the candidate function values and
blends the corresponding x points proportionally.  The temperature
anneals over time, converging to a near-greedy selection in late
iterations while maintaining smooth exploration early on.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFLS2"]


class ASGFLS2(ASGF):
    """ASGF with softmax-weighted step blending.

    Parameters
    ----------
    candidates : tuple of float
        Step-size multipliers.  Default ``(0.25, 0.5, 1.0, 2.0)``.
    temperature : float
        Initial softmax temperature.  Higher = more uniform blending.
        Default ``1.0``.
    anneal_rate : float
        Multiplicative temperature decay per iteration.
        Default ``0.97`` (halves every ~23 iterations).
    min_temperature : float
        Floor for the temperature to avoid numerical issues.
        Default ``1e-4``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFLS2"

    def __init__(
        self,
        candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        temperature: float = 1.0,
        anneal_rate: float = 0.97,
        min_temperature: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ls_candidates = candidates
        self._temperature = temperature
        self._anneal_rate = anneal_rate
        self._min_temperature = min_temperature
        self._current_temp: float = temperature

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._current_temp = self._temperature

    # ------------------------------------------------------------------
    # Step computation
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

        points: list[np.ndarray] = []
        f_vals: list[float] = []

        for factor in self._ls_candidates:
            alpha = step_size * factor
            x_cand = x + alpha * direction
            f_cand = f(x_cand)
            if np.isfinite(f_cand):
                points.append(x_cand)
                f_vals.append(f_cand)

        if not points:
            return x.copy(), f(x)

        if len(points) == 1:
            return points[0], f_vals[0]

        # Softmax: lower f → higher weight (minimization).
        #   p_i ∝ exp(-f_i / T)
        f_arr = np.array(f_vals)
        temp = max(self._current_temp, self._min_temperature)
        neg_f = -f_arr / temp
        neg_f -= np.max(neg_f)  # stabilise
        weights = np.exp(neg_f)
        weights /= weights.sum()

        # Blend candidate points proportionally.
        blended = np.zeros_like(x, dtype=float)
        for i, pt in enumerate(points):
            blended += weights[i] * pt

        self._current_temp *= self._anneal_rate

        return blended, f(blended)
