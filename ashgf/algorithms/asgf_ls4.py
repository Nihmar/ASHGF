"""ASGF-LS4: Meta-controller for line search.

Monitors the quality of the trajectory over a rolling window.  When
the trajectory degrades (function value increasing), line search is
disabled and the algorithm falls back to pure ASGF.  It re-enables
when good progress is detected again.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGFLS4"]


class ASGFLS4(ASGF):
    """ASGF with meta-controlled line search.

    Parameters
    ----------
    candidates : tuple of float
        Step-size multipliers.  Default ``(0.25, 0.5, 1.0, 2.0)``.
    window : int
        Number of iterations over which to measure trajectory quality.
        Default ``10``.
    degr_threshold : float
        If the best f in the window has deteriorated by more than this
        fraction, disable LS.  Default ``0.0`` (any degradation).
    improve_threshold : float
        Relative improvement over the window that re-enables LS.
        Default ``0.1`` (10% improvement).
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFLS4"

    def __init__(
        self,
        candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        window: int = 10,
        degr_threshold: float = 0.0,
        improve_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ls_candidates = candidates
        self._window = window
        self._degr_threshold = degr_threshold
        self._improve_threshold = improve_threshold
        self._ls_enabled: bool = True
        self._f_history: list[float] = []

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        super()._setup(f, dim, x)
        self._ls_enabled = True
        self._f_history = []

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

        if self._ls_enabled:
            # Greedy line search
            best_f = float("inf")
            best_x = None
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
        else:
            # Pure ASGF step
            x_new = x + step_size * direction
            best_x, best_f = x_new, f(x_new)

        return best_x, best_f

    # ------------------------------------------------------------------
    # Post-iteration: meta-controller
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        self._f_history.append(f_val)
        # Keep a bounded history
        max_len = self._window * 5
        if len(self._f_history) > max_len:
            self._f_history = self._f_history[-max_len:]

        if len(self._f_history) >= self._window:
            recent = self._f_history[-self._window:]
            f_start = recent[0]
            f_end = recent[-1]
            abs_start = max(abs(f_start), 1e-15)

            improvement = (f_start - f_end) / abs_start  # > 0 means improving

            if improvement < -self._degr_threshold:
                # f is increasing — LS may be harming
                self._ls_enabled = False
                logger.debug("LS4: LS disabled (improvement=%.4f)", improvement)
            elif improvement > self._improve_threshold:
                # Good progress — re-enable LS
                self._ls_enabled = True
                logger.debug("LS4: LS enabled  (improvement=%.4f)", improvement)

        super()._post_iteration(iteration, x, grad, f_val)
