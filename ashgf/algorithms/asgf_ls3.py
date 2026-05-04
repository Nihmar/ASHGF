"""ASGF-LS3: Restart + LS fallback.

Uses greedy line search as the primary strategy, but when progress
stalls for ``stall_window`` iterations, restarts from the best point
found with pure ASGF (no line search) to escape plateaus.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.sampling import _random_orthogonal

logger = logging.getLogger(__name__)

__all__ = ["ASGFLS3"]


class ASGFLS3(ASGF):
    """ASGF with line search + restart fallback.

    Parameters
    ----------
    candidates : tuple of float
        Step-size multipliers.  Default ``(0.25, 0.5, 1.0, 2.0)``.
    stall_window : int
        Iterations without improvement before triggering a restart.
        Default ``30``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFLS3"

    def __init__(
        self,
        candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        stall_window: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ls_candidates = candidates
        self._stall_window = stall_window
        self._best_x: np.ndarray | None = None
        self._best_f: float = float("inf")
        self._stall_count: int = 0
        self._needs_restart: bool = False

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
        self._best_x = x.copy()
        self._best_f = f(x)
        self._stall_count = 0
        self._needs_restart = False

    def _before_gradient(self, x: np.ndarray) -> np.ndarray:
        if self._needs_restart and self._best_x is not None:
            self._needs_restart = False
            return self._best_x.copy()
        return x

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
            return x.copy(), f(x)

        return best_x, best_f

    # ------------------------------------------------------------------
    # Post-iteration: track best point and trigger restart on stall
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        if f_val < self._best_f:
            self._best_f = f_val
            self._best_x = x.copy()
            self._stall_count = 0
        else:
            self._stall_count += 1

        # Trigger restart: jump to best point, reset sigma/basis
        if self._stall_count >= self._stall_window and self._r > 0:
            logger.debug(
                "ASGFLS3 restart k: stalled=%d, best_f=%.4e",
                self._stall_count,
                self._best_f,
            )
            self._needs_restart = True
            self._stall_count = 0
            self._sigma = self._sigma_zero
            self._basis = _random_orthogonal(len(x), self._rng)
            self._A = self.A_init
            self._B = self.B_init
            self._L_nabla = 0.0
            self._lipschitz = np.ones(len(x))
            self._r -= 1
            return

        super()._post_iteration(iteration, x, grad, f_val)
