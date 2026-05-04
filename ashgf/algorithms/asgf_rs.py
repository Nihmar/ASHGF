"""ASGF-RS: ASGF with Restart Scheduling.

Periodically restarts from the best point found with annealed sigma
when progress stalls, giving the algorithm fresh exploration chances.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.sampling import _random_orthogonal

logger = logging.getLogger(__name__)

__all__ = ["ASGFRS"]


class ASGFRS(ASGF):
    """ASGF with restart scheduling.

    When no improvement is seen for ``restart_window`` iterations,
    the algorithm restarts from the best point found with sigma
    annealed by ``annealing`` each restart.  The usual sigma-based
    reset mechanism of ASGF remains active as a fallback.

    Parameters
    ----------
    restart_window : int
        Iterations without improvement before a scheduled restart.
    annealing : float
        Multiplicative factor applied to sigma at each restart
        (sigma *= annealing^restart_count).
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFRS"

    def __init__(
        self,
        restart_window: int = 50,
        annealing: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.restart_window = restart_window
        self.annealing = annealing
        self._best_x: np.ndarray | None = None
        self._best_f: float = float("inf")
        self._restart_k: int = 0
        self._iters_stalled: int = 0
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
        self._restart_k = 0
        self._iters_stalled = 0
        self._needs_restart = False

    def _before_gradient(self, x: np.ndarray) -> np.ndarray:
        if self._needs_restart and self._best_x is not None:
            self._needs_restart = False
            return self._best_x.copy()
        return x

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        # --- Track best point ---
        if f_val < self._best_f:
            self._best_f = f_val
            self._best_x = x.copy()
            self._iters_stalled = 0
        else:
            self._iters_stalled += 1

        # --- Scheduled restart ---
        if (self._iters_stalled >= self.restart_window
                and self._restart_k < self.r_init):
            logger.debug(
                "ASGFRS restart k=%d from f_best=%.4e, sigma=%.2e -> %.2e",
                self._restart_k,
                self._best_f,
                self._sigma,
                self._sigma_zero * (self.annealing ** (self._restart_k + 1)),
            )
            self._needs_restart = True
            self._restart_k += 1
            self._sigma = self._sigma_zero * (self.annealing ** self._restart_k)
            self._basis = _random_orthogonal(len(x), self._rng)
            self._A = self.A_init
            self._B = self.B_init
            self._L_nabla = 0.0
            self._lipschitz = np.ones(len(x))
            self._iters_stalled = 0
            return

        # --- Normal ASGF adaptation ---
        super()._post_iteration(iteration, x, grad, f_val)
