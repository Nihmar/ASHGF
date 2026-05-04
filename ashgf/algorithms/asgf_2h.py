"""ASGF-2H: frequency-adaptive Try-2x with self-tuning parameters.

Warmup and cooldown self-calibrate based on the 2x acceptance rate
over a rolling window.  Functions where 2x consistently helps get
more frequent attempts; functions where it rarely helps get fewer.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2H"]


class ASGF2H(ASGF):
    """Frequency-adaptive Try-2x with self-tuning warmup/cooldown.

    Parameters
    ----------
    warmup : int
        Initial consecutive improvements before 2x is attempted.
        Default ``3``.
    cooldown : int
        Initial cooldown after a rejected 2x attempt.
        Default ``5``.
    window : int
        Rolling window over which acceptance rate is measured.
        Default ``20``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2H"

    def __init__(
        self,
        warmup: int = 3,
        cooldown: int = 5,
        window: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._cooldown = cooldown
        self._window = window
        self._improve_streak: int = 0
        self._skip_until: int = 0
        self._prev_f: float | None = None
        self._iter_count: int = 0
        # Rolling acceptance tracking
        self._attempts: list[bool] = []  # True = accepted, False = rejected

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._skip_until = 0
        self._prev_f = None
        self._iter_count = 0
        self._attempts = []

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

        # Track improvement streak
        if self._prev_f is not None and f_base < self._prev_f:
            self._improve_streak += 1
        else:
            self._improve_streak = 0
        self._prev_f = f_base
        self._iter_count += 1

        # Self-tune warmup/cooldown from rolling acceptance rate
        accept_rate = 0.0
        if self._attempts:
            accept_rate = sum(self._attempts) / len(self._attempts)
        warmup = max(1, self._warmup - int(5 * accept_rate))
        cooldown = max(1, self._cooldown + int(10 * (1.0 - accept_rate)))

        # Try 2x when confident
        if self._improve_streak >= warmup and self._iter_count > self._skip_until:
            x_big = x + 2.0 * step_size * direction
            f_big = f(x_big)
            if np.isfinite(f_big) and f_big < f_base:
                self._attempts.append(True)
                if len(self._attempts) > self._window:
                    self._attempts = self._attempts[-self._window:]
                return x_big, f_big
            # Rejected
            self._attempts.append(False)
            if len(self._attempts) > self._window:
                self._attempts = self._attempts[-self._window:]
            self._skip_until = self._iter_count + cooldown

        return x_base, f_base
