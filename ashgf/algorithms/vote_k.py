"""ASGF-2SLVK: 2SLV with vote-aware sigma decay.

When the 2SLV vote accepts a big step (indicating the gradient is
reliable), sigma decreases more aggressively, accelerating convergence.
When no big step passes the safety gate, sigma decays normally.
This is a zero-cost meta-signal from the vote mechanism.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote import VOTE

logger = logging.getLogger(__name__)

__all__ = ["VOTEK"]

_SIGMA_DECAY = 0.98


class VOTEK(VOTE):
    """2SLV with vote-accelerated sigma decay.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    sigma_decay : float
        Extra multiplier applied to sigma each time a big step is accepted.
        Default ``0.98``.
    **kwargs :
        Passed to :class:`VOTE`.
    """

    kind = "VOTEK"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        sigma_decay: float = _SIGMA_DECAY,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, **kwargs)
        self._sigma_decay = sigma_decay
        self._big_step_accepted: bool = False

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._big_step_accepted = False

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

        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            f_cur = getattr(self, "_f_at_x", f(x))
            lipschitz = self._lipschitz
            can_aniso = (
                lipschitz is not None
                and not np.all(lipschitz == 0.0)
                and float(np.mean(lipschitz)) >= 1e-12
            )

            candidates: list[tuple[np.ndarray, float]] = []

            x_uni = x + k * step_size * direction
            f_uni = f(x_uni)
            if np.isfinite(f_uni) and f_uni < f_base and f_uni < f_cur:
                candidates.append((x_uni, f_uni))

            if can_aniso:
                l_mean = float(np.mean(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_aniso = confidence / ratio
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    candidates.append((x_ani, f_ani))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                self._big_step_accepted = True
                return candidates[0]

        self._big_step_accepted = False
        return x_base, f_base

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        super()._post_iteration(iteration, x, grad, f_val)
        if self._big_step_accepted:
            self._sigma *= self._sigma_decay
