"""ASGF-2SLB: 2SL with adaptive isotropic blend.

Blends the uniform step and the Lipschitz-weighted step:
``k_final = (1 - beta) * k_uniform + beta * k_lipschitz``.
The blend coefficient ``beta`` is adapted online based on 2x success
rate: it increases when anisotropy helps, decreases when it hurts.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLB"]


class ASGF2SLB(ASGF):
    """2SL with adaptive blend between isotropic and anisotropic steps.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    beta : float
        Starting blend coefficient (0 = fully isotropic, 1 = fully 2SL).
        Default ``0.5``.
    beta_lr : float
        Learning rate for beta adaptation.  Default ``0.1``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SLB"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        beta: float = 0.5,
        beta_lr: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._lip_clip = lip_clip
        self._beta = beta
        self._beta_lr = beta_lr
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

        self._last_uniform_accepted: bool = False
        self._last_aniso_accepted: bool = False

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None
        self._last_uniform_accepted = False
        self._last_aniso_accepted = False

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

        if confidence > 0.0:
            lipschitz = self._lipschitz
            use_uniform = (
                lipschitz is None
                or np.all(lipschitz == 0.0)
                or float(np.mean(lipschitz)) < 1e-12
            )

            k_uniform = confidence * np.ones_like(direction)

            if use_uniform:
                k_aniso = k_uniform.copy()
            else:
                l_mean = float(np.mean(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_aniso = confidence / ratio

            k_vec = (1.0 - self._beta) * k_uniform + self._beta * k_aniso
            max_k = float(np.max(k_vec))
            if max_k < 0.01:
                return x_base, f_base

            x_big = x + (1.0 + k_vec) * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))

            big_accepted = (
                np.isfinite(f_big) and f_big < f_base and f_big < f_cur
            )

            if big_accepted:
                aniso_better = float(np.mean(k_aniso > k_uniform)) > 0.5
                uniform_better = float(np.mean(k_uniform > k_aniso)) > 0.5

                if aniso_better and not uniform_better:
                    self._beta = min(1.0, self._beta + self._beta_lr)
                elif uniform_better and not aniso_better:
                    self._beta = max(0.0, self._beta - self._beta_lr)

                return x_big, f_big

        return x_base, f_base
