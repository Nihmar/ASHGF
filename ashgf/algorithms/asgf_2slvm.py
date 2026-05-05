"""ASGF-2SLVM: 2SLV with improvement-magnitude memory.

Tracks the average relative improvement from each candidate type
(uniform vs anisotropic) over a sliding window.  The candidate type
that has historically produced larger improvements gets a selection
bonus, helping the vote escape locally-good-but-globally-bad choices.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf_2slv import ASGF2SLV

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SLVM"]

_MEMORY_EPS = 1e-14
_WINDOW = 10


class ASGF2SLVM(ASGF2SLV):
    """2SLV with improvement-magnitude memory.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    memory_window : int
        Number of past acceptances to remember per type.  Default ``10``.
    memory_eps : float
        Bonus magnitude for the historically better candidate type.
        Default ``1e-14``.
    **kwargs :
        Passed to :class:`ASGF2SLV`.
    """

    kind = "ASGF2SLVM"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        memory_window: int = _WINDOW,
        memory_eps: float = _MEMORY_EPS,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, **kwargs)
        self._memory_window = memory_window
        self._memory_eps = memory_eps
        self._improvements: dict[str, deque[float]] = {
            "uni": deque(maxlen=memory_window),
            "aniso": deque(maxlen=memory_window),
        }

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improvements = {
            "uni": deque(maxlen=self._memory_window),
            "aniso": deque(maxlen=self._memory_window),
        }

    def _mean_improvement(self, ttype: str) -> float:
        dq = self._improvements.get(ttype)
        if not dq:
            return 0.0
        return float(np.mean(list(dq)))

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

            candidates: list[tuple[np.ndarray, float, str]] = []

            x_uni = x + k * step_size * direction
            f_uni = f(x_uni)
            if np.isfinite(f_uni) and f_uni < f_base and f_uni < f_cur:
                candidates.append((x_uni, f_uni, "uni"))

            if can_aniso:
                l_mean = float(np.mean(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_aniso = confidence / ratio
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    candidates.append((x_ani, f_ani, "aniso"))

            if candidates:
                # Compute bonuses from historical improvement averages
                avg_uni = self._mean_improvement("uni")
                avg_aniso = self._mean_improvement("aniso")
                bonus_uni = 0.0
                bonus_aniso = 0.0
                if avg_aniso > avg_uni * 1.05 and len(self._improvements["aniso"]) >= 3:
                    bonus_aniso = self._memory_eps
                elif avg_uni > avg_aniso * 1.05 and len(self._improvements["uni"]) >= 3:
                    bonus_uni = self._memory_eps

                def key(t):
                    b = bonus_aniso if t[2] == "aniso" else bonus_uni
                    return t[1] + b

                candidates.sort(key=key)
                chosen = candidates[0]

                # Record the improvement from the winning candidate
                rel_imp = (f_cur - chosen[1]) / max(abs(f_cur), 1e-12)
                self._improvements[chosen[2]].append(max(rel_imp, 0.0))

                return chosen[0], chosen[1]

        return x_base, f_base
