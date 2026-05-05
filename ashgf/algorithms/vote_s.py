"""ASGF-2SLVC: 2SLV with conditional sqrt candidate.

Adds the sqrt-weighted candidate (confidence / sqrt(ratio)) only when
the Lipschitz spread is in a moderate range [1.5, 4.0].  Outside this
range the algorithm falls back to uniform + full-aniso (standard 2SLV),
avoiding the decision noise that hurt 2SLV2 on most functions.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.vote import VOTE

logger = logging.getLogger(__name__)

__all__ = ["VOTES"]

_SPREAD_MIN = 1.5
_SPREAD_MAX = 4.0


class VOTES(VOTE):
    """2SLV with conditional sqrt candidate.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    spread_min : float
        Minimum Lipschitz spread to add the sqrt candidate.
        Default ``1.5``.
    spread_max : float
        Maximum Lipschitz spread for the sqrt candidate.
        Default ``4.0``.
    **kwargs :
        Passed to :class:`VOTE`.
    """

    kind = "VOTES"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        spread_min: float = _SPREAD_MIN,
        spread_max: float = _SPREAD_MAX,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, **kwargs)
        self._spread_min = spread_min
        self._spread_max = spread_max

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
                l_spread = float(np.max(lipschitz)) / max(l_mean, 1e-12)

                k_aniso = confidence / ratio
                x_ani = x + (1.0 + k_aniso) * step_size * direction
                f_ani = f(x_ani)
                if np.isfinite(f_ani) and f_ani < f_base and f_ani < f_cur:
                    candidates.append((x_ani, f_ani))

                # Conditional sqrt candidate
                if self._spread_min < l_spread < self._spread_max:
                    k_sqrt = confidence / np.sqrt(ratio)
                    x_sqrt = x + (1.0 + k_sqrt) * step_size * direction
                    f_sqrt = f(x_sqrt)
                    if np.isfinite(f_sqrt) and f_sqrt < f_base and f_sqrt < f_cur:
                        candidates.append((x_sqrt, f_sqrt))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                return candidates[0]

        return x_base, f_base
