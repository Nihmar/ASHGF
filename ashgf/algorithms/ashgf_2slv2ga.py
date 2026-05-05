"""ASHGF-2SLV-2GA: alternate between ASGF and ASHGF gradients.

Each iteration uses a single gradient source — ASGF (Householder basis)
on odd iterations, ASHGF (history basis) on even iterations — and
applies the 2SLV step vote to it.  Over N iterations the algorithm
experiences both gradient types, but each iteration costs only one
quadrature evaluation (vs two for the full dual-gradient).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf_2slv import ASHGF2SLV
from ashgf.gradient.estimators import estimate_lipschitz_constants, gauss_hermite_derivative
from ashgf.gradient.sampling import _random_orthogonal, _rotate_basis_householder

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2SLV2GA"]


class ASHGF2SLV2GA(ASHGF2SLV):
    """Alternating-gradient 2SLV.

    Parameters
    ----------
    warmup : int
        Streak length at which full boost is reached.  Default ``3``.
    lip_clip : float
        Clipping factor for the Lipschitz-to-mean ratio.  Default ``5.0``.
    **kwargs :
        Passed to :class:`ASHGF2SLV`.
    """

    kind = "ASHGF2SLV2GA"

    def __init__(
        self,
        warmup: int = 3,
        lip_clip: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(warmup=warmup, lip_clip=lip_clip, **kwargs)
        self._iter_count: int = 0
        self._basis_asgf: np.ndarray | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._iter_count = 0
        self._basis_asgf = _random_orthogonal(dim)

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        self._iter_count += 1
        step_size = self._get_step_size()

        # Alternate gradient source
        if self._iter_count % 2 == 1:
            # ---- ASGF-style gradient (Householder basis) ----
            assert self._basis_asgf is not None
            f_x = f(x)
            grad_asgf, _, _, _ = gauss_hermite_derivative(
                x, f, self._sigma, self._basis_asgf, self.m, f_x
            )
            direction = grad_asgf if maximize else -grad_asgf
            # Update Lipschitz for the 2SLV vote
            if self._basis_asgf is not None:
                grad_norm = float(np.linalg.norm(grad_asgf))
                if grad_norm > 1e-12:
                    self._basis_asgf = _rotate_basis_householder(
                        self._basis_asgf, grad_asgf / grad_norm
                    )
        else:
            # ---- ASHGF-style gradient (from parent's grad_estimator) ----
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
                return candidates[0]

        return x_base, f_base
