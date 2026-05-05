"""ASHGF-2SLV-2G: dual gradient vote.

Computes TWO gradients per iteration — one via ASHGF's gradient-history
basis and one via ASGF's Householder-rotated basis — then generates
step candidates from both and picks the best that passes the safety gate.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf_2slv import ASHGF2SLV
from ashgf.gradient.estimators import estimate_lipschitz_constants, gauss_hermite_derivative
from ashgf.gradient.sampling import _random_orthogonal, _rotate_basis_householder

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2SLV2G"]


class ASHGF2SLV2G(ASHGF2SLV):
    """ASHGF-2SLV with dual-gradient vote (ASHGF + ASGF).

    Parameters
    ----------
    **kwargs :
        Passed to :class:`ASHGF2SLV`.
    """

    kind = "ASHGF2SLV2G"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._basis_asgf: np.ndarray | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._basis_asgf = _random_orthogonal(dim)

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction_ashgf = grad if maximize else -grad

        # --- ASGF-style gradient ---
        assert self._basis_asgf is not None
        f_x = f(x)
        grad_asgf, _evaluations, _points, _derivatives = gauss_hermite_derivative(
            x, f, self._sigma, self._basis_asgf, self.m, f_x
        )
        direction_asgf = grad_asgf if maximize else -grad_asgf

        # --- Base step (use ASHGF gradient, same as standard flow) ---
        x_base = x + step_size * direction_ashgf
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

            # Candidates from ASHGF gradient
            x_uni_ashgf = x + k * step_size * direction_ashgf
            f_uni_ashgf = f(x_uni_ashgf)
            if np.isfinite(f_uni_ashgf) and f_uni_ashgf < f_base and f_uni_ashgf < f_cur:
                candidates.append((x_uni_ashgf, f_uni_ashgf))

            # Candidates from ASGF gradient
            x_uni_asgf = x + k * step_size * direction_asgf
            f_uni_asgf = f(x_uni_asgf)
            if np.isfinite(f_uni_asgf) and f_uni_asgf < f_base and f_uni_asgf < f_cur:
                candidates.append((x_uni_asgf, f_uni_asgf))

            # Anisotropic candidates (both gradients)
            if can_aniso:
                l_mean = float(np.mean(lipschitz))
                ratio = np.clip(lipschitz / l_mean, 1e-12, self._lip_clip)
                k_aniso = confidence / ratio

                x_ani_ashgf = x + (1.0 + k_aniso) * step_size * direction_ashgf
                f_ani_ashgf = f(x_ani_ashgf)
                if np.isfinite(f_ani_ashgf) and f_ani_ashgf < f_base and f_ani_ashgf < f_cur:
                    candidates.append((x_ani_ashgf, f_ani_ashgf))

                x_ani_asgf = x + (1.0 + k_aniso) * step_size * direction_asgf
                f_ani_asgf = f(x_ani_asgf)
                if np.isfinite(f_ani_asgf) and f_ani_asgf < f_base and f_ani_asgf < f_cur:
                    candidates.append((x_ani_asgf, f_ani_asgf))

            if candidates:
                candidates.sort(key=lambda t: t[1])
                return candidates[0]

        return x_base, f_base

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        # Standard ASHGF post iteration (handles sigma, alpha, basis)
        super()._post_iteration(iteration, x, grad, f_val)

        # Also rotate the ASGF basis via Householder
        dim = len(x)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            self._basis_asgf = _rotate_basis_householder(
                self._basis_asgf, grad / grad_norm
            )
