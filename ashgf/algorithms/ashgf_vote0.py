"""ASHGF-2SLV-0: Householder O(d²) basis rotation, no warm-up.

Replaces the QR-based basis construction (O(d³)) with Householder
reflection (O(d²)), exactly as in ASGF.  Uses the gradient history
buffer for a consensus direction that serves as the Householder target.
No warm-up iterations are wasted on random exploration.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf_vote import ASHGFVOTE
from ashgf.gradient.estimators import estimate_lipschitz_constants, gauss_hermite_derivative
from ashgf.gradient.sampling import _random_orthogonal, _rotate_basis_householder

logger = logging.getLogger(__name__)

__all__ = ["ASHGFVOTE0"]


class ASHGFVOTE0(ASHGFVOTE):
    """ASHGF-2SLV with Householder basis rotation (O(d²)) and no warm-up.

    Parameters
    ----------
    **kwargs :
        Passed to :class:`ASHGFVOTE`.
    """

    kind = "ASHGFVOTE0"

    def __init__(self, **kwargs) -> None:
        super().__init__(t=1, **kwargs)

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        t_actual = max(self.t, 1)
        self._G_buffer = np.zeros((t_actual, dim))
        self._G_count = 0
        self._G_idx = 0
        self._basis = _random_orthogonal(dim)

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        dim = len(x)
        f_x = f(x)

        # Gauss-Hermite quadrature along current basis
        grad, evaluations, points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, self._basis, self.m, f_x
        )

        # Per-direction Lipschitz
        self._lipschitz = estimate_lipschitz_constants(evaluations, points, self._sigma)
        max_lip = float(np.max(self._lipschitz))
        self._L_nabla = (1.0 - self.gamma_L) * max_lip + self.gamma_L * self._L_nabla

        self._last_derivatives = derivatives
        self._last_evaluations = evaluations
        self._f_at_x = float(f_x)

        # Gradient history buffer (consensus direction)
        self._G_buffer[self._G_idx] = grad
        self._G_idx = (self._G_idx + 1) % self.t
        self._G_count = min(self._G_count + 1, self.t)

        return grad

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        del f_val
        dim = len(x)

        if self._last_derivatives is None:
            return

        derivatives = self._last_derivatives

        # Reset check
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            logger.debug("iter=%d sigma=%.4e < ro → resetting", iteration, self._sigma)
            self._basis = _random_orthogonal(dim)
            self._sigma = self.sigma_zero_ref
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            return

        # Householder rotation
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
            self._basis = _rotate_basis_householder(self._basis, grad_dir)

        # Sigma and threshold adaptation
        assert self._lipschitz is not None
        safe_ratio = np.abs(derivatives) / np.maximum(self._lipschitz, 1e-12)
        value = float(np.max(safe_ratio))

        if value < self._A:
            self._sigma *= self.gamma_sigma_minus
            self._A *= self.A_minus
        elif value > self._B:
            self._sigma *= self.gamma_sigma_plus
            self._B *= self.B_plus
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus
