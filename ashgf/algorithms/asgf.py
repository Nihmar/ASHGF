"""Adaptive Stochastic Gradient-Free (ASGF) optimizer."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
)
from ashgf.gradient.sampling import _random_orthogonal, _rotate_basis_householder

logger = logging.getLogger(__name__)

__all__ = ["ASGF"]


class ASGF(BaseOptimizer):
    """Adaptive Stochastic Gradient-Free optimizer.

    Uses Gauss-Hermite quadrature for gradient estimation with
    adaptive smoothing parameter ``sigma`` and basis rotation.

    The algorithm maintains per-direction Lipschitz constants and
    uses them to adapt the smoothing bandwidth.  When ``sigma``
    drops below a fraction ``ro`` of its initial value, the
    orthonormal basis is reset and ``sigma`` is restored.

    Parameters
    ----------
    m : int
        Number of Gauss-Hermite quadrature points (must be odd, so
        that the central node is exactly at 0).
    A : float
        Lower threshold for the ``|derivative| / Lipschitz`` ratio.
    B : float
        Upper threshold for the ``|derivative| / Lipschitz`` ratio.
    A_minus : float
        Multiplicative decrease factor for ``A`` when the ratio
        falls below ``A``.
    A_plus : float
        Multiplicative increase factor for ``A`` when the ratio
        lies inside the safe region ``[A, B]``.
    B_minus : float
        Multiplicative decrease factor for ``B`` when the ratio
        lies inside the safe region ``[A, B]``.
    B_plus : float
        Multiplicative increase factor for ``B`` when the ratio
        exceeds ``B``.
    gamma_L : float
        Exponential-smoothing factor for the global Lipschitz
        estimate ``L_nabla``.
    gamma_sigma : float
        Multiplicative factor for ``sigma`` changes (``sigma`` is
        multiplied or divided by ``gamma_sigma``).
    r : int
        Number of allowed basis resets before resets are disabled.
    ro : float
        Fraction of the initial ``sigma`` that triggers a reset.
    threshold : float
        Threshold for adaptive quadrature (reserved for future use).
    sigma_zero : float
        Fallback initial ``sigma`` when ``‖x₀‖ == 0``.
    seed : int
        Random seed for reproducibility.
    eps : float
        Convergence threshold on step length.
    """

    kind = "ASGF"

    def __init__(
        self,
        m: int = 5,
        A: float = 0.1,
        B: float = 0.9,
        A_minus: float = 0.95,
        A_plus: float = 1.02,
        B_minus: float = 0.98,
        B_plus: float = 1.01,
        gamma_L: float = 0.9,
        gamma_sigma: float = 0.9,
        r: int = 2,
        ro: float = 0.01,
        threshold: float = 1e-6,
        sigma_zero: float = 0.01,
        seed: int = 2003,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(seed=seed, eps=eps)

        # -- Quadrature --------------------------------------------------
        if m % 2 == 0:
            raise ValueError(f"m must be odd for Gauss-Hermite, got {m}")
        self.m = m

        # -- Adaptation thresholds ---------------------------------------
        self.A_init = A
        self.B_init = B
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.B_minus = B_minus
        self.B_plus = B_plus

        # -- Adaptation rates --------------------------------------------
        self.gamma_L = gamma_L
        self.gamma_sigma = gamma_sigma

        # -- Reset logic -------------------------------------------------
        self.r_init = r
        self.ro = ro

        # -- Misc --------------------------------------------------------
        self.threshold = threshold
        self.sigma_zero_fallback = sigma_zero

        # ------------------------------------------------------------------
        # Adaptive state (initialised lazily in ``_setup``)
        # ------------------------------------------------------------------
        self._sigma: float = sigma_zero
        self._sigma_zero: float = sigma_zero
        self._A: float = A
        self._B: float = B
        self._r: int = r
        self._L_nabla: float = 0.0
        self._lipschitz: np.ndarray | None = None
        self._basis: np.ndarray | None = None

        # Transient storage shared between ``grad_estimator`` and
        # ``_post_iteration``.
        self._last_derivatives: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Hooks from BaseOptimizer
    # ------------------------------------------------------------------

    def _get_step_size(self) -> float:
        """Learning rate :math:`\\alpha = \\sigma / L_{\\nabla}`.

        When ``L_nabla`` is numerically zero the step size defaults
        to ``sigma`` to avoid division-by-zero.
        """
        if self._L_nabla < 1e-12:
            return self._sigma
        return self._sigma / self._L_nabla

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        """Initialise adaptive parameters before the first iteration.

        Sets the initial ``sigma`` proportional to ``‖x₀‖``,
        generates a random orthonormal basis, and resets the
        Lipschitz and threshold trackers to their default values.
        """
        x_norm = float(np.linalg.norm(x))
        if x_norm > 0:
            self._sigma = max(x_norm / 10.0, 1e-6)
        else:
            self._sigma = self.sigma_zero_fallback

        # Remember the *effective* initial sigma for reset logic.
        self._sigma_zero = self._sigma

        self._A = self.A_init
        self._B = self.B_init
        self._r = self.r_init
        self._L_nabla = 0.0
        self._lipschitz = np.ones(dim)
        self._basis = _random_orthogonal(dim, self._rng)

        logger.debug(
            "ASGF setup: sigma=%.6e sigma_zero=%.6e dim=%d",
            self._sigma,
            self._sigma_zero,
            dim,
        )

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Estimate the gradient using Gauss-Hermite quadrature.

        Computes directional derivatives along each basis vector,
        updates the per-direction Lipschitz constants and the
        smoothed global constant ``L_nabla``, and stores the raw
        directional derivatives for the post-iteration adaptation
        step.

        Returns
        -------
        np.ndarray
            Gradient estimate :math:`\\nabla f(x)`.
        """
        f_x = f(x)
        assert self._basis is not None, "_setup must be called before grad_estimator"

        grad, _evaluations, _points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, self._basis, self.m, f_x
        )

        # Update per-direction Lipschitz constants.
        self._lipschitz = estimate_lipschitz_constants(
            _evaluations, _points, self._sigma
        )

        # Exponentially smoothed global Lipschitz constant.
        max_lip = float(np.max(self._lipschitz))
        self._L_nabla = (1.0 - self.gamma_L) * max_lip + self.gamma_L * self._L_nabla

        # Keep derivatives for the adaptation logic in ``_post_iteration``.
        self._last_derivatives = derivatives

        # Cache f(x) so subclasses can reuse it (avoids double eval)
        self._f_at_x = float(f_x)

        return grad

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        """Adapt ``sigma``, the orthonormal basis, and the thresholds.

        The adaptation rule compares the magnitude of the directional
        derivatives against the per-direction Lipschitz estimates:

        * If the max ratio < ``A`` → shrink ``sigma``.
        * If the max ratio > ``B`` → grow ``sigma``.
        * Otherwise (safe region) → widen ``[A, B]``.

        When ``sigma`` becomes too small relative to its initial
        value, a *reset* is triggered: a fresh random basis is drawn
        and ``sigma`` is restored.
        """
        if self._last_derivatives is None or self._lipschitz is None:
            return

        derivatives = self._last_derivatives
        dim = len(x)

        # -- Reset check -------------------------------------------------
        if self._r > 0 and self._sigma < self.ro * self._sigma_zero:
            logger.debug(
                "ASGF reset: sigma=%.6e < ro*sigma_zero=%.6e (r=%d)",
                self._sigma,
                self.ro * self._sigma_zero,
                self._r,
            )
            self._basis = _random_orthogonal(dim, self._rng)
            self._sigma = self._sigma_zero
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            return

        # -- Basis rotation (Householder: O(d²) instead of QR O(d³)) ---
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
            self._basis = _rotate_basis_householder(self._basis, grad_dir)
        else:
            self._basis = _random_orthogonal(dim, self._rng)

        # -- Threshold-based sigma adaptation ----------------------------
        safe_lipschitz = np.maximum(self._lipschitz, 1e-12)
        ratio = np.abs(derivatives) / safe_lipschitz
        value = float(np.max(ratio))

        if value < self._A:
            self._sigma *= self.gamma_sigma
            self._A *= self.A_minus
            logger.debug(
                "value=%.6e < A=%.6e → shrink sigma=%.6e",
                value,
                self._A / self.A_minus,  # A before update
                self._sigma,
            )
        elif value > self._B:
            self._sigma /= self.gamma_sigma
            self._B *= self.B_plus
            logger.debug(
                "value=%.6e > B=%.6e → grow sigma=%.6e",
                value,
                self._B / self.B_plus,  # B before update
                self._sigma,
            )
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus
            logger.debug(
                "value=%.6e in [A,B] → widen A=%.6e B=%.6e",
                value,
                self._A,
                self._B,
            )
