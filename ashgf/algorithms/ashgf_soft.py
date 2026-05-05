"""ASHGF-SOFT: ASHGF with soft basis evolution.

Replaces the aggressive basis reset of ASHGF with a smooth blending
(convex combination) of the old basis and a new QR decomposition,
followed by re-orthonormalisation.

When ``sigma`` drops below ``ro * sigma_zero``, instead of a hard
reset that destroys all accumulated information, the basis is rotated
more strongly (higher blending factor ``eta_reset``) and ``sigma`` is
partially recovered (rather than fully restored).

Mathematical details
--------------------
Let :math:`B_{\\text{old}}` be the current orthonormal basis and
:math:`Q_{\\text{new}}` a new QR factor whose first row is aligned with
the current gradient.  The updated basis is:

.. math::

    B_{\\text{new}} = \\operatorname{QR}\\bigl(
        (1 - \\eta) B_{\\text{old}} + \\eta \\, Q_{\\text{new}}
    \\bigr)

where :math:`\\eta = \\eta_{\\text{base}}` normally, and
:math:`\\eta = \\eta_{\\text{reset}}` when
:math:`\\sigma < \\rho \\cdot \\sigma_0`.  In the latter case,
:math:`\\sigma` is partially recovered:

.. math::

    \\sigma \\leftarrow
    \\sigma_{\\min} + \\lambda_{\\text{rec}} (\\sigma_0 - \\sigma_{\\min})

where :math:`\\sigma_{\\min} = \\max(\\sigma, \\rho \\cdot \\sigma_0)`.

Parameters
----------
eta_base : float
    Blending factor for normal basis evolution.  Default ``0.05``.
eta_reset : float
    Blending factor when σ < ρ·σ₀ (stronger rotation).  Default ``0.30``.
sigma_recovery : float
    Partial recovery factor for σ in near-reset condition.
    0 = hard reset, 1 = no recovery.  Default ``0.5``.

All other parameters are inherited from :class:`ASHGF`.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGFSOFT"]


class ASHGFSOFT(ASHGF):
    """ASHGF with soft basis evolution."""

    kind = "ASHGF-SOFT"

    def __init__(
        self,
        eta_base: float = 0.05,
        eta_reset: float = 0.30,
        sigma_recovery: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # -- Soft basis evolution parameters --
        if not 0.0 <= eta_base <= 1.0:
            raise ValueError(f"eta_base must be in [0, 1], got {eta_base}")
        if not 0.0 <= eta_reset <= 1.0:
            raise ValueError(f"eta_reset must be in [0, 1], got {eta_reset}")
        if not 0.0 <= sigma_recovery <= 1.0:
            raise ValueError(f"sigma_recovery must be in [0, 1], got {sigma_recovery}")

        self.eta_base: float = eta_base
        self.eta_reset: float = eta_reset
        self.sigma_recovery: float = sigma_recovery

    # ------------------------------------------------------------------
    # Post-iteration hook — override basis update only
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        """Adapt sigma, basis (soft), and alpha after each iteration."""
        dim = len(x)

        if self._last_derivatives is None:
            return

        derivatives = self._last_derivatives
        has_history = self._G_count >= self.t

        # --------------------------------------------------------------
        # 1. Alpha update (inherited from ASHGF)
        # --------------------------------------------------------------
        if has_history and iteration >= self.t + 1:
            self._update_alpha(dim)

        # --------------------------------------------------------------
        # 2. Sigma reset counter (decrement if near-reset, but don't
        #    hard-reset — soft evolution handles it below)
        # --------------------------------------------------------------
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            self._r -= 1

        # --------------------------------------------------------------
        # 3. Basis evolution (SOFT — replaces hard reset)
        # --------------------------------------------------------------
        if has_history:
            self._evolve_basis(dim, grad)
        else:
            # Warm-up: random basis
            from ashgf.gradient.sampling import _random_orthogonal

            self._M = dim // 2
            self._basis = _random_orthogonal(dim)

        # --------------------------------------------------------------
        # 4. Sigma and threshold adaptation (inherited bang-bang)
        # --------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Soft basis evolution via QR blending
    # ------------------------------------------------------------------

    def _evolve_basis(self, dim: int, grad: np.ndarray) -> None:
        """Evolve the orthonormal basis via QR blending.

        Builds a new basis candidate whose first row is aligned with the
        current gradient, then blends with the old basis via a convex
        combination before re-orthonormalising.

        When σ < ρ·σ₀, a stronger rotation (``eta_reset``) is used
        and σ is partially recovered.
        """
        basis_old = self._basis
        if basis_old is None:
            return

        # Build new basis candidate: first row = gradient direction,
        # remaining rows = random, then QR
        grad_norm = float(np.linalg.norm(grad))
        # Use a deterministic seed based on gradient hash for reproducibility
        seed = max(1, abs(hash(grad.tobytes())) % (2**31))
        local_rng = np.random.default_rng(seed)

        M = local_rng.standard_normal((dim, dim))

        if grad_norm > 1e-12:
            M[0, :] = grad / grad_norm

        Q_new, _ = np.linalg.qr(M)

        # Determine blending factor
        sigma_min = self.ro * self.sigma_zero_ref
        if self._sigma < sigma_min:
            # Near-reset: stronger rotation + partial sigma recovery
            clamped = max(self._sigma, sigma_min)
            self._sigma = clamped + self.sigma_recovery * (
                self.sigma_zero_ref - clamped
            )
            eta = self.eta_reset
            logger.debug(
                "ASHGF-SOFT near-reset: sigma %.4e → %.4e  eta=%.2f",
                self._sigma,
                clamped + self.sigma_recovery * (self.sigma_zero_ref - clamped),
                eta,
            )
        else:
            eta = self.eta_base

        # Blend and re-orthonormalise
        blended = (1.0 - eta) * basis_old + eta * Q_new
        Q_blend, _ = np.linalg.qr(blended)
        self._basis = Q_blend
