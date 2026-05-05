"""ASHGF-PID: ASHGF with PID sigma controller.

Replaces the bang-bang sigma adaptation of ASHGF with a
Proportional-Integral-Derivative (PID) controller in log-space.

The PID controller tracks a target ratio ``r_max = max_i |D_i f| / L_i``
and adjusts ``sigma`` continuously, eliminating the oscillations that
plague the original bang-bang rule.

Mathematical details
--------------------
Let :math:`e_k = r_{\\max}(k) - r_{\\text{target}}` be the error at
iteration :math:`k`.  The integral error is clipped for anti-windup:

.. math::

    E_k = \\operatorname{clip}(E_{k-1} + e_k,\\, -E_{\\max},\\, E_{\\max})

The log-sigma update is:

.. math::

    \\log \\sigma_{k+1} = \\log \\sigma_k
    - \\bigl(K_p e_k + K_i E_k + K_d (e_k - e_{k-1})\\bigr)

and :math:`\\sigma` is clipped to :math:`[\\rho\\cdot\\sigma_0,\\, 10\\cdot\\sigma_0]`.

Parameters
----------
k_p : float
    Proportional gain.  Default ``0.5``.
k_i : float
    Integral gain.  Default ``0.05``.
k_d : float
    Derivative gain.  Default ``0.1``.
e_max : float
    Anti-windup bound on the integral error.  Default ``2.0``.
r_target : float
    Target value for ``r_max = max_i |D_i f| / L_i``.
    Default ``0.5``.

All other parameters are inherited from :class:`ASHGF`.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGFPID"]


class ASHGFPID(ASHGF):
    """ASHGF with PID sigma controller."""

    kind = "ASHGF-PID"

    def __init__(
        self,
        # PID parameters
        k_p: float = 0.5,
        k_i: float = 0.05,
        k_d: float = 0.1,
        e_max: float = 2.0,
        r_target: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # -- PID gains --
        self.k_p: float = k_p
        self.k_i: float = k_i
        self.k_d: float = k_d
        self.e_max: float = e_max
        self.r_target: float = r_target

        # -- PID state (initialised in _setup) --
        self._e_prev: float = 0.0
        self._e_integral: float = 0.0

    # ------------------------------------------------------------------
    # Setup hook
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        """Initialise adaptive state + PID state."""
        super()._setup(f, dim, x)
        self._e_prev = 0.0
        self._e_integral = 0.0

    # ------------------------------------------------------------------
    # Post-iteration hook — override sigma adaptation only
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        """Adapt sigma via PID, then call parent for everything else."""
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
        # 2. Sigma reset when it becomes too small (keep safety net)
        # --------------------------------------------------------------
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            logger.debug(
                "iter=%d sigma=%.4e < ro*sigma_zero → resetting basis & sigma",
                iteration,
                self._sigma,
            )
            from ashgf.gradient.sampling import _random_orthogonal

            self._basis = _random_orthogonal(dim)
            self._sigma = self.sigma_zero_ref
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            self._M = dim // 2
            # Also reset PID state
            self._e_prev = 0.0
            self._e_integral = 0.0
            return

        # --------------------------------------------------------------
        # 3. Basis update — only needed during warm-up
        # --------------------------------------------------------------
        if not has_history:
            from ashgf.gradient.sampling import _random_orthogonal

            self._M = dim // 2
            self._basis = _random_orthogonal(dim)

        # --------------------------------------------------------------
        # 4. PID sigma adaptation (REPLACES bang-bang)
        # --------------------------------------------------------------
        assert self._lipschitz is not None
        safe_ratio = np.abs(derivatives) / np.maximum(self._lipschitz, 1e-12)
        r_max = float(np.max(safe_ratio))

        # PID in log-space
        error = r_max - self.r_target

        # Integral with anti-windup
        self._e_integral = float(
            np.clip(self._e_integral + error, -self.e_max, self.e_max)
        )

        # Derivative
        e_deriv = error - self._e_prev

        # PID output
        delta_log_sigma = (
            self.k_p * error + self.k_i * self._e_integral + self.k_d * e_deriv
        )

        log_sigma = np.log(self._sigma) - delta_log_sigma

        # Clamp sigma to reasonable bounds
        sigma_min = self.ro * self.sigma_zero_ref
        sigma_max = 10.0 * self.sigma_zero_ref
        self._sigma = float(np.clip(np.exp(log_sigma), sigma_min, sigma_max))

        self._e_prev = error

        # --------------------------------------------------------------
        # 5. Still update A, B for compatibility (though not used by PID)
        # --------------------------------------------------------------
        # We keep A, B updated so that the reset logic (which restores
        # them) remains consistent, but they no longer drive sigma.
        if r_max < self._A:
            self._A *= self.A_minus
        elif r_max > self._B:
            self._B *= self.B_plus
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus
