"""Shared adaptive-smoothing machinery for ASGF and ASHGF.

Implements Algorithm "Parameter update" (ASHGFSubroutine) from
``thesis/chapters/thealgorithm.tex``: threshold-based adaptation of the smoothing radius
sigma against the normalized directional-derivative ratios, with a finite number of
full resets to initial conditions when sigma collapses below ``rho * sigma_zero``.

The frozen prototypes implement exactly this scheme; our version only removes their
mutable class-level state and makes every parameter explicit configuration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AdaptiveSmoothingState:
    """Mutable state of the parameter-update subroutine for one run."""

    def __init__(
        self,
        *,
        sigma_zero: float,
        a: float = 0.1,
        b: float = 0.9,
        a_minus: float = 0.95,
        a_plus: float = 1.02,
        b_minus: float = 0.98,
        b_plus: float = 1.01,
        gamma_l: float = 0.9,
        gamma_sigma: float = 0.9,
        resets: int = 10,
        rho: float = 0.01,
    ) -> None:
        if sigma_zero <= 0:
            raise ValueError("sigma_zero must be positive")
        self.sigma_zero = sigma_zero
        self.a_init = a
        self.b_init = b
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.b_minus = b_minus
        self.b_plus = b_plus
        self.gamma_l = gamma_l
        self.gamma_sigma = gamma_sigma
        self.rho = rho
        self.resets_left = resets

        self.sigma = sigma_zero
        self.a = a
        self.b = b
        self.l_nabla = 0.0

    def update_lipschitz_momentum(self, l_g: float) -> None:
        """L_nabla <- (1 - gamma_L) L_G + gamma_L L_nabla."""
        self.l_nabla = (1.0 - self.gamma_l) * l_g + self.gamma_l * self.l_nabla

    @property
    def learning_rate(self) -> float:
        """lambda = sigma / L_nabla with a numerical floor on L_nabla."""
        return self.sigma / max(self.l_nabla, 1e-30)

    def adapt(
        self, derivatives: NDArray[np.float64], lipschitz_constants: NDArray[np.float64]
    ) -> bool:
        """Run the threshold update after one completed iteration.

        Returns True iff a full reset happened (the caller must then draw a fresh random
        orthonormal basis); False otherwise (caller rebuilds directions normally).
        """
        if self.resets_left > 0 and self.sigma < self.rho * self.sigma_zero:
            self.sigma = self.sigma_zero
            self.a = self.a_init
            self.b = self.b_init
            self.resets_left -= 1
            return True

        abs_deriv = np.abs(derivatives)
        positive_lip = lipschitz_constants > 0.0
        ratio = np.empty_like(abs_deriv)
        ratio[positive_lip] = abs_deriv[positive_lip] / lipschitz_constants[positive_lip]
        # Degenerate directions: consistent pairs give 0; inconsistencies push sigma up.
        ratio[~positive_lip] = np.where(abs_deriv[~positive_lip] > 0.0, np.inf, 0.0)
        value = float(np.max(ratio))
        if value < self.a:
            self.sigma *= self.gamma_sigma
            self.a *= self.a_minus
        elif value > self.b:
            self.sigma /= self.gamma_sigma
            self.b *= self.b_plus
        else:
            self.a *= self.a_plus
            self.b *= self.b_minus
        return False
