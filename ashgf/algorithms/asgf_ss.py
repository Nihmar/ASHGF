"""ASGF-SS: ASGF with Smooth Sigma scheduling.

Replaces the binary bang-bang sigma adaptation with an exponentially
smoothed target tracker that aims for the centre of the [A, B] band.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.sampling import _random_orthogonal

logger = logging.getLogger(__name__)

__all__ = ["ASGFSS"]


class ASGFSS(ASGF):
    """ASGF with smooth sigma adaptation via exponential moving average.

    Instead of multiplying/dividing sigma by discrete factors, this
    variant computes a target sigma that would bring the
    ``max|derivative|/Lipschitz`` ratio to the centre of [A, B], then
    blends toward that target with an EMA.

    Parameters
    ----------
    sigma_smooth : float
        EMA blending factor (0 = instant, 1 = no change).
        Default ``0.7``.
    sigma_min_factor : float
        Lower bound of sigma as fraction of sigma_zero.
        Default ``0.01`` (same as ASGF ro).
    sigma_max_factor : float
        Upper bound of sigma as fraction of sigma_zero.
        Default ``10.0``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFSS"

    def __init__(
        self,
        sigma_smooth: float = 0.7,
        sigma_min_factor: float = 0.01,
        sigma_max_factor: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma_smooth = sigma_smooth
        self.sigma_min_factor = sigma_min_factor
        self.sigma_max_factor = sigma_max_factor

    # ------------------------------------------------------------------
    # Override only the sigma adaptation part
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        if self._last_derivatives is None or self._lipschitz is None:
            return

        derivatives = self._last_derivatives
        dim = len(x)

        # --- Reset check (same as ASGF) ---
        if self._r > 0 and self._sigma < self.ro * self._sigma_zero:
            logger.debug(
                "ASGFSS reset: sigma=%.6e < ro*sigma_zero → reset",
                self._sigma,
            )
            self._basis = _random_orthogonal(dim, self._rng)
            self._sigma = self._sigma_zero
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            return

        # --- Basis rotation (same as ASGF) ---
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            M = self._rng.standard_normal((dim, dim))
            M[0] = grad / grad_norm
            Q, _ = np.linalg.qr(M.T)
            self._basis = Q.T
        else:
            self._basis = _random_orthogonal(dim, self._rng)

        # --- Smooth sigma adaptation (REPLACES bang-bang) ---
        safe_lipschitz = np.maximum(self._lipschitz, 1e-12)
        ratio = np.abs(derivatives) / safe_lipschitz
        value = float(np.max(ratio))

        # Target: bring ratio to midpoint of [A, B]
        mid = (self._A + self._B) / 2.0
        if value > 1e-15:
            sigma_target = self._sigma * mid / value
        else:
            sigma_target = self._sigma_max_factor * self._sigma_zero

        # Clamp
        sigma_min = self.sigma_min_factor * self._sigma_zero
        sigma_max = self.sigma_max_factor * self._sigma_zero
        sigma_target = np.clip(sigma_target, sigma_min, sigma_max)

        # EMA
        self._sigma = (self.sigma_smooth * self._sigma
                       + (1.0 - self.sigma_smooth) * sigma_target)

        # Adapt thresholds (same direction as ASGF, but gentler)
        if value < self._A:
            self._A *= self.A_minus
        elif value > self._B:
            self._B *= self.B_plus
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus

        logger.debug(
            "ASGFSS sigma=%.6e value=%.6e mid=%.6e",
            self._sigma,
            value,
            mid,
        )
