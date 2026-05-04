"""ASGF-AQ: ASGF with Adaptive Quadrature depth.

Varies the number of Gauss-Hermite quadrature nodes ``m`` based on
local smoothness: uses ``m=3`` in smooth regions and increases to
``m=5`` or ``m=7`` in rough regions.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
)

logger = logging.getLogger(__name__)

__all__ = ["ASGFAQ"]


class ASGFAQ(ASGF):
    """ASGF with adaptive Gauss-Hermite quadrature depth.

    Starts with ``m=3`` nodes.  After each gradient estimation, checks
    the consistency of derivatives estimated from even vs. odd nodes.
    If the discrepancy exceeds ``tol_m``, ``m`` is increased for the
    next iteration; if it is very small, ``m`` may be decreased.

    Parameters
    ----------
    m_min : int
        Minimum number of quadrature nodes (must be odd). Default ``3``.
    m_max : int
        Maximum number of quadrature nodes (must be odd). Default ``7``.
    tol_m : float
        Tolerance on the relative discrepancy between even- and odd-node
        derivative estimates. Default ``0.1``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFAQ"

    def __init__(
        self,
        m_min: int = 3,
        m_max: int = 7,
        tol_m: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.m_min = m_min
        self.m_max = m_max
        self.tol_m = tol_m
        self._current_m: int = 3

    # ------------------------------------------------------------------
    # Override gradient estimator with adaptive m
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        super()._setup(f, dim, x)
        self._current_m = self.m_min

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        f_x = f(x)
        assert self._basis is not None

        # Use current_m instead of fixed self.m
        grad, evaluations, points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, self._basis, self._current_m, f_x
        )

        self._lipschitz = estimate_lipschitz_constants(
            evaluations, points, self._sigma
        )

        max_lip = float(np.max(self._lipschitz))
        self._L_nabla = (1.0 - self.gamma_L) * max_lip + self.gamma_L * self._L_nabla

        self._last_derivatives = derivatives

        # --- Adapt m for next iteration ---
        self._current_m = self._adapt_m(evaluations)

        return grad

    def _adapt_m(self, evaluations: np.ndarray) -> int:
        """Choose m for the next iteration based on local smoothness.

        Cross-validates derivative estimates from even-indexed and
        odd-indexed quadrature nodes.  If they disagree significantly,
        increases m.
        """
        m_current = self._current_m
        if m_current <= 3 or evaluations.shape[1] < 5:
            # Not enough data for cross-validation; keep current
            return m_current

        mid = m_current // 2
        # Use nodes excluding the central one for parity split
        even_idx = [j for j in range(m_current) if j != mid and j % 2 == 0]
        odd_idx = [j for j in range(m_current) if j != mid and j % 2 == 1]
        if len(even_idx) == 0 or len(odd_idx) == 0:
            return m_current

        # Approximate derivatives from even and odd subsets
        # We use the stored derivatives as reference
        from ashgf.gradient.estimators import _get_gauss_hermite

        _p_nodes, p_w, _weights = _get_gauss_hermite(m_current)
        quad_scale = 2.0 / (self._sigma * np.sqrt(np.pi))

        even_deriv = quad_scale * (evaluations[:, even_idx] @ p_w[even_idx])
        odd_deriv = quad_scale * (evaluations[:, odd_idx] @ p_w[odd_idx])

        ref = np.maximum(np.abs(even_deriv), 1e-15)
        delta = np.mean(np.abs(even_deriv - odd_deriv) / ref)

        if delta > self.tol_m:
            return min(m_current + 2, self.m_max)
        elif delta < self.tol_m / 2.0:
            return max(m_current - 2, self.m_min)
        return m_current
