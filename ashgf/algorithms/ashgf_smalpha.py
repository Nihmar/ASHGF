"""ASHGF-SMALPHA: ASHGF with smoothed alpha update.

Replaces the binary (bang-bang) alpha update of ASHGF with an
exponential moving average driven by a fuzzy (sigmoid) signal.

The original update:

.. math::

    \\alpha \\leftarrow
    \\begin{cases}
        \\alpha / \\delta & \\text{if } r < \\hat r \\\\
        \\alpha \\cdot \\delta & \\text{otherwise}
    \\end{cases}

is a random walk with absorbing barriers at ``k2`` and ``k1``,
causing ``alpha`` to drift to extremes and stay there.

The smoothed update replaces this with:

.. math::

    s = \\sigma\\bigl((r - \\hat r)/\\tau\\bigr)

    \\alpha \\leftarrow \\gamma \\cdot \\alpha + (1-\\gamma) \\cdot s

    \\alpha \\leftarrow \\operatorname{clip}(\\alpha,\\, k_2,\\, k_1)

where :math:`\\sigma` is the sigmoid function, :math:`\\tau` is a
temperature parameter (smaller → harder decision), and :math:`\\gamma`
is an exponential decay factor.

Parameters
----------
gamma_alpha : float
    Exponential decay factor for the smoothed alpha.
    Higher → more inertia, slower adaptation.  Default ``0.95``.
tau_alpha : float
    Temperature of the sigmoid.  Smaller → harder decision
    (closer to the original binary behaviour).
    Default ``0.01``.

All other parameters are inherited from :class:`ASHGF`.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGFSMALPHA"]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


class ASHGFSMALPHA(ASHGF):
    """ASHGF with smoothed (EMA) alpha update."""

    kind = "ASHGF-SMALPHA"

    def __init__(
        self,
        gamma_alpha: float = 0.95,
        tau_alpha: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not 0.0 < gamma_alpha <= 1.0:
            raise ValueError(f"gamma_alpha must be in (0, 1], got {gamma_alpha}")
        if tau_alpha < 0:
            raise ValueError(f"tau_alpha must be >= 0, got {tau_alpha}")

        self.gamma_alpha: float = gamma_alpha
        self.tau_alpha: float = tau_alpha

    # ------------------------------------------------------------------
    # Override alpha update with smoothed version
    # ------------------------------------------------------------------

    def _update_alpha(self, dim: int) -> None:
        """Smoothed alpha update via exponential moving average.

        Let:

        - ``r`` = mean of min-evaluations along gradient-subspace directions
        - ``r_hat`` = mean of min-evaluations along random directions
        - ``s = sigmoid((r - r_hat) / tau)``  (fuzzy signal)

        Then update:

        .. math::

            \\alpha \\leftarrow \\gamma \\cdot \\alpha + (1-\\gamma) \\cdot s

        ``alpha`` is clipped to ``[k2, k1]``.
        """
        if self._last_evaluations is None:
            return

        evaluations = self._last_evaluations
        M = max(self._M, 0)

        # Build evaluation matrix: (dim, m) and take min per direction
        evals_matrix = np.array([evaluations[i] for i in range(dim)])
        min_per_dir = np.min(evals_matrix, axis=1)  # (dim,)

        # r: mean of minima over gradient-subspace directions (0:M)
        r: float | None = float(np.mean(min_per_dir[:M])) if M > 0 else None

        # r_hat: mean of minima over random directions (M:dim)
        r_hat: float | None = float(np.mean(min_per_dir[M:])) if M < dim else None

        if r is None or r_hat is None:
            return

        # Fuzzy signal via sigmoid
        tau = self.tau_alpha
        if tau <= 0:
            # Hard decision (no smoothing) — matches original behaviour
            s = 1.0 if r < r_hat else 0.0
        else:
            # Scale (r - r_hat) by tau; note: r and r_hat can be very large
            # for some functions, so we normalise by a robust scale.
            scale = max(abs(r) + abs(r_hat), 1e-12)
            raw_signal = (r - r_hat) / (tau * scale)
            s = _sigmoid(raw_signal)

        # Exponential moving average
        self._current_alpha = (
            self.gamma_alpha * self._current_alpha + (1.0 - self.gamma_alpha) * s
        )

        # Clip to bounds
        self._current_alpha = float(np.clip(self._current_alpha, self.k2, self.k1))

        logger.debug(
            "alpha smoothed: r=%.6e r_hat=%.6e s=%.4f → alpha=%.4f",
            r,
            r_hat,
            s,
            self._current_alpha,
        )
