"""ASGF-CD: ASGF with Conjugate-like Directions.

Aligns a small number of basis vectors to combinations of the current
and past gradients instead of just one, giving a lightweight conjugate-
direction bias without a full gradient-history buffer.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.sampling import _random_orthogonal

logger = logging.getLogger(__name__)

__all__ = ["ASGFCD"]


class ASGFCD(ASGF):
    """ASGF with conjugate-like direction mixing.

    Instead of aligning only the first basis vector to the current
    gradient, this variant aligns a few vectors to combinations of
    the current gradient and an EMA of past gradient directions.

    Parameters
    ----------
    num_structured : int
        Number of basis vectors built from gradient combinations.
        Default ``3`` (1 = plain ASGF behaviour).
    ema_beta : float
        Decay factor for the EMA of past gradient direction.
        Default ``0.6``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFCD"

    def __init__(
        self,
        num_structured: int = 3,
        ema_beta: float = 0.6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_structured = num_structured
        self.ema_beta = ema_beta
        self._ema_dir: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Override only the basis-building part of _post_iteration
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        super()._setup(f, dim, x)
        self._ema_dir = None

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
            logger.debug("ASGFCD reset sigma=%.6e → resetting", self._sigma)
            self._basis = _random_orthogonal(dim, self._rng)
            self._sigma = self._sigma_zero
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            self._ema_dir = None
            return

        # --- Update EMA of gradient direction ---
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            cur_dir = grad / grad_norm
            if self._ema_dir is None:
                self._ema_dir = cur_dir.copy()
            else:
                self._ema_dir = (self.ema_beta * self._ema_dir
                                 + (1.0 - self.ema_beta) * cur_dir)
                ema_norm = float(np.linalg.norm(self._ema_dir))
                if ema_norm > 1e-12:
                    self._ema_dir /= ema_norm

        # --- Conjugate-like basis (REPLACES plain ASGF rotation) ---
        if grad_norm > 1e-12:
            M = self._rng.standard_normal((dim, dim))
            M[0] = grad / grad_norm

            k = min(self.num_structured, dim)
            # Additional structured directions: blends of current gradient and EMA
            if self._ema_dir is not None and k > 1:
                M[1] = self._ema_dir
                for j in range(2, k):
                    t = (j - 1) / (k - 1)
                    blend = t * M[0] + (1.0 - t) * self._ema_dir
                    blend_norm = np.linalg.norm(blend)
                    if blend_norm > 1e-12:
                        M[j] = blend / blend_norm

            Q, _ = np.linalg.qr(M.T)
            self._basis = Q.T
        else:
            self._basis = _random_orthogonal(dim, self._rng)

        # --- Sigma adaptation (same as ASGF) ---
        safe_lipschitz = np.maximum(self._lipschitz, 1e-12)
        ratio = np.abs(derivatives) / safe_lipschitz
        value = float(np.max(ratio))

        if value < self._A:
            self._sigma *= self.gamma_sigma
            self._A *= self.A_minus
        elif value > self._B:
            self._sigma /= self.gamma_sigma
            self._B *= self.B_plus
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus
