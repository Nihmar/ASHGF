"""ASGF-BW: ASGF with Blended Warm-start Directions.

Uses a small buffer of the last few gradients (with exponential decay
weights) to bias a subset of the basis directions, offering a
lightweight alternative to ASHGF's full covariance estimation.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF
from ashgf.gradient.sampling import _random_orthogonal

logger = logging.getLogger(__name__)

__all__ = ["ASGFBW"]


class ASGFBW(ASGF):
    """ASGF with blended warm-start directional bias.

    Maintains an EMA of recent gradient directions and aligns a subset
    of basis vectors toward it.  Unlike ASHGF, no covariance matrix is
    estimated and alpha is not adapted — the directional bias is a
    simple decaying blend.

    Parameters
    ----------
    buffer_size : int
        Number of past gradients to EMA-blend. Default ``3``.
    bias_strength : float
        Weight of the blended direction vs. random in the basis.
        Default ``0.3`` (0 = pure ASGF, 1 = fully biased).
    ema_decay : float
        Exponential decay weight for older gradients.
        Default ``0.5``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGFBW"

    def __init__(
        self,
        buffer_size: int = 3,
        bias_strength: float = 0.3,
        ema_decay: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.bias_strength = bias_strength
        self.ema_decay = ema_decay
        self._grad_buffer: np.ndarray | None = None
        self._buf_count: int = 0
        self._buf_idx: int = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        super()._setup(f, dim, x)
        self._grad_buffer = np.zeros((self.buffer_size, dim))
        self._buf_count = 0
        self._buf_idx = 0

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

        # --- Store gradient in small circular buffer ---
        assert self._grad_buffer is not None
        self._grad_buffer[self._buf_idx] = grad
        self._buf_idx = (self._buf_idx + 1) % self.buffer_size
        self._buf_count = min(self._buf_count + 1, self.buffer_size)

        # --- Reset check (same as ASGF) ---
        if self._r > 0 and self._sigma < self.ro * self._sigma_zero:
            logger.debug("ASGFBW reset sigma=%.6e → resetting", self._sigma)
            self._basis = _random_orthogonal(dim, self._rng)
            self._sigma = self._sigma_zero
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            return

        # --- Blended basis (REPLACES plain ASGF rotation) ---
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            M = self._rng.standard_normal((dim, dim))
            M[0] = grad / grad_norm

            # Build weighted blend of past gradients
            if self._buf_count > 0:
                n_biased = max(1, int(self.bias_strength * dim))
                # EMA blend: older gradients get exponentially less weight
                count = min(self._buf_count, self.buffer_size)
                start = (self._buf_idx - count) % self.buffer_size
                blend = np.zeros(dim)
                weight_sum = 0.0
                for j in range(count):
                    idx = (start + j) % self.buffer_size
                    w = self.ema_decay ** (count - 1 - j)
                    buf_grad = self._grad_buffer[idx]
                    g_norm = float(np.linalg.norm(buf_grad))
                    if g_norm > 1e-12:
                        blend += w * buf_grad / g_norm
                    weight_sum += w
                if weight_sum > 1e-15:
                    blend /= weight_sum
                    blend_norm = np.linalg.norm(blend)
                    if blend_norm > 1e-12:
                        blend /= blend_norm
                        for j in range(1, min(n_biased, dim)):
                            t = j / n_biased
                            M[j] = t * M[0] + (1.0 - t) * blend
                            mn = np.linalg.norm(M[j])
                            if mn > 1e-12:
                                M[j] /= mn

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
