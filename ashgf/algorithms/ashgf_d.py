"""ASHGF-D: ASHGF with r/r_hat history detector.

Uses the r/r_hat signal already computed by ASHGF's alpha update:
when random directions consistently outperform gradient-history
directions (r > r_hat for a rolling window), history is disabled
and the algorithm falls back to a random orthogonal basis.
History is periodically re-tested after a cooldown.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF
from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
)
from ashgf.gradient.sampling import _random_orthogonal, compute_directions_ashgf

logger = logging.getLogger(__name__)

__all__ = ["ASHGFD"]


class ASHGFD(ASHGF):
    """ASHGF with r/r_hat-based history detector.

    Parameters
    ----------
    window : int
        Rolling window for averaging ``r - r_hat``.  Default ``5``.
    margin : float
        If ``avg(r - r_hat) > margin``, random outperforms history
        and history is disabled.  Default ``0.0`` (any edge to random).
    cooldown : int
        Iterations before re-testing history after disabling it.
        Default ``15``.
    **kwargs :
        Passed to :class:`ASHGF`.
    """

    kind = "ASHGFD"

    def __init__(
        self,
        window: int = 5,
        margin: float = 0.0,
        cooldown: int = 15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._det_window = window
        self._det_margin = margin
        self._det_cooldown = cooldown
        self._use_history: bool = False
        self._disabled_until: int = 0
        self._iter_count: int = 0
        self._r_diff_history: list[float] = []

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
        self._use_history = False
        self._disabled_until = 0
        self._iter_count = 0
        self._r_diff_history = []

    # ------------------------------------------------------------------
    # Gradient estimator
    # ------------------------------------------------------------------

    def grad_estimator(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        dim = len(x)
        f_x = f(x)

        # Re-enable history after cooldown to re-test it
        if self._iter_count >= self._disabled_until:
            self._use_history = True

        # ---- Choose basis ----
        if self._G_count >= self.t and self._use_history:
            G_slice = self._G_buffer[: self._G_count]
            directions, M = compute_directions_ashgf(
                dim, G_slice, self._current_alpha, self._M
            )
            self._M = M
            Q, _ = np.linalg.qr(directions.T)
            basis: np.ndarray = Q.T
        else:
            assert self._basis is not None
            basis = self._basis
            self._M = 0

        self._basis = basis

        # ---- Gauss-Hermite quadrature ----
        grad, evaluations, points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, basis, self.m, f_x
        )

        # ---- Lipschitz ----
        self._lipschitz = estimate_lipschitz_constants(evaluations, points, self._sigma)
        M_eff = max(self._M, 1)
        max_lip_M = float(np.max(self._lipschitz[:M_eff]))
        self._L_nabla = (1.0 - self.gamma_L) * max_lip_M + self.gamma_L * self._L_nabla

        self._last_derivatives = derivatives
        self._last_evaluations = evaluations

        # Always update gradient buffer
        self._G_buffer[self._G_idx] = grad
        self._G_idx = (self._G_idx + 1) % self.t
        self._G_count = min(self._G_count + 1, self.t)

        return grad

    # ------------------------------------------------------------------
    # Post-iteration: r/r_hat detector + standard ASHGF adaptation
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        del f_val
        dim = len(x)

        if self._last_derivatives is None:
            return

        derivatives = self._last_derivatives
        has_history = self._G_count >= self.t
        self._iter_count = iteration

        # ---- r/r_hat detector (only when history is active) ----
        if has_history and self._use_history and self._last_evaluations is not None:
            evaluations = self._last_evaluations
            M = max(self._M, 0)
            if M > 0 and M < dim:
                evals_matrix = np.array([evaluations[i] for i in range(dim)])
                min_per_dir = np.min(evals_matrix, axis=1)
                r = float(np.mean(min_per_dir[:M]))
                r_hat = float(np.mean(min_per_dir[M:]))

                self._r_diff_history.append(r - r_hat)
                if len(self._r_diff_history) > self._det_window:
                    self._r_diff_history = self._r_diff_history[-self._det_window:]

                avg_diff = sum(self._r_diff_history) / len(self._r_diff_history)

                if avg_diff > self._det_margin:
                    # Random directions outperform history → disable
                    self._use_history = False
                    self._disabled_until = self._iter_count + self._det_cooldown
                    self._r_diff_history = []
                    logger.debug(
                        "detector: r-r_hat=%.4e > margin → disabling history for %d iters",
                        avg_diff,
                        self._det_cooldown,
                    )

        # ---- Alpha update (only with active history) ----
        if has_history and iteration >= self.t + 1 and self._use_history:
            self._update_alpha(dim)

        # ---- Sigma reset ----
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            logger.debug("ASHGFD sigma reset")
            self._basis = _random_orthogonal(dim)
            self._sigma = self.sigma_zero_ref
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            self._M = dim // 2
            return

        # ---- Basis update during warm-up ----
        if not has_history:
            self._M = dim // 2
            self._basis = _random_orthogonal(dim)

        # ---- Sigma and threshold adaptation ----
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
