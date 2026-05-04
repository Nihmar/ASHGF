"""ASHGF-2SMA: self-tuning covariance with pure ASGF fallback.

Extends :class:`ASHGF`.  Runs with gradient-history covariance for
*decision_iter* steps.  If the average *alpha* (probability of random
directions) exceeds *alpha_threshold* at that point — meaning covariance
is not helping — the algorithm **permanently falls back to pure
ASGF-2SMA**: random orthogonal basis with Householder rotation, ASGF
sigma adaptation, and all-direction L_nabla.

On functions where covariance does help the results are identical to
ASHGF-2SMA; on all others the algorithm becomes indistinguishable from
ASGF-2SMA, giving the best of both worlds.
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
from ashgf.gradient.sampling import (
    _random_orthogonal,
    _rotate_basis_householder,
    compute_directions_ashgf,
)

logger = logging.getLogger(__name__)

__all__ = ["ASHGF2SMA"]


class ASHGF2SMA(ASHGF):
    """ASHGF with 2SMA step logic and self-tuning covariance.

    Parameters
    ----------
    warmup : int
        2S streak length for full 2x boost.  Default ``3``.
    beta_min : float
        Minimum momentum.  Default ``0.2``.
    beta_max : float
        Maximum momentum.  Default ``0.8``.
    consistency_ema : float
        EMA for gradient-consistency tracking.  Default ``0.9``.
    cov_strength : float
        ``alpha = 1 - cov_strength / √dim``.  Default ``0.6``.
    decision_iter : int
        Iteration at which to evaluate covariance usefulness.
        Default ``100``.
    alpha_threshold : float
        Disable covariance when avg alpha exceeds this.  Default ``0.92``.
    All ASHGF parameters are also accepted (see :class:`ASHGF`).
    """

    kind = "ASHGF2SMA"

    def __init__(
        self,
        warmup: int = 3,
        beta_min: float = 0.2,
        beta_max: float = 0.8,
        consistency_ema: float = 0.9,
        cov_strength: float = 0.6,
        decision_iter: int = 100,
        alpha_threshold: float = 0.92,
        **kwargs,
    ) -> None:
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.8
        super().__init__(**kwargs)

        self._warmup = warmup
        self._beta_min = beta_min
        self._beta_max = beta_max
        self._consistency_ema = consistency_ema
        self._cov_strength = cov_strength
        self._decision_iter = decision_iter
        self._alpha_threshold = alpha_threshold

        # 2SMA state
        self._velocity: np.ndarray | None = None
        self._improve_streak: int = 0
        self._prev_f_base: float | None = None
        self._prev_grad: np.ndarray | None = None
        self._consistency: float = 0.0

        # Self-tuning
        self._cov_enabled: bool = True
        self._alpha_log: list[float] = []

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)

        cov_influence = self._cov_strength / np.sqrt(dim)
        self._current_alpha = max(1.0 - cov_influence, 0.5)
        self.alpha_init = self._current_alpha

        s = np.sqrt(dim)
        self.k1 = min(1.0 - self._cov_strength / (2.0 * s), 0.99)
        self.k2 = max(0.05, 0.5 - self._cov_strength / s)

        self._velocity = np.zeros(dim)
        self._improve_streak = 0
        self._prev_f_base = None
        self._prev_grad = None
        self._consistency = 0.0
        self._cov_enabled = True
        self._alpha_log = []

    # ------------------------------------------------------------------
    # Self-tuning logic in post-iteration
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        if not self._cov_enabled:
            self._post_iteration_asgf(iteration, x, grad, f_val)
            return

        # Standard ASHGF post-iteration
        super()._post_iteration(iteration, x, grad, f_val)

        if iteration >= self.t:
            self._alpha_log.append(self._current_alpha)

        if iteration == self._decision_iter and len(self._alpha_log) >= 20:
            recent = self._alpha_log[-20:]
            avg_alpha = sum(recent) / len(recent)
            if avg_alpha > self._alpha_threshold:
                self._disable_covariance(iteration, avg_alpha)

    # ------------------------------------------------------------------
    # ASGF-style post-iteration (Householder rotation + single gamma)
    # ------------------------------------------------------------------

    def _post_iteration_asgf(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        """Pure ASGF post-iteration: Householder rotation, single gamma."""
        del iteration, f_val

        if self._last_derivatives is None or self._lipschitz is None:
            return

        derivatives = self._last_derivatives
        dim = len(x)

        # Reset check (same as ASGF)
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            self._basis = _random_orthogonal(dim)
            self._sigma = self.sigma_zero_ref
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            return

        # Householder rotation
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
            self._basis = _rotate_basis_householder(
                self._basis, grad_dir  # type: ignore[arg-type]
            )
        else:
            self._basis = _random_orthogonal(dim)

        # Sigma adaptation (single gamma_sigma, like ASGF)
        safe_lipschitz = np.maximum(self._lipschitz, 1e-12)
        ratio = np.abs(derivatives) / safe_lipschitz
        value = float(np.max(ratio))

        if value < self._A:
            self._sigma *= self.gamma_sigma_minus  # = gamma_sigma in ASGF
            self._A *= self.A_minus
        elif value > self._B:
            self._sigma *= self.gamma_sigma_plus  # = 1/gamma_sigma in ASGF
            self._B *= self.B_plus
        else:
            self._A *= self.A_plus
            self._B *= self.B_minus

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

        if self._cov_enabled and self._G_count >= self.t:
            # Covariance-based directions (ASHGF path)
            G_slice = self._G_buffer[: self._G_count]
            directions, M = compute_directions_ashgf(
                dim, G_slice, self._current_alpha, self._M
            )
            self._M = M
            Q, _ = np.linalg.qr(directions.T)
            basis: np.ndarray = Q.T
        else:
            # Pure ASGF: random orthogonal basis
            assert self._basis is not None
            basis = self._basis
            self._M = dim  # use all directions for L_nabla (ASGF behaviour)

        self._basis = basis

        grad, evaluations, points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, basis, self.m, f_x
        )

        self._lipschitz = estimate_lipschitz_constants(
            evaluations, points, self._sigma
        )

        # L_nabla: ASGF uses max of ALL Lipschitz; ASHGF uses first M
        if self._cov_enabled:
            M_eff = max(self._M, 1)
            max_lip = float(np.max(self._lipschitz[:M_eff]))
        else:
            max_lip = float(np.max(self._lipschitz))

        self._L_nabla = (1.0 - self.gamma_L) * max_lip + self.gamma_L * self._L_nabla

        self._last_derivatives = derivatives
        self._last_evaluations = evaluations

        if self._cov_enabled:
            self._G_buffer[self._G_idx] = grad
            self._G_idx = (self._G_idx + 1) % self.t
            self._G_count = min(self._G_count + 1, self.t)

        self._f_at_x = float(f_x)

        return grad

    # ------------------------------------------------------------------
    # 2SMA step logic (same for both modes)
    # ------------------------------------------------------------------

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        assert self._velocity is not None

        if self._prev_grad is not None:
            cos_sim = np.dot(self._prev_grad, grad) / max(
                np.linalg.norm(self._prev_grad) * np.linalg.norm(grad), 1e-12,
            )
            self._consistency = (
                self._consistency_ema * self._consistency
                + (1.0 - self._consistency_ema) * max(cos_sim, 0.0)
            )

        beta = self._beta_min + (self._beta_max - self._beta_min) * (
            1.0 - self._consistency
        )

        self._velocity = beta * self._velocity + (1.0 - beta) * grad
        self._prev_grad = grad.copy()

        step_size = self._get_step_size()
        direction = self._velocity if maximize else -self._velocity

        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _disable_covariance(self, iteration: int, avg_alpha: float) -> None:
        """Permanently switch to pure ASGF-2SMA behaviour."""
        self._cov_enabled = False
        self._current_alpha = 1.0
        self._M = len(self._basis) if self._basis is not None else 1
        # Switch to a fresh random basis (ASGF will rotate it via Householder)
        dim = self._basis.shape[0] if self._basis is not None else 1
        self._basis = _random_orthogonal(dim)
        # Stop buffer growth so _post_iteration_asgf handles warm-up too
        self._G_count = 0
        logger.info(
            "ASHGF-2SMA iter=%d: covariance DISABLED → pure ASGF-2SMA "
            "(avg alpha=%.4f > %.2f)",
            iteration,
            avg_alpha,
            self._alpha_threshold,
        )
