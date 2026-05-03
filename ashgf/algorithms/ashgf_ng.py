"""ASHGF-NG (Next Generation): derivative-free optimiser with PID sigma control,
Bayesian alpha filtering, soft basis evolution, Nesterov momentum,
trust-region step control, adaptive quadrature, and smart restart.

ASHGF-NG extends the ASHGF algorithm with 7 new components:

1. **PID σ controller** — replaces bang-bang multiplication with a
   Proportional-Integral-Derivative controller in log-space that tracks
   a target ratio ``r_max = max_i |D_i f| / L_i``.
2. **Bayesian α filter** — replaces the binary alpha rule with a Beta
   posterior updated via a fuzzy (sigmoid) signal.
3. **Soft basis evolution** — instead of hard-resetting the orthonormal
   basis when σ drops too low, blends the old basis with a new QR
   factor via a convex combination, then re-orthonormalises.
4. **Nesterov momentum** — Nesterov accelerated gradient with an adaptive
   μ coefficient that depends on the magnitude of the function decrease.
5. **Trust-region step** — a backtracking trust-region acceptance loop
   that compares actual vs. predicted reduction.
6. **Adaptive quadrature** — the number of Gauss-Hermite nodes
   ``m ∈ {3, 5, 7}`` is chosen dynamically based on the coefficient of
   variation of the directional derivatives.
7. **Smart restart** — when stall is detected (no improvement for
   ``restart_patience`` iterations), the algorithm resets to the best
   point with halved σ.

See `ASHGF_PROPOSAL.md` for the full design document.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
)
from ashgf.gradient.sampling import _random_orthogonal, compute_directions_ashgf

logger = logging.getLogger(__name__)

__all__ = ["ASHGFNG"]


# ---------------------------------------------------------------------------
# Helper: stable sigmoid
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# ---------------------------------------------------------------------------
# ASHGF-NG
# ---------------------------------------------------------------------------


class ASHGFNG(BaseOptimizer):
    """ASHGF Next Generation optimiser.

    Replaces the binary adaptation rules of ASHGF with continuous controllers
    and adds momentum, trust-region, and restart mechanisms.

    Parameters
    ----------
    m_min : int
        Minimum number of Gauss-Hermite quadrature nodes (must be odd).
        Default ``3``.
    m_max : int
        Maximum number of Gauss-Hermite quadrature nodes (must be odd).
        Default ``7``.
    tol_m : float
        Tolerance for adaptive quadrature: if the change in CV of
        directional derivatives exceeds this, increase ``m``.
        Default ``0.1``.

    k_p : float
        PID proportional gain.  Default ``0.5``.
    k_i : float
        PID integral gain.  Default ``0.05``.
    k_d : float
        PID derivative gain.  Default ``0.1``.
    e_max : float
        Anti-windup bound on the integral error.  Default ``2.0``.
    r_target : float
        Target value for ``r_max = max_i |D_i f| / L_i``.
        Default ``0.5``.

    gamma_alpha : float
        Exponential decay factor for the Beta posterior parameters.
        Default ``0.95``.
    tau_alpha : float
        Temperature of the fuzzy sigmoid for the Bayesian α update.
        Smaller → harder decision.  Default ``0.01``.

    eta_base : float
        Blending factor for normal basis evolution.
        Default ``0.05``.
    eta_reset : float
        Blending factor when σ < ρ·σ₀ (stronger rotation).
        Default ``0.30``.
    sigma_recovery : float
        Partial recovery factor for σ in near-reset condition.
        0 = hard reset, 1 = no recovery.  Default ``0.5``.

    mu_min : float
        Minimum Nesterov momentum coefficient.  Default ``0.5``.
    mu_max : float
        Maximum Nesterov momentum coefficient.  Default ``0.95``.

    eta_accept : float
        Trust-region acceptance threshold for ρ = ared / pred.
        Default ``0.1``.
    kappa : float
        Factor for ``step_max = κ·‖x₀‖``.  Default ``5.0``.
    max_backtracks : int
        Maximum number of backtracking halvings per iteration.
        Default ``3``.

    restart_patience : int
        Number of stall iterations before smart restart.
        Default ``100``.

    k1 : float
        Upper bound for ``alpha``.  Default ``0.9``.
    k2 : float
        Lower bound for ``alpha``.  Default ``0.1``.
    alpha_init : float
        Initial value of ``alpha``.  Default ``0.5``.
    delta : float
        (Retained for compatibility; unused by Bayesian filter.)
        Default ``1.1``.
    t : int
        Number of warm-up iterations before gradient history is used.
        Default ``50``.

    a_init : float
        Legacy: lower threshold for r_max ratio.  Default ``0.1``.
    b_init : float
        Legacy: upper threshold for r_max ratio.  Default ``0.9``.
    a_minus, a_plus : float
        Legacy threshold adjustment factors.  Default ``0.95``, ``1.02``.
    b_minus, b_plus : float
        Legacy threshold adjustment factors.  Default ``0.98``, ``1.01``.
    gamma_l : float
        EMA factor for L_nabla.  Default ``0.9``.
    ro : float
        Fraction of sigma_zero below which near-reset condition triggers.
        Default ``0.01``.
    sigma_zero_ref : float
        Reference initial sigma (overridden by ‖x₀‖/10 in setup).
        Default ``0.01``.
    r_init : int
        Initial value for the reset counter (retained as safety).
        Default ``10``.

    seed : int
        Random seed.  Default ``2003``.
    eps : float
        Convergence threshold on step size.  Default ``1e-8``.
    n_jobs : int
        Number of parallel threads for function evaluations.
        Default ``1``.
    """

    kind = "ASHGF-NG"

    def __init__(
        self,
        # Quadrature
        m_min: int = 3,
        m_max: int = 7,
        tol_m: float = 0.1,
        # PID
        k_p: float = 0.5,
        k_i: float = 0.05,
        k_d: float = 0.1,
        e_max: float = 2.0,
        r_target: float = 0.5,
        # Bayesian alpha
        gamma_alpha: float = 0.95,
        tau_alpha: float = 0.01,
        # Basis evolution
        eta_base: float = 0.05,
        eta_reset: float = 0.30,
        sigma_recovery: float = 0.5,
        # Nesterov
        mu_min: float = 0.5,
        mu_max: float = 0.95,
        # Trust region
        eta_accept: float = 0.1,
        kappa: float = 5.0,
        max_backtracks: int = 3,
        # Smart restart
        restart_patience: int = 100,
        # Gradient history
        k1: float = 0.9,
        k2: float = 0.1,
        alpha_init: float = 0.5,
        delta: float = 1.1,
        t: int = 50,
        # Legacy ASGF
        a_init: float = 0.1,
        b_init: float = 0.9,
        a_minus: float = 0.95,
        a_plus: float = 1.02,
        b_minus: float = 0.98,
        b_plus: float = 1.01,
        gamma_l: float = 0.9,
        ro: float = 0.01,
        sigma_zero_ref: float = 0.01,
        r_init: int = 10,
        # Base
        seed: int = 2003,
        eps: float = 1e-8,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(seed=seed, eps=eps)

        # Validate
        if m_min % 2 == 0:
            raise ValueError(f"m_min must be odd, got {m_min}")
        if m_max % 2 == 0:
            raise ValueError(f"m_max must be odd, got {m_max}")
        if m_min > m_max:
            raise ValueError(f"m_min ({m_min}) must be ≤ m_max ({m_max})")
        if not 0.0 <= k2 <= k1 <= 1.0:
            raise ValueError(f"Require 0 ≤ k2 ≤ k1 ≤ 1, got k2={k2}, k1={k1}")

        # -- Quadrature --
        self.m_min: int = m_min
        self.m_max: int = m_max
        self.tol_m: float = tol_m

        # -- PID --
        self.k_p: float = k_p
        self.k_i: float = k_i
        self.k_d: float = k_d
        self.e_max: float = e_max
        self.r_target: float = r_target

        # -- Bayesian alpha --
        self.gamma_alpha: float = gamma_alpha
        self.tau_alpha: float = tau_alpha

        # -- Basis evolution --
        self.eta_base: float = eta_base
        self.eta_reset: float = eta_reset
        self.sigma_recovery: float = sigma_recovery

        # -- Nesterov --
        self.mu_min: float = mu_min
        self.mu_max: float = mu_max

        # -- Trust region --
        self.eta_accept: float = eta_accept
        self.kappa: float = kappa
        self.max_backtracks: int = max_backtracks

        # -- Smart restart --
        self.restart_patience: int = restart_patience

        # -- Gradient history --
        self.k1: float = k1
        self.k2: float = k2
        self.alpha_init: float = alpha_init
        self.delta: float = delta
        self.t: int = t

        # -- Legacy ASGF --
        self.a_init: float = a_init
        self.b_init: float = b_init
        self.a_minus: float = a_minus
        self.a_plus: float = a_plus
        self.b_minus: float = b_minus
        self.b_plus: float = b_plus
        self.gamma_l: float = gamma_l
        self.ro: float = ro
        self.sigma_zero_ref: float = sigma_zero_ref
        self.r_init: int = r_init

        # -- Parallelism --
        self.n_jobs: int = n_jobs

        # -- State: ASGF legacy --
        self._sigma: float = self.sigma_zero_ref
        self._sigma_zero: float = self.sigma_zero_ref
        self._a: float = self.a_init
        self._b: float = self.b_init
        self._r_resets: int = self.r_init
        self._l_nabla: float = 0.0
        self._lipschitz: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        self._last_derivatives: np.ndarray | None = None
        self._last_evaluations: np.ndarray | None = None

        # -- State: PID --
        self._e_prev: float = 0.0
        self._e_integral: float = 0.0

        # -- State: Bayesian alpha --
        self._theta1: float = 5.0
        self._theta2: float = 5.0
        self._current_alpha: float = self.alpha_init

        # -- State: gradient buffer --
        self._g_buffer: np.ndarray | None = None
        self._g_count: int = 0
        self._g_idx: int = 0
        self._m_dir: int = 0

        # -- State: Nesterov --
        self._velocity: np.ndarray | None = None
        self._prev_f_val: float = float("inf")
        self._mu: float = self.mu_min

        # -- State: smart restart --
        self._stall_count: int = 0
        self._x_best: np.ndarray | None = None
        self._f_best: float = float("inf")
        self._sigma_best: float = self.sigma_zero_ref

        # -- State: adaptive quadrature --
        self._current_m: int = self.m_min
        self._prev_delta_cv: float = 0.0

        # -- State: trust-region bounds --
        self._x0_norm: float = 0.0

    # ==================================================================
    # Step size computation
    # ==================================================================

    def _get_step_size(self) -> float:
        """Compute the adaptive step size with trust-region clipping.

        Returns
        -------
        float
            ``clip(sigma / L_nabla, 1e-10, kappa * x0_norm)``.
        """
        if self._l_nabla < 1e-12:
            raw_step = self._sigma
        else:
            raw_step = self._sigma / self._l_nabla

        max_step = self.kappa * max(self._x0_norm, 1e-6)
        return float(np.clip(raw_step, 1e-10, max_step))

    # ==================================================================
    # Setup hook
    # ==================================================================

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        """Initialise all adaptive state from the starting point."""
        del f  # not needed here

        x_norm = float(np.linalg.norm(x))
        self._x0_norm = max(x_norm, 1e-6)

        if x_norm > 0.0:
            self._sigma = max(x_norm / 10.0, 1e-6)
        else:
            self._sigma = self.sigma_zero_ref
        self._sigma_zero = self._sigma
        self._sigma_best = self._sigma

        self._a = self.a_init
        self._b = self.b_init
        self._r_resets = self.r_init
        self._l_nabla = 0.0
        self._lipschitz = np.ones(dim)
        self._m_dir = dim
        self._current_m = self.m_min

        # PID reset
        self._e_prev = 0.0
        self._e_integral = 0.0

        # Bayesian alpha reset
        self._theta1 = 5.0
        self._theta2 = 5.0
        self._current_alpha = self.alpha_init

        # Gradient buffer
        self._g_buffer = np.zeros((self.t, dim))
        self._g_count = 0
        self._g_idx = 0

        # Nesterov
        self._velocity = np.zeros(dim)
        self._prev_f_val = float("inf")
        self._mu = self.mu_min

        # Smart restart
        self._stall_count = 0
        self._x_best = x.copy()
        self._f_best = float("inf")

        # Adaptive m
        self._prev_delta_cv = 0.0

        # Initial basis
        self._basis = _random_orthogonal(dim)

        logger.debug(
            "ASHGF-NG setup: dim=%d sigma_zero=%.4e sigma=%.4e x0_norm=%.4e",
            dim,
            self._sigma_zero,
            self._sigma,
            self._x0_norm,
        )

    # ==================================================================
    # 1. PID sigma update (replaces bang-bang multiplication)
    # ==================================================================

    def _update_sigma_pid(self, r_max: float) -> None:
        """Update sigma via PID controller in log-space.

        .. math::

            e_k = r_{\\max}(k) - r_{\\text{target}}

            E_k = \\operatorname{clip}(E_{k-1} + e_k, -E_{\\max}, E_{\\max})

            \\log \\sigma_{k+1} = \\log \\sigma_k
                - (K_p e_k + K_i E_k + K_d (e_k - e_{k-1}))

            \\sigma_{k+1} = \\operatorname{clip}(\\exp(\\log \\sigma_{k+1}),
            \\; \\rho \\cdot \\sigma_0, \\; 10 \\cdot \\sigma_0)

        Parameters
        ----------
        r_max : float
            Current maximum ratio ``max_i |D_i f| / L_i``.
        """
        error = r_max - self.r_target

        # Integral with anti-windup
        self._e_integral = float(
            np.clip(self._e_integral + error, -self.e_max, self.e_max)
        )

        # Derivative
        e_deriv = error - self._e_prev

        # PID in log-space
        delta_log_sigma = (
            self.k_p * error + self.k_i * self._e_integral + self.k_d * e_deriv
        )

        log_sigma = np.log(self._sigma) - delta_log_sigma

        # Clamp sigma
        sigma_min = self.ro * self._sigma_zero
        sigma_max = 10.0 * self._sigma_zero
        self._sigma = float(np.clip(np.exp(log_sigma), sigma_min, sigma_max))

        self._e_prev = error

    # ==================================================================
    # 2. Bayesian alpha update (replaces binary alpha rule)
    # ==================================================================

    def _update_alpha_bayesian(self, dim: int) -> None:
        """Update alpha via a Bayesian Beta posterior with fuzzy signal.

        Let:

        - ``r`` = mean of min-evaluations along gradient-subspace directions
        - ``r_hat`` = mean of min-evaluations along random directions
        - ``s = sigmoid((r - r_hat) / tau)``  (fuzzy signal)

        Then update the Beta parameters:

        .. math::

            \\theta_1 \\leftarrow \\gamma \\cdot \\theta_1 + (1-\\gamma) \\cdot s

            \\theta_2 \\leftarrow \\gamma \\cdot \\theta_2 + (1-\\gamma) \\cdot (1-s)

            \\alpha = \\theta_1 / (\\theta_1 + \\theta_2)

        Alpha is soft-clipped to ``[k2, k1]``.
        """
        evaluations = self._last_evaluations
        if evaluations is None:
            return

        m_dir = self._m_dir
        if m_dir == 0 or m_dir >= dim:
            return

        # Min per direction
        min_per_dir = np.min(evaluations, axis=1)  # (dim,)

        r = float(np.mean(min_per_dir[:m_dir]))
        r_hat = float(np.mean(min_per_dir[m_dir:]))

        # Fuzzy signal
        tau = self.tau_alpha
        if tau <= 0:
            s = 1.0 if r < r_hat else 0.0
        else:
            s = _sigmoid((r - r_hat) / tau)

        # Beta posterior with exponential decay
        self._theta1 = self.gamma_alpha * self._theta1 + (1.0 - self.gamma_alpha) * s
        self._theta2 = self.gamma_alpha * self._theta2 + (1.0 - self.gamma_alpha) * (
            1.0 - s
        )

        # Expected value of Beta
        self._current_alpha = self._theta1 / (self._theta1 + self._theta2)

        # Soft clip to [k2, k1]
        self._current_alpha = float(np.clip(self._current_alpha, self.k2, self.k1))

    # ==================================================================
    # 3. Soft basis evolution via QR blending
    # ==================================================================

    def _evolve_basis(self, dim: int, grad: np.ndarray) -> None:
        """Evolve the orthonormal basis via QR blending.

        Builds a new basis candidate whose first row is aligned with the
        current gradient, then blends with the old basis via a convex
        combination before re-orthonormalising:

        .. math::

            B_{\\text{new}} = \\operatorname{QR}((1-\\eta) \\cdot B_{\\text{old}}
            + \\eta \\cdot Q_{\\text{new}})

        When σ < ρ·σ₀, a stronger rotation (``eta_reset``) is used
        and σ is partially recovered.
        """
        basis_old = self._basis
        if basis_old is None:
            return

        # Build new basis candidate: first row = gradient direction,
        # remaining rows = random, then QR
        grad_norm = float(np.linalg.norm(grad))
        seed = max(1, abs(hash(grad.tobytes())) % (2**31))
        local_rng = np.random.default_rng(seed)

        M = local_rng.standard_normal((dim, dim))

        if grad_norm > 1e-12:
            M[0, :] = grad / grad_norm

        Q_new, _ = np.linalg.qr(M)

        # Determine blending factor
        if self._sigma < self.ro * self._sigma_zero:
            # Near-reset: stronger rotation + partial sigma recovery
            clamped = max(self._sigma, self.ro * self._sigma_zero)
            self._sigma = clamped + self.sigma_recovery * (self._sigma_zero - clamped)
            eta = self.eta_reset
        else:
            eta = self.eta_base

        # Blend and re-orthonormalise
        blended = (1.0 - eta) * basis_old + eta * Q_new
        Q_blend, _ = np.linalg.qr(blended)
        self._basis = Q_blend

    # ==================================================================
    # 4. Nesterov momentum coefficient adaptation
    # ==================================================================

    def _adapt_momentum(self, delta_f: float) -> float:
        """Compute the adaptive Nesterov momentum coefficient.

        .. math::

            \\mu = \\mu_{\\min} + (\\mu_{\\max} - \\mu_{\\min})
            \\cdot \\exp(-|\\Delta f| / \\text{scale})

        where ``scale = sigma_zero * x0_norm``.

        Parameters
        ----------
        delta_f : float
            Absolute change in function value ``|f(x_k) - f(x_{k-1})|``.

        Returns
        -------
        float
            Momentum coefficient in ``[mu_min, mu_max]``.
        """
        scale = max(self._sigma_zero * self._x0_norm, 1e-8)
        mu = self.mu_min + (self.mu_max - self.mu_min) * np.exp(-abs(delta_f) / scale)
        return float(np.clip(mu, self.mu_min, self.mu_max))

    # ==================================================================
    # 5. Trust-region step control (embedded in optimize)
    # ==================================================================

    # The trust-region logic is implemented inline in `optimize`.
    # See the backtracking loop there for details.

    # ==================================================================
    # 6. Adaptive quadrature
    # ==================================================================

    def _adapt_quadrature(self, derivatives: np.ndarray) -> int:
        """Choose m based on stability of directional derivative estimates.

        Computes the coefficient of variation of the derivatives and
        compares the change to the tolerance:

        - If ``|ΔCV| > tol_m`` and m < m_max → increase m by 2.
        - If ``|ΔCV| < tol_m / 2`` and m > m_min → decrease m by 2.
        - Otherwise keep current m.

        Always ensures m is odd.

        Parameters
        ----------
        derivatives : np.ndarray, shape (d,)
            Estimated directional derivatives.

        Returns
        -------
        int
            The new (or unchanged) quadrature order.
        """
        d = len(derivatives)
        if d < 2:
            return self._current_m

        mean = float(np.mean(derivatives))
        var = float(np.mean((derivatives - mean) ** 2))
        norm = max(float(np.mean(derivatives**2)), 1e-12)
        cv = np.sqrt(var / norm)

        delta_cv = abs(cv - self._prev_delta_cv)
        self._prev_delta_cv = cv

        if delta_cv > self.tol_m and self._current_m < self.m_max:
            self._current_m = min(self._current_m + 2, self.m_max)
            # Ensure odd
            if self._current_m % 2 == 0:
                self._current_m += 1
        elif delta_cv < self.tol_m / 2.0 and self._current_m > self.m_min:
            self._current_m = max(self._current_m - 2, self.m_min)
            if self._current_m % 2 == 0:
                self._current_m += 1

        return self._current_m

    # ==================================================================
    # 7. Smart restart from best point
    # ==================================================================

    def _try_smart_restart(self, x: np.ndarray, f_val: float) -> bool:
        """Track best point and perform smart restart if stalled.

        Updates the internal stall counter and best-point records.
        If ``stall_count >= restart_patience``, restores ``x_best``
        and halves sigma.

        Parameters
        ----------
        x : np.ndarray
            Current point (modified in-place on restart).
        f_val : float
            Current function value.

        Returns
        -------
        bool
            ``True`` if a restart was performed.
        """
        if f_val < self._f_best:
            self._f_best = f_val
            self._x_best = x.copy()
            self._sigma_best = self._sigma
            self._stall_count = 0
            return False
        else:
            self._stall_count += 1

        if self.restart_patience > 0 and self._stall_count >= self.restart_patience:
            if self._x_best is not None:
                np.copyto(x, self._x_best)
                self._sigma = self._sigma_best / 2.0
                self._sigma = max(self._sigma, self.ro * self._sigma_zero)
                self._stall_count = 0
                logger.debug("ASHGF-NG: smart restart — sigma=%.4e", self._sigma)
                return True
        return False

    # ==================================================================
    # Gradient estimator
    # ==================================================================

    def grad_estimator(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Estimate the gradient via Gauss-Hermite quadrature.

        The Nesterov look-ahead is performed by the caller (optimize)
        before invoking this method.  We estimate the gradient at the
        exact point we receive.

        Parameters
        ----------
        x : np.ndarray
            Point at which to estimate the gradient (already
            look-ahead adjusted by the caller).
        f : callable
            Objective function.

        Returns
        -------
        grad : np.ndarray
            Estimated gradient vector.
        """
        dim = len(x)

        # The look-ahead is now performed in optimize() before calling us.
        # We estimate the gradient at exactly the point we receive.
        f_x = f(x)

        # Determine basis
        if self._g_count >= self.t and self._g_buffer is not None:
            g_slice = self._g_buffer[: self._g_count]
            directions, m_dir_val = compute_directions_ashgf(
                dim, g_slice, self._current_alpha, self._m_dir
            )
            self._m_dir = m_dir_val
            Q, _ = np.linalg.qr(directions.T)
            basis = Q.T
        else:
            assert self._basis is not None
            basis = self._basis
            self._m_dir = dim // 2

        self._basis = basis

        # Gauss-Hermite quadrature with adaptive m
        m = self._current_m
        grad, evaluations, nodes, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, basis, m, f_x
        )

        # Lipschitz constants
        self._lipschitz = estimate_lipschitz_constants(evaluations, nodes, self._sigma)

        # L_nabla update
        has_history = self._g_count >= self.t
        if has_history:
            m_eff = max(self._m_dir, 1)
            max_lip = float(np.max(self._lipschitz[:m_eff]))
        else:
            max_lip = float(np.max(self._lipschitz))

        self._l_nabla = (1.0 - self.gamma_l) * max_lip + self.gamma_l * self._l_nabla

        self._last_derivatives = derivatives
        self._last_evaluations = evaluations

        # Update gradient buffer
        if self._g_buffer is not None:
            self._g_buffer[self._g_idx] = grad
            self._g_idx = (self._g_idx + 1) % self.t
            self._g_count = min(self._g_count + 1, self.t)

        return grad

    # ==================================================================
    # Post-iteration hook
    # ==================================================================

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        """Adapt sigma, basis, alpha, quadrature after each iteration.

        This is called from the custom ``optimize`` loop.
        """
        del f_val
        derivatives = self._last_derivatives
        if derivatives is None:
            return

        dim = len(grad)
        has_history = self._g_count >= self.t

        # 1. Bayesian alpha update
        if has_history and iteration >= self.t + 1:
            self._update_alpha_bayesian(dim)

        # 2. Reset counter (safety only; soft evolution handles near-reset)
        if self._r_resets > 0 and self._sigma < self.ro * self._sigma_zero:
            self._r_resets -= 1

        # 3. Basis evolution (replaces hard reset)
        if has_history:
            self._evolve_basis(dim, grad)
        else:
            # Warm-up: random basis
            self._m_dir = dim // 2
            self._basis = _random_orthogonal(dim)

        # 4. PID sigma adaptation
        assert self._lipschitz is not None
        safe_ratio = np.abs(derivatives) / np.maximum(self._lipschitz, 1e-12)
        r_max = float(np.max(safe_ratio))
        self._update_sigma_pid(r_max)

        # 5. Adaptive quadrature
        self._current_m = self._adapt_quadrature(derivatives)

        # Note: smart restart is handled in optimize() itself
        # because it needs mutable access to x.

    # ==================================================================
    # Main optimization loop (overridden for trust-region + Nesterov)
    # ==================================================================

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        dim: int = 100,
        max_iter: int = 1000,
        x_init: np.ndarray | None = None,
        debug: bool = True,
        log_interval: int = 25,
        maximize: bool = False,
        patience: int | None = None,
        ftol: float | None = None,
    ) -> tuple[list[tuple[np.ndarray, float]], list[float]]:
        """Run the ASHGF-NG optimization loop.

        Overrides the base-class template method to implement:

        - Nesterov accelerated gradient updates
        - Trust-region backtracking acceptance
        - Smart restart on stall
        - Convergence checks

        Parameters
        ----------
        f : callable
            Objective function f: R^d → R.
        dim : int
            Problem dimension.
        max_iter : int
            Maximum number of iterations.
        x_init : np.ndarray or None
            Initial point. If None, use N(0, I) random vector.
        debug : bool
            If True, emit log messages during optimization.
        log_interval : int
            Log progress every ``log_interval`` iterations.
        maximize : bool
            If True, maximize f instead of minimizing.
        patience : int or None
            If set, stop early after ``patience`` iterations without
            improvement.
        ftol : float or None
            Tolerance on |f(x_new) - f(x)| for stagnation detection.

        Returns
        -------
        best_values : list of (x, f(x))
            Sequence of best points found.
        all_values : list of float
            Sequence of function values at each iteration.
        """
        # Seed RNG
        np.random.seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        # Initial point
        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = np.copy(x_init)

        # Storage
        all_values_arr: np.ndarray = np.empty(max_iter + 1)
        current_val = f(x)
        all_values_arr[0] = current_val

        x_prev: np.ndarray = x.copy()
        f_prev: float = current_val

        best_value = current_val
        best_values: list[tuple[np.ndarray, float]] = [(x.copy(), best_value)]

        # Stagnation tracking
        stall_count_base: int = 0

        if debug:
            logger.info(
                "algorithm=%-8s dim=%-4d init_val=%.6e max_iter=%d%s",
                self.kind,
                dim,
                current_val,
                max_iter,
                f" patience={patience}" if patience else "",
            )

        # Setup
        self._setup(f, dim, x)

        actual_iter = 0

        for i in range(1, max_iter + 1):
            actual_iter = i

            try:
                if debug and i % log_interval == 0:
                    logger.info(
                        "iter=%5d  f(x)=%.6e  best=%.6e  sigma=%.4e  alpha=%.3f  m=%d",
                        i,
                        all_values_arr[i - 1],
                        best_value,
                        self._sigma,
                        self._current_alpha,
                        self._current_m,
                    )

                # ---- Compute momentum (once per iteration, outside backtracking) ----
                if np.isfinite(self._prev_f_val):
                    delta_f = abs(f_prev - self._prev_f_val)
                else:
                    delta_f = self._sigma_zero * self._x0_norm
                self._mu = self._adapt_momentum(delta_f)

                # ---- Trust-region + backtracking loop ----
                accepted = False
                step = self._get_step_size()
                dim = len(x)
                grad: np.ndarray = np.zeros(dim)
                grad_accepted: np.ndarray = np.zeros(dim)
                v_new: np.ndarray = np.zeros(dim)
                x_new = x.copy()
                current_val = f_prev

                for bt in range(self.max_backtracks + 1):
                    # 1. Nesterov look-ahead (OUTSIDE grad_estimator)
                    v_old = (
                        self._velocity if self._velocity is not None else np.zeros(dim)
                    )
                    x_look = x + self._mu * v_old

                    # 2. Estimate gradient at the look-ahead point
                    grad = self.grad_estimator(x_look, f)

                    if not np.all(np.isfinite(grad)):
                        logger.warning(
                            "iter=%d: gradient contains NaN/inf — terminating", i
                        )
                        break

                    # 3. Nesterov velocity update
                    v_new = self._mu * v_old + step * grad

                    if maximize:
                        x_new = x + v_new
                    else:
                        x_new = x - v_new

                    if not np.all(np.isfinite(x_new)):
                        logger.warning("iter=%d: x contains NaN/inf — terminating", i)
                        break

                    current_val = f(x_new)

                    if not np.isfinite(current_val):
                        logger.warning(
                            "iter=%d: f(x) = %s — terminating", i, current_val
                        )
                        break

                    # 4. Trust-region acceptance
                    if maximize:
                        # For maximization, accept if improved
                        accepted = current_val > f_prev
                    else:
                        pred = step * float(np.dot(grad, grad))
                        ared = f_prev - current_val
                        rho = ared / max(pred, 1e-14)
                        accepted = rho > self.eta_accept

                        if not accepted and bt < self.max_backtracks:
                            step /= 2.0
                            logger.debug(
                                "iter=%d backtrack=%d: rho=%.4f → step halved to %.4e",
                                i,
                                bt,
                                rho,
                                step,
                            )
                            continue

                    # Store the gradient from the accepted (or last) step
                    grad_accepted = grad.copy()
                    break  # exit backtracking loop on acceptance or last backtrack

                if not accepted and np.isfinite(current_val):
                    logger.debug(
                        "iter=%d: all backtracks exhausted, accepting last attempt",
                        i,
                    )

                all_values_arr[i] = current_val

                # 4. Track best (for return value)
                improved = False
                if (maximize and current_val > best_value) or (
                    not maximize and current_val < best_value
                ):
                    best_value = current_val
                    best_values.append((x_new.copy(), best_value))
                    improved = True

                # 5a. Stagnation detection (base patience)
                if patience is not None and patience > 0:
                    if improved:
                        stall_count_base = 0
                    elif ftol is not None:
                        if abs(current_val - f_prev) < ftol:
                            stall_count_base += 1
                        else:
                            stall_count_base = 0
                    else:
                        stall_count_base += 1

                    if stall_count_base >= patience:
                        logger.info(
                            "Stopped at iteration %d (no improvement for %d iters)",
                            i,
                            patience,
                        )
                        break

                # 5b. Convergence check (max norm, every 5 iterations)
                if i % 5 == 0:
                    max_step = float(np.max(np.abs(x_new - x_prev)))
                    if max_step < self.eps:
                        logger.info("Converged at iteration %d (step < eps)", i)
                        x_prev = x_new
                        f_prev = current_val
                        break

                # Update state
                self._prev_f_val = f_prev
                x_prev = x_new.copy()
                f_prev = current_val
                # velocity: v_new = x - x_new  (since x_new = x - v_new
                # for minimization, and x_new = x + v_new for maximization)
                if maximize:
                    self._velocity = x_new - x
                else:
                    self._velocity = x - x_new
                x = x_new

                # 6. Smart restart (inline — replaces _try_smart_restart call)
                if current_val < self._f_best:
                    self._f_best = current_val
                    self._x_best = x.copy()
                    self._sigma_best = self._sigma
                    self._stall_count = 0
                else:
                    self._stall_count += 1

                if (
                    self.restart_patience > 0
                    and self._stall_count >= self.restart_patience
                ):
                    if self._x_best is not None:
                        x = self._x_best.copy()
                        self._sigma = self._sigma_best / 2.0
                        self._sigma = max(self._sigma, self.ro * self._sigma_zero)
                        self._stall_count = 0
                        logger.debug(
                            "ASHGF-NG: smart restart — sigma=%.4e", self._sigma
                        )

                # 7. Post-iteration adaptation (with REAL gradient from accepted step)
                self._post_iteration(i, x, grad_accepted, f_prev)

            except Exception:
                logger.exception("Error at iteration %d", i)
                break

        if debug:
            last_val = all_values_arr[actual_iter]
            logger.info(
                "final  f(x)=%.6e  iter=%d  best=%.6e",
                last_val,
                actual_iter,
                best_value,
            )

        all_values = all_values_arr[: actual_iter + 1].tolist()
        return best_values, all_values
