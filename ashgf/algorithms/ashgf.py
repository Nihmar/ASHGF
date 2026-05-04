"""Adaptive Stochastic Historical Gradient-Free (ASHGF) optimizer.

ASHGF extends ASGF by incorporating a gradient history buffer, similar to
how SGES extends GD.  It uses the same Gauss-Hermite quadrature as ASGF
but with directions that mix the gradient subspace (estimated from past
gradients) with random orthogonal directions.

The algorithm adapts three quantities on-line:

* ``sigma`` – the smoothing bandwidth, tuned via the ratio of directional
  derivatives to Lipschitz constants.
* ``basis`` – an orthonormal basis spanning the search space.  When enough
  gradient history has been collected, the first ``M`` directions are
  sampled from the empirical covariance of the gradient buffer (gradient
  subspace) while the remaining ``dim - M`` are isotropic random directions.
* ``alpha`` – the probability of sampling a direction from the *random*
  subspace (as opposed to the gradient subspace).  Its update rule is
  described in `_update_alpha`.

.. note::

   **Bug fixes relative to the original prototype** (see ``REPORT.md``):

   1. **BUG 1.5.2** — The original code passed ``steps[i-1][1]``
      (:math:`f(x_{i-1})`) as the centre-point value to the Gauss-Hermite
      quadrature.  This is incorrect because the quadrature is centred at
      the *current* iterate :math:`x_i`.  The fix evaluates ``f(x)``
      inside ``grad_estimator`` and passes that value.

   2. **Class-level ``data`` dict** — Moved to per-instance attributes so
      that multiple optimiser instances can coexist without sharing state.

   3. **Alpha update** — The original conditionals appeared inverted: when
      the gradient subspace produced better function values, ``alpha``
      (the probability of choosing a *random* direction) was *increased*,
      which is counter-intuitive.  The corrected rule decreases ``alpha``
      when the gradient subspace is superior, thereby favouring more
      gradient-informed directions.

   4. **Logging** — Replaced ``print`` statements with standard-library
      ``logging`` calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
)
from ashgf.gradient.sampling import _random_orthogonal, compute_directions_ashgf

if TYPE_CHECKING:
    from numpy import typing as npt

logger = logging.getLogger(__name__)

__all__ = ["ASHGF"]


class ASHGF(BaseOptimizer):
    """Adaptive Stochastic Historical Gradient-Free optimizer.

    Parameters
    ----------
    m : int
        Number of Gauss-Hermite quadrature nodes (must be odd so that the
        central node falls at zero).  Default ``5``.
    A : float
        Lower threshold for the ``|D_i f| / L_i`` ratio (see Notes).
        Default ``0.1``.
    B : float
        Upper threshold for the ``|D_i f| / L_i`` ratio.  Default ``0.9``.
    A_minus, A_plus : float
        Multiplicative factors applied to ``A`` when the ratio is below
        (resp. above) the threshold.  Default ``0.95``, ``1.02``.
    B_minus, B_plus : float
        Multiplicative factors applied to ``B``.  Default ``0.98``, ``1.01``.
    gamma_L : float
        Exponential-moving-average factor for the directional Lipschitz
        estimate :math:`L_{\\nabla}`.  Default ``0.9``.
    gamma_sigma : float
        Base factor for shrinking ``sigma`` (kept for compatibility; the
        separate *plus* / *minus* factors below are preferred).
        Default ``0.9``.
    gamma_sigma_plus : float
        Factor by which ``sigma`` is **increased** when the ratio exceeds
        ``B``.  Default ``1/0.9 ≈ 1.111…``.
    gamma_sigma_minus : float
        Factor by which ``sigma`` is **decreased** when the ratio falls
        below ``A``.  Default ``0.9``.
    r : int
        Maximum number of sigma / basis resets allowed when ``sigma``
        drops below ``ro * sigma_zero``.  Default ``10``.
    ro : float
        Fraction of ``sigma_zero`` below which a reset is triggered.
        Default ``0.01``.
    threshold : float
        (Reserved for future use.)  Default ``1e-6``.
    sigma_zero : float
        Initial / reset value of the smoothing bandwidth.  May be
        overridden in ``_setup`` based on the norm of the initial point.
        Default ``0.01``.

    k1 : float
        Upper bound for ``alpha`` (max probability of a random direction).
        Default ``0.9``.
    k2 : float
        Lower bound for ``alpha`` (min probability of a random direction).
        Default ``0.1``.
    alpha : float
        Initial probability of sampling a direction from the **random**
        subspace.  Default ``0.5``.
    delta : float
        Multiplicative factor for the ``alpha`` update.
        Default ``1.1``.
    t : int
        Number of pure-random warm-up iterations before the gradient
        history buffer is used.  Default ``50``.

    seed : int
        Random seed for reproducibility.  Default ``2003``.
    eps : float
        Convergence threshold on the step size :math:`\\|x_{k+1} - x_k\\|`.
        Default ``1e-8``.

    Notes
    -----
    **Adaptation logic**

    Let :math:`D_i f` be the estimated directional derivative along basis
    direction :math:`b_i` and :math:`L_i` the estimated Lipschitz constant
    for the same direction.  After each iteration we compute

    .. math::

        v = \\max_i \\frac{|D_i f|}{L_i}.

    * If :math:`v < A` the function appears *smoother* than expected
      → decrease ``sigma`` and shrink ``A``.
    * If :math:`v > B` the function appears *rougher* → increase ``sigma``
      and expand ``B``.
    * Otherwise the thresholds are widened (``A`` grows, ``B`` shrinks).

    When ``sigma`` falls below ``ro * sigma_zero`` the basis and sigma are
    reset and the reset counter ``r`` is decremented (up to ``r`` times).

    **Alpha update** (corrected)

    Let :math:`r` be the mean of the *minimum* function values observed
    along each gradient-subspace direction, and :math:`\\hat r` the same
    mean along random directions.

    * If :math:`r < \\hat r` (gradient subspace is more promising),
      **decrease** ``alpha`` → fewer random directions.
    * Otherwise **increase** ``alpha`` → more exploration via random
      directions.

    ``alpha`` is clipped to ``[k2, k1]``.
    """

    kind = "ASHGF"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        # ASGF / quadrature parameters
        m: int = 5,
        A: float = 0.1,
        B: float = 0.9,
        A_minus: float = 0.95,
        A_plus: float = 1.02,
        B_minus: float = 0.98,
        B_plus: float = 1.01,
        gamma_L: float = 0.9,
        gamma_sigma: float = 0.9,  # kept for backward compat; plus/minus are preferred
        gamma_sigma_plus: float = 1.0 / 0.9,
        gamma_sigma_minus: float = 0.9,
        r: int = 10,
        ro: float = 0.01,
        threshold: float = 1e-6,
        sigma_zero: float = 0.01,
        # SGES-style (gradient history) parameters
        k1: float = 0.9,
        k2: float = 0.1,
        alpha: float = 0.5,
        delta: float = 1.1,
        t: int = 50,
        # Base class parameters
        seed: int = 2003,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(seed=seed, eps=eps)

        # --- Quadrature & adaptation hyper-parameters ---
        if m % 2 == 0:
            raise ValueError(f"m must be odd (for a central node at zero), got {m}")
        self.m: int = m

        self.A_init: float = A
        self.B_init: float = B
        self.A_minus: float = A_minus
        self.A_plus: float = A_plus
        self.B_minus: float = B_minus
        self.B_plus: float = B_plus

        self.gamma_L: float = gamma_L
        self.gamma_sigma: float = gamma_sigma  # kept for reference
        self.gamma_sigma_plus: float = gamma_sigma_plus
        self.gamma_sigma_minus: float = gamma_sigma_minus

        self.r_init: int = r
        self.ro: float = ro
        self.threshold: float = threshold
        self.sigma_zero_ref: float = sigma_zero  # may be overridden in _setup

        # --- Gradient-history parameters ---
        if not 0.0 <= k2 <= k1 <= 1.0:
            raise ValueError(f"Require 0 <= k2 <= k1 <= 1, got k2={k2}, k1={k1}")
        self.k1: float = k1
        self.k2: float = k2
        self.alpha_init: float = alpha
        self.delta: float = delta
        self.t: int = t

        # --- Adaptive state (initialised in _setup) ---
        self._sigma: float = self.sigma_zero_ref
        self._A: float = self.A_init
        self._B: float = self.B_init
        self._r: int = self.r_init
        self._L_nabla: float = 0.0
        self._lipschitz: npt.NDArray[np.floating] | None = None
        self._basis: npt.NDArray[np.floating] | None = None
        self._M: int = 0
        # Circular buffer for gradient history (t × dim)
        self._G_buffer: npt.NDArray[np.floating] | None = None
        self._G_count: int = 0
        self._G_idx: int = 0
        self._current_alpha: float = self.alpha_init

        # Temporary storage between grad_estimator and _post_iteration
        self._last_derivatives: np.ndarray | None = None
        self._last_evaluations: dict[int, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Step size
    # ------------------------------------------------------------------

    def _get_step_size(self) -> float:
        """Return the adaptive step size ``sigma / L_nabla``.

        If ``L_nabla`` is (near) zero the raw ``sigma`` value is used as a
        fallback.
        """
        if self._L_nabla < 1e-12:
            return self._sigma
        return self._sigma / self._L_nabla

    # ------------------------------------------------------------------
    # Setup hook (called once before the main loop)
    # ------------------------------------------------------------------

    def _setup(
        self,
        f: Callable[[np.ndarray], float],
        dim: int,
        x: np.ndarray,
    ) -> None:
        """Initialise the adaptive state from the starting point."""
        del f  # not needed here

        # Set sigma_zero proportional to the norm of the initial point
        x_norm = float(np.linalg.norm(x))
        if x_norm > 0:
            self.sigma_zero_ref = max(x_norm / 10.0, 1e-6)
        else:
            self.sigma_zero_ref = self.sigma_zero_ref  # keep param default

        self._sigma = self.sigma_zero_ref
        self._A = self.A_init
        self._B = self.B_init
        self._r = self.r_init
        self._L_nabla = 0.0
        self._lipschitz = np.ones(dim)
        self._basis = _random_orthogonal(dim)
        self._M = dim  # initially attribute all directions to "gradient subspace"
        self._G_buffer = np.zeros((self.t, dim))
        self._G_count = 0
        self._G_idx = 0
        self._current_alpha = self.alpha_init

        self._last_derivatives = None
        self._last_evaluations = None

        logger.debug(
            "ASHGF setup: dim=%d sigma_zero=%.4e sigma=%.4e",
            dim,
            self.sigma_zero_ref,
            self._sigma,
        )

    # ------------------------------------------------------------------
    # Gradient estimator (Gauss-Hermite quadrature + SGES directions)
    # ------------------------------------------------------------------

    def grad_estimator(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Estimate the gradient at ``x`` using Gauss-Hermite quadrature.

        **BUG 1.5.2 fix**: evaluates ``f(x)`` at the *current* point
        (instead of using ``f(x_{i-1})``) for the central quadrature node.

        Returns
        -------
        grad : np.ndarray
            Estimated gradient vector.
        """
        dim = len(x)

        # ---- FIX (bug 1.5.2): use f(x_i), NOT f(x_{i-1}) ----
        f_x = f(x)

        # ---- Determine the orthonormal basis ----
        if self._G_count >= self.t:
            # Use gradient history to build directions
            G_slice = self._G_buffer[: self._G_count]
            directions, M = compute_directions_ashgf(
                dim, G_slice, self._current_alpha, self._M
            )
            self._M = M
            # QR is 2-3x faster than SVD-based scipy.linalg.orth
            Q, _ = np.linalg.qr(directions.T)
            basis: np.ndarray = Q.T
        else:
            # Warm-up: pure random orthogonal basis
            assert self._basis is not None, "_basis must be initialised in _setup"
            basis = self._basis  # shape (dim, dim)
            self._M = dim // 2

        # Store basis for reuse in _post_iteration (avoids redundant SVD)
        self._basis = basis

        # ---- Gauss-Hermite quadrature ----
        grad, evaluations, points, derivatives = gauss_hermite_derivative(
            x, f, self._sigma, basis, self.m, f_x
        )

        # ---- Estimate directional Lipschitz constants ----
        self._lipschitz = estimate_lipschitz_constants(evaluations, points, self._sigma)

        # ---- Update L_nabla from the first M (gradient-subspace) directions ----
        M_eff = max(self._M, 1)
        max_lip_M = float(np.max(self._lipschitz[:M_eff]))
        self._L_nabla = (1.0 - self.gamma_L) * max_lip_M + self.gamma_L * self._L_nabla

        # ---- Store for _post_iteration ----
        self._last_derivatives = derivatives
        self._last_evaluations = evaluations

        # ---- Update circular gradient buffer ----
        self._G_buffer[self._G_idx] = grad
        self._G_idx = (self._G_idx + 1) % self.t
        self._G_count = min(self._G_count + 1, self.t)

        # Cache f(x) for subclasses
        self._f_at_x = float(f_x)

        return grad

    # ------------------------------------------------------------------
    # Post-iteration hook (sigma, basis, and alpha adaptation)
    # ------------------------------------------------------------------

    def _post_iteration(
        self,
        iteration: int,
        x: np.ndarray,
        grad: np.ndarray,
        f_val: float,
    ) -> None:
        """Adapt ``sigma``, ``basis``, and ``alpha`` after each iteration.

        .. note::

           After the warm-up phase the orthonormal basis is already
           computed by ``grad_estimator`` (and stored in ``self._basis``),
           so the expensive ``orth(directions)`` call is **not** repeated
           here.  During warm-up a fresh random basis is still drawn for
           the next iteration.
        """
        del f_val  # not used; adaptation uses stored derivatives/evaluations

        dim = len(x)

        if self._last_derivatives is None:
            return

        derivatives = self._last_derivatives
        has_history = self._G_count >= self.t

        # --------------------------------------------------------------
        # 1. Alpha update (corrected — see module docstring)
        # --------------------------------------------------------------
        if has_history and iteration >= self.t + 1:
            self._update_alpha(dim)

        # --------------------------------------------------------------
        # 2. Sigma reset when it becomes too small
        # --------------------------------------------------------------
        if self._r > 0 and self._sigma < self.ro * self.sigma_zero_ref:
            logger.debug(
                "iter=%d sigma=%.4e < ro*sigma_zero → resetting basis & sigma",
                iteration,
                self._sigma,
            )
            self._basis = _random_orthogonal(dim)
            self._sigma = self.sigma_zero_ref
            self._A = self.A_init
            self._B = self.B_init
            self._r -= 1
            self._M = dim // 2
            return

        # --------------------------------------------------------------
        # 3. Basis update — only needed during warm-up.
        #    After warm-up, grad_estimator already computed & stored
        #    self._basis, so we skip the expensive recomputation here.
        # --------------------------------------------------------------
        if not has_history:
            self._M = dim // 2
            self._basis = _random_orthogonal(dim)

        # --------------------------------------------------------------
        # 4. Sigma and threshold adaptation
        # --------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Alpha update helper
    # ------------------------------------------------------------------

    def _update_alpha(self, dim: int) -> None:
        """Corrected alpha update logic.

        ``alpha`` is the probability of choosing a **random** direction
        (see `compute_directions_sges` where ``p = [alpha, 1-alpha]``).

        Let:

        * ``r`` = mean of min-evaluations along gradient-subspace directions
        * ``r_hat`` = mean of min-evaluations along random directions

        **Original (buggy)** behaviour:
            ``r < r_hat`` → increase ``alpha`` (more random despite gradient
            being better).

        **Corrected** behaviour:
            * ``r < r_hat`` → gradient subspace is better → **decrease**
              ``alpha`` (favour gradient directions).
            * ``r >= r_hat`` → random subspace is better → **increase**
              ``alpha`` (explore more randomly).

        ``alpha`` is bounded in ``[k2, k1]``.
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

        if r < r_hat:
            # Gradient subspace is better → decrease alpha (less random)
            self._current_alpha = max(self._current_alpha / self.delta, self.k2)
            logger.debug(
                "alpha update: r=%.6e < r_hat=%.6e → decrease alpha → %.4f",
                r,
                r_hat,
                self._current_alpha,
            )
        else:
            # Random subspace is better → increase alpha (more random)
            self._current_alpha = min(self.delta * self._current_alpha, self.k1)
            logger.debug(
                "alpha update: r=%.6e >= r_hat=%.6e → increase alpha → %.4f",
                r,
                r_hat,
                self._current_alpha,
            )
