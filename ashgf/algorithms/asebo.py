r"""ASEBO: Adaptive Evolution Strategies with Active Subspaces.

ASEBO uses Principal Component Analysis (PCA) on the gradient-history
buffer to identify a low-dimensional *active subspace*, then samples
search directions from a blended covariance matrix that concentrates
sampling probability in the active subspace while maintaining
exploration across the full ambient space.

Mathematical overview
---------------------
Let :math:`G \in \mathbb{R}^{T \times d}` be the buffer of the last
:math:`T` gradient estimates.  After a warm-up of :math:`k` iterations,
we perform PCA on :math:`G` and retain the first
:math:`r = \min\{i \mid \sum_{j=1}^i \lambda_j / \sum \lambda_j \ge \tau\}`
principal components :math:`U_r \in \mathbb{R}^{r \times d}` (rows are
orthonormal).  The covariance used for direction sampling is a convex
blend of the isotropic component and the active-subspace projector:

.. math::

    \Sigma = \sigma \,
    \Bigl(
        \frac{\alpha}{d}\, I_d
        \;+\;
        \frac{1-\alpha}{r}\, U_r^{\mathsf{T}} U_r
    \Bigr),

which satisfies :math:`\operatorname{tr}(\Sigma) = \sigma` for every
choice of :math:`\alpha \in [0,1]`.  Search directions are drawn as

.. math::

    \mathbf{d}_j \sim \mathcal{N}(\mathbf{0}, \Sigma),
    \qquad j = 1, \dots, M,

and the gradient is estimated via antithetic Gaussian smoothing:

.. math::

    \widehat{\nabla f}(\mathbf{x}) =
    \frac{1}{2\sigma M}
    \sum_{j=1}^{M}
    \bigl[
        f(\mathbf{x} + \sigma \mathbf{d}_j)
      - f(\mathbf{x} - \sigma \mathbf{d}_j)
    \bigr] \, \mathbf{d}_j.

The blending parameter :math:`\alpha` is updated adaptively by
comparing the gradient norm projected onto the active subspace
versus its orthogonal complement.

.. note::

    **Design choice – blended covariance vs. probabilistic mixture.**
    The original thesis describes a *probabilistic mixture* where each
    direction is drawn either from the active subspace *or* from the
    isotropic component according to a Bernoulli trial.  This
    implementation uses a *single blended covariance* matrix that
    interpolates the two covariance structures.  Both approaches are
    valid; the blended form is simpler to sample via a single Cholesky
    factorisation and is more amenable to vectorised computation.

References
----------
* [ASEBO] (thesis / paper reference)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import _parallel_eval

# ---------------------------------------------------------------------------
# Optional dependency – scikit-learn is required for PCA-based active
# subspace detection.
# ---------------------------------------------------------------------------
try:
    from sklearn.decomposition import PCA as _PCA

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False
    _PCA = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # Re-import so the type-checker treats _PCA as the real class.
    from sklearn.decomposition import PCA as _PCA  # noqa: F811

logger = logging.getLogger(__name__)

__all__ = ["ASEBO"]


class ASEBO(BaseOptimizer):
    r"""Adaptive Evolution Strategies with Active Subspaces.

    Uses PCA on the gradient history to identify a low-dimensional
    active subspace, then samples directions from a blended covariance
    matrix.

    Parameters
    ----------
    lr : float
        Fixed learning rate (:math:`\eta`).
    sigma : float
        Base smoothing bandwidth (:math:`\sigma`).
    k : int
        Number of warm-up iterations before performing the first PCA.
    lambd : float
        Regularisation term added to the diagonal of the covariance
        matrix to guarantee positive-definiteness (Tikhonov
        regularisation for the Cholesky factorisation).
    thresh : float
        Explained-variance-ratio threshold for determining the active
        subspace dimension.
    buffer_size : int
        Maximum number of gradient vectors stored in the circular
        history buffer.  Older entries are evicted in FIFO order.
    seed : int
        Random seed for reproducibility.
    eps : float
        Convergence threshold on step-size norm.
    """

    kind = "ASEBO"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        k: int = 50,
        lambd: float = 0.1,
        thresh: float = 1e-4,
        buffer_size: int = 200,
        seed: int = 2003,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(seed=seed, eps=eps)

        if not _HAS_SKLEARN:
            raise ImportError(
                "ASEBO requires scikit-learn for PCA-based active-subspace "
                "detection.  Install it with:\n"
                "    pip install scikit-learn"
            )

        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {lr}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if k < 2:
            raise ValueError(f"k (warm-up) must be at least 2, got {k}")
        if buffer_size < k:
            raise ValueError(f"buffer_size ({buffer_size}) must be >= k ({k})")
        if not 0.0 <= lambd <= 1.0:
            raise ValueError(f"lambd must be in [0, 1], got {lambd}")
        if not 0.0 < thresh < 1.0:
            raise ValueError(f"thresh must be in (0, 1), got {thresh}")

        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.buffer_size = buffer_size
        self.lambd = lambd
        self.thresh = thresh

        # ------------------------------------------------------------------
        # Internal state (initialised in _setup)
        # ------------------------------------------------------------------
        # Circular buffer for gradient history: (buffer_size, dim)
        self._G: np.ndarray | None = None
        self._G_idx: int = 0  # next write position
        self._G_count: int = 0  # number of entries stored (≤ buffer_size)
        self._alpha: float = 1.0  # current blending parameter

    # ------------------------------------------------------------------
    # Template-method hooks
    # ------------------------------------------------------------------

    def _get_step_size(self) -> float:
        return self.lr

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        """Reset internal state before a fresh optimisation run."""
        self._G = np.zeros((self.buffer_size, dim))
        self._G_idx = 0
        self._G_count = 0
        self._alpha = 1.0

    # ------------------------------------------------------------------
    # Gradient estimator
    # ------------------------------------------------------------------

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Estimate :math:`\\nabla f(x)` via antithetic Gaussian smoothing
        with directions drawn from the active-subspace-adapted covariance.

        Implements the three critical bug fixes:
          1. divides by ``n_samples`` in the gradient formula,
          2. does **not** normalise directions to unit norm (preserving
             the :math:`\\chi(d)`-distribution required by the thesis),
          3. uses a blended covariance (documented in the class
             docstring).
        """
        dim = len(x)

        # ==============================================================
        # 1. Determine active-subspace dimension and projectors
        # ==============================================================
        if self._G is not None and self._G_count >= self.k:
            # --- PCA on the *valid* portion of the gradient-history buffer ---
            G_valid = self._G[: self._G_count]  # (count, d)
            pca = _PCA()  # type: ignore[misc]  # validated in __init__
            pca.fit(G_valid)

            var_cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = int(np.argmax(var_cumsum >= self.thresh) + 1)
            n_components = max(n_components, 10)
            n_components = min(n_components, dim)

            # Active-subspace projector  P_active = U_active^T @ U_active
            U_active = pca.components_[:n_components]  # (r, d)
            P_active = np.dot(U_active.T, U_active)  # (d, d)

            # Number of Monte Carlo directions M
            if self._G_count == self.k:
                # First PCA iteration – use dim directions for stability
                M = dim
            else:
                M = n_components
        else:
            # --- Warm-up phase: isotropic sampling ---
            U_active = np.empty((0, dim))  # empty, for alpha update
            P_active = np.zeros((dim, dim))
            n_components = dim  # placeholder (not used in covariance)
            M = dim

        # ==============================================================
        # 2. Build blended covariance
        #    Σ = σ·[ (α/d)·I_d  +  ((1-α)/r)·P_active ] + λ·I_d
        # ==============================================================
        cov = (self._alpha / dim) * np.eye(dim) + (
            (1.0 - self._alpha) / max(n_components, 1)
        ) * P_active
        cov *= self.sigma

        # Tikhonov regularisation – guarantees Cholesky success
        if self.lambd > 0:
            cov += self.lambd * self.sigma * np.eye(dim)

        # ==============================================================
        # 3. Sample directions from N(0, Σ)
        #    CRITICAL: do NOT normalise to unit length.
        #    The thesis requires E[‖d‖²] = trace(Σ) = σ  (chi-like
        #    distribution), which would be destroyed by normalisation.
        # ==============================================================
        try:
            L = np.linalg.cholesky(cov)  # (d, d) lower-tri
            Z = np.random.standard_normal((dim, M))  # (d, M)
            A = np.dot(L, Z).T  # (M, d)
        except np.linalg.LinAlgError:
            logger.warning(
                "Cholesky decomposition failed – falling back to isotropic directions."
            )
            A = np.random.randn(M, dim)

        # ==============================================================
        # 4. Antithetic gradient estimation (vectorised)
        # ==============================================================
        sigma_A = self.sigma * A  # (M, d) pre-scaled
        points: list[np.ndarray] = []
        for j in range(M):
            d = sigma_A[j]
            points.append(x + d)
            points.append(x - d)

        # Evaluate (parallel if ASHGF_N_JOBS > 1)
        results = _parallel_eval(f, points)

        # ---- Vectorised gradient assembly ----
        # diff[j] = f(x + σ·d_j) - f(x - σ·d_j)
        results_arr = np.asarray(results)  # (2M,)
        diff = results_arr[0::2] - results_arr[1::2]  # (M,)
        grad = np.dot(diff, A) / (2.0 * self.sigma * M)  # (d,) = (M,) @ (M, d)

        # ==============================================================
        # 5. Update circular gradient-history buffer
        # ==============================================================
        # Exponential decay on existing entries
        if self._G_count > 0:
            self._G[: self._G_count] *= 0.99
        self._G[self._G_idx] = grad
        self._G_idx = (self._G_idx + 1) % self.buffer_size
        self._G_count = min(self._G_count + 1, self.buffer_size)

        # ==============================================================
        # 6. Adapt blending parameter α
        #    α ← clip(‖P_ort·g‖ / ‖P_active·g‖, 0, 1)
        #
        #    Optimisation: instead of building the (d×d) matrix P_ort
        #    explicitly, we use the identity
        #        ‖P_ort·g‖² = ‖g‖² - ‖P_active·g‖²
        #    which follows from the Pythagorean theorem since the two
        #    subspaces are orthogonal complements.
        # ==============================================================
        if self._G_count >= self.k and U_active.shape[0] > 0:
            # Projection onto active subspace: g_active = U_active @ g  (r,)
            g_active = np.dot(U_active, grad)  # (r,)
            norm_active_sq = float(np.dot(g_active, g_active))
            norm_total_sq = float(np.dot(grad, grad))
            norm_ort_sq = max(norm_total_sq - norm_active_sq, 0.0)

            norm_active = np.sqrt(norm_active_sq)
            norm_ort = np.sqrt(norm_ort_sq)

            if norm_active > 1e-12:
                self._alpha = norm_ort / norm_active

            self._alpha = float(np.clip(self._alpha, 0.0, 1.0))

        return grad
