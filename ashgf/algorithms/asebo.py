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
        if not 0.0 <= lambd <= 1.0:
            raise ValueError(f"lambd must be in [0, 1], got {lambd}")
        if not 0.0 < thresh < 1.0:
            raise ValueError(f"thresh must be in (0, 1), got {thresh}")

        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.lambd = lambd
        self.thresh = thresh

        # ------------------------------------------------------------------
        # Internal state (initialised in _setup)
        # ------------------------------------------------------------------
        self._G: np.ndarray | None = None  # gradient-history buffer (T, d)
        self._alpha: float = 1.0  # current blending parameter

    # ------------------------------------------------------------------
    # Template-method hooks
    # ------------------------------------------------------------------

    def _get_step_size(self) -> float:
        return self.lr

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        """Reset internal state before a fresh optimisation run."""
        self._G = None
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
        if self._G is not None and self._G.shape[0] >= self.k:
            # --- PCA on gradient-history buffer ---
            pca = _PCA()  # type: ignore[misc]  # validated in __init__
            pca.fit(self._G)

            var_cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = int(np.argmax(var_cumsum >= self.thresh) + 1)
            n_components = max(n_components, 10)
            n_components = min(n_components, dim)

            # Active-subspace projector  P_active = U_active^T @ U_active
            U_active = pca.components_[:n_components]  # (r, d)
            P_active = np.dot(U_active.T, U_active)  # (d, d)

            # Orthogonal-complement projector  P_ort = U_ort^T @ U_ort
            U_ort = pca.components_[n_components:]  # (d-r, d)
            if U_ort.shape[0] > 0:
                P_ort = np.dot(U_ort.T, U_ort)
            else:
                P_ort = np.zeros((dim, dim))

            # Number of Monte Carlo directions M
            if self._G.shape[0] == self.k:
                # First PCA iteration – use more samples for stability
                M = 100
            else:
                M = n_components
        else:
            # --- Warm-up phase: isotropic sampling ---
            P_active = np.zeros((dim, dim))
            P_ort = np.eye(dim)
            n_components = dim  # placeholder (not used in covariance)
            M = 100

        # ==============================================================
        # 2. Build blended covariance
        #    Σ = σ·[ (α/d)·I_d  +  ((1-α)/r)·P_active ] + λ·I_d
        # ==============================================================
        cov = (self._alpha / dim) * np.eye(dim) + (
            (1.0 - self._alpha) / n_components
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
        # 4. Antithetic gradient estimation
        # ==============================================================
        grad = np.zeros(dim)
        for j in range(M):
            d = A[j].reshape(x.shape)
            f_plus = f(x + self.sigma * d)
            f_minus = f(x - self.sigma * d)
            grad += (f_plus - f_minus) * d.reshape(grad.shape)

        # FIXED (Bug 1.5.2): divide by M – was missing in original,
        # resulting in a gradient scaled by M.
        grad /= 2.0 * self.sigma * M

        # ==============================================================
        # 5. Update gradient-history buffer (exponential decay)
        # ==============================================================
        g_row = grad.reshape(1, -1)
        if self._G is None:
            self._G = g_row
        else:
            self._G *= 0.99
            self._G = np.vstack([self._G, g_row])

        # ==============================================================
        # 6. Adapt blending parameter α
        #    α ← clip(‖P_ort·g‖ / ‖P_active·g‖, 0, 1)
        #    If the gradient lies mostly in the active subspace, α → 0
        #    and future directions concentrate there.  If the gradient
        #    has a large orthogonal component, α → 1 and directions
        #    become more isotropic.
        # ==============================================================
        if self._G is not None and self._G.shape[0] >= self.k:
            norm_ort = float(np.linalg.norm(np.dot(grad, P_ort)))
            norm_active = float(np.linalg.norm(np.dot(grad, P_active)))

            if norm_active > 1e-12:
                self._alpha = norm_ort / norm_active

            self._alpha = float(np.clip(self._alpha, 0.0, 1.0))

        return grad
