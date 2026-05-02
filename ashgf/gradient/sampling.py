"""Direction sampling strategies for gradient estimation.

Provides functions for generating search directions used in
gradient-free optimization, including:

- Pure random (Gaussian isotropic) directions
- SGES-style adaptive mixing of gradient-history and random directions
- ASHGF-style directions (delegates to SGES)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_directions",
    "compute_directions_sges",
    "compute_directions_ashgf",
]


def compute_directions(dim: int) -> np.ndarray:
    """Generate a ``dim × dim`` matrix of i.i.d. standard normal directions.

    The rows are drawn from :math:`\\mathcal{N}(0, I)` and are **not**
    normalised to unit length (the Gaussian smoothing formula uses the
    raw Gaussian directions).

    Parameters
    ----------
    dim : int
        Dimensionality of the ambient space.

    Returns
    -------
    np.ndarray, shape (dim, dim)
        Matrix whose rows are i.i.d. :math:`\\mathcal{N}(0, 1)` vectors.
    """
    return np.random.randn(dim, dim)


def compute_directions_sges(
    dim: int,
    G: list[np.ndarray] | np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Generate directions mixing gradient-subspace and isotropic components.

    Parameters
    ----------
    dim : int
        Dimensionality of the ambient space.
    G : list of np.ndarray or np.ndarray, shape (T, d)
        Gradient-history buffer; each row is a past gradient estimate.
    alpha : float
        Probability of sampling a direction from the gradient subspace
        (isotropic otherwise).  Must be in [0, 1].

    Returns
    -------
    dirs : np.ndarray, shape (dim, dim)
        Orthonormal matrix whose rows are unit-norm directions.
    choices : int
        Number of directions actually sampled from the gradient subspace
        (may differ from ``alpha * dim`` due to the Bernoulli process).

    Raises
    ------
    ValueError
        If ``alpha`` is not in [0, 1].

    Notes
    -----
    The gradient-subspace covariance is Tikhonov-regularised
    (``1e-6 * I`` added) to prevent SVD non-convergence.  If the
    covariance contains non-finite values (e.g., due to overflow in
    the objective function), the gradient-subspace directions fall
    back to isotropic sampling.

    **Optimisation note (std normalisation removed).**
    Earlier versions applied an intermediate ``dirs_grad /= std(dirs_grad)``
    before the final unit-norm normalisation.  Because dividing by
    ``std(v)`` and then by ``‖v/σ‖`` is algebraically equivalent to
    dividing by ``‖v‖`` alone, the intermediate step had no effect on
    the result.  It has been removed to save a ``std`` + division pass.

    Examples
    --------
    >>> np.random.seed(42)
    >>> G = np.random.randn(10, 5)   # 10 past gradients in R^5
    >>> dirs, k = compute_directions_sges(5, G, 0.3)
    >>> dirs.shape
    (5, 5)
    >>> np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    True
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Ensure G is a 2-D array
    G_arr: np.ndarray = np.array(G)

    # Empirical covariance of the gradient buffer (column-wise)
    cov_L_G: np.ndarray = np.cov(G_arr.T)

    # Guard against non-finite covariance (caused by Inf/NaN in gradients)
    if not np.all(np.isfinite(cov_L_G)):
        # Fall back to isotropic directions
        dirs = np.random.randn(dim, dim)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return dirs / norms, 0

    # Add Tikhonov regularisation to prevent singular covariance
    cov_L_G += 1e-6 * np.eye(dim)

    # Vectorised binomial: how many directions come from the gradient subspace
    choices: int = int(np.random.binomial(dim, 1.0 - alpha))
    choices = max(0, min(choices, dim))

    # --- Gradient-subspace directions (Cholesky + affine transform) ---
    dirs_grad: np.ndarray = np.zeros((choices, dim))
    if choices > 0:
        try:
            # Cholesky factorisation: cov = L @ Lᵀ, then Z @ Lᵀ ~ N(0, cov)
            L = np.linalg.cholesky(cov_L_G)  # (dim, dim) lower-tri
            Z = np.random.standard_normal((choices, dim))
            dirs_grad = np.dot(Z, L.T)  # (choices, dim)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: if Cholesky still fails, use isotropic directions
            dirs_grad = np.random.standard_normal((choices, dim))

    # --- Random (isotropic) directions ---
    n_random = dim - choices
    dirs_random: np.ndarray = np.random.standard_normal((n_random, dim))

    # Assemble the direction matrix
    if choices > 0:
        dirs: np.ndarray = np.concatenate((dirs_grad, dirs_random), axis=0)
    else:
        dirs = dirs_random

    # Normalize every direction to unit length
    norms: np.ndarray = np.linalg.norm(dirs, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return dirs / norms, choices


def compute_directions_ashgf(
    dim: int,
    G: list[np.ndarray] | np.ndarray,
    alpha: float,
    M: int,
) -> tuple[np.ndarray, int]:
    """Generate directions for ASHGF (currently delegates to SGES).

    Parameters
    ----------
    dim : int
        Dimensionality of the ambient space.
    G : list of np.ndarray or np.ndarray
        Gradient-history buffer.
    alpha : float
        Probability of gradient-subspace sampling.
    M : int
        Unused; kept for backward compatibility.

    Returns
    -------
    dirs : np.ndarray, shape (dim, dim)
    choices : int
    """
    del M  # reserved for future per-dimension control
    return compute_directions_sges(dim, G, alpha)
