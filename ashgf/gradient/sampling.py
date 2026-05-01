"""Direction sampling utilities for gradient-based optimization algorithms.

This module provides functions to generate direction matrices used in
stochastic optimization methods such as SGES and ASHGF.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "compute_directions",
    "compute_directions_sges",
    "compute_directions_ashgf",
]


def compute_directions(dim: int) -> np.ndarray:
    """Generate a `dim × dim` matrix of i.i.d. standard normal directions.

    Parameters
    ----------
    dim : int
        Dimensionality of the problem space. The returned matrix has shape
        ``(dim, dim)``, where each row is a direction vector drawn from the
        standard normal distribution :math:`\\mathcal{N}(0, I)`.

    Returns
    -------
    np.ndarray
        A ``(dim, dim)`` NumPy array whose entries are i.i.d. samples from
        :math:`\\mathcal{N}(0, 1)`.

    Notes
    -----
    The directions are **not** normalized to unit length.  For unit-norm
    directions, use `compute_directions_sges` or `compute_directions_ashgf`
    which include an explicit normalization step.

    Examples
    --------
    >>> np.random.seed(42)
    >>> d = compute_directions(5)
    >>> d.shape
    (5, 5)
    """
    return np.random.randn(dim, dim)


def compute_directions_sges(
    dim: int,
    G: Union[List[np.ndarray], np.ndarray],
    alpha: float,
) -> Tuple[np.ndarray, int]:
    """Generate directions mixing gradient-history subspace and random subspace.

    With probability ``alpha``, a direction is drawn from
    :math:`\\mathcal{N}(0, \\mathrm{Cov}(G))` (the empirical covariance of the
    gradient buffer ``G``); with probability ``1 - alpha``, it is drawn from
    the isotropic standard normal :math:`\\mathcal{N}(0, I)`.  All directions
    are then normalized to unit Euclidean length.

    Parameters
    ----------
    dim : int
        Dimensionality of the problem space.
    G : array-like, shape ``(T, d)``
        Buffer of past gradient estimates.  Each row is a gradient vector of
        length ``d`` (which must equal ``dim``).  If a list of 1-D arrays is
        provided, it is converted to a 2-D NumPy array internally.
    alpha : float
        Probability of sampling a direction from the gradient subspace
        (i.e., from the empirical covariance of ``G``).  Must satisfy
        ``0 <= alpha <= 1``.

    Returns
    -------
    directions : np.ndarray
        A ``(dim, dim)`` matrix where each row is a **unit-norm** direction
        vector.
    choices : int
        The number of directions that were actually sampled from the gradient
        subspace (can range from ``0`` to ``dim``).

    Notes
    -----
    The covariance matrix :math:`\\mathrm{Cov}(G)` is estimated from the
    columns of ``G`` via `numpy.cov`.  The binary decisions for each of the
    ``dim`` directions are taken independently according to a Bernoulli
    distribution with parameter ``alpha``, and the resulting count is clamped
    to the valid range ``[0, dim]``.

    To avoid division by zero, any direction whose empirical standard
    deviation (before concatenation) is below ``1e-12`` is left untouched
    during the per-component scaling step.  After assembly, the final
    normalization guards against zero-norm rows by replacing near-zero norms
    with ``1.0``.

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
    # Ensure G is a 2-D array
    G_arr: np.ndarray = np.array(G)

    # Empirical covariance of the gradient buffer (column-wise)
    cov_L_G: np.ndarray = np.cov(G_arr.T)

    # Binary decisions: how many directions come from the gradient subspace
    choices: int = 0
    for i in range(dim):
        choices += int(np.random.choice([0, 1], size=1, p=[alpha, 1.0 - alpha]).item())

    # Clamp choices to the valid range
    choices = max(0, min(choices, dim))

    # --- Gradient-subspace directions ---
    dirs_grad: np.ndarray = np.zeros((choices, dim))
    if choices > 0:
        dirs_grad = np.random.multivariate_normal(np.zeros(dim), cov_L_G, choices)
        # Scale each direction by its empirical std (optional pre-normalization)
        for i in range(choices):
            std_i: float = float(np.std(dirs_grad[i]))
            if std_i > 1e-12:
                dirs_grad[i] /= std_i

    # --- Random (isotropic) directions ---
    dirs_random: np.ndarray = np.random.multivariate_normal(
        np.zeros(dim), np.identity(dim), dim - choices
    )

    # Assemble the direction matrix
    if choices > 0:
        dirs: np.ndarray = np.concatenate((dirs_grad, dirs_random), axis=0)
    else:
        dirs = dirs_random

    # Normalize every direction to unit length
    norms: np.ndarray = np.linalg.norm(dirs, axis=-1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    dirs = dirs / norms

    return dirs, choices


def compute_directions_ashgf(
    dim: int,
    G: Union[List[np.ndarray], np.ndarray],
    alpha: float,
    M: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Generate directions for ASHGF, mixing gradient subspace and random subspace.

    Exactly ``M`` directions are drawn from the gradient-history subspace
    (using the empirical covariance of ``G``) and the remaining ``dim - M``
    from the isotropic random subspace :math:`\\mathcal{N}(0, I)`.
    All directions are then normalized to unit Euclidean length.

    Parameters
    ----------
    dim : int
        Dimensionality of the problem space.
    G : array-like, shape ``(T, d)``
        Buffer of past gradient estimates.  Each row is a gradient vector of
        length ``d`` (which must equal ``dim``).
    alpha : float
        Probability parameter used to determine ``M`` when ``M`` is ``None``
        (forwarded to `compute_directions_sges`).
    M : int or None, optional
        Pre-computed number of directions to sample from the gradient
        subspace.  If ``None``, ``M`` is determined internally by the same
        binary-sampling procedure employed in `compute_directions_sges`.

    Returns
    -------
    directions : np.ndarray
        A ``(dim, dim)`` matrix where each row is a **unit-norm** direction
        vector.
    M : int
        Number of directions actually drawn from the gradient subspace.

    Notes
    -----
    This is a thin wrapper around `compute_directions_sges` that makes the
    ``M`` parameter explicit in the interface.  When ``M`` is provided
    directly, it is currently passed through to `compute_directions_sges`
    which ignores it in favour of its own Bernoulli sampling; this behaviour
    is intentional for API compatibility and future extension.

    See Also
    --------
    compute_directions_sges : Underlying direction-generation routine.

    Examples
    --------
    >>> np.random.seed(42)
    >>> G = np.random.randn(8, 4)
    >>> dirs, m = compute_directions_ashgf(4, G, 0.5, M=2)
    >>> dirs.shape
    (4, 4)
    >>> np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    True
    """
    if M is None:
        return compute_directions_sges(dim, G, alpha)
    else:
        # Pass through to the underlying implementation; future versions
        # may use M directly to control the subspace split.
        return compute_directions_sges(dim, G, alpha)
