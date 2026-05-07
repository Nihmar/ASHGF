"""Gradient estimators: Gaussian smoothing, Gauss-Hermite quadrature, and
Lipschitz-constant estimation.

All critical loops are vectorised via NumPy to minimise Python-level
overhead in high-dimensional settings.
"""

from __future__ import annotations

import os
_os = os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache for Gauss-Hermite quadrature nodes & weights (computed once per m)
# ---------------------------------------------------------------------------
_GH_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_gauss_hermite(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_nodes, p_w, weights) for m-point Gauss-Hermite quadrature.

    Results are cached so that the eigenvalue decomposition of the
    tridiagonal Jacobi matrix is performed only once per distinct ``m``.
    """
    if m not in _GH_CACHE:
        p_nodes, weights = np.polynomial.hermite.hermgauss(m)
        p_w = p_nodes * weights  # pre-multiplied: w_k * p_k
        _GH_CACHE[m] = (p_nodes, p_w, weights)
    return _GH_CACHE[m]


# ---------------------------------------------------------------------------
# Parallel / batched evaluation of the objective function
# ---------------------------------------------------------------------------

# Number of threads to use when ``n_jobs`` is not explicitly given.
# Set via the environment variable ``ASHGF_N_JOBS``, defaulting to 1.
_DEFAULT_N_JOBS: int = int(os.environ.get("ASHGF_N_JOBS", "1"))


def _get_n_jobs(n_jobs: int | None = None) -> int:
    """Return the effective number of parallel workers."""
    if n_jobs is not None:
        return max(1, n_jobs)
    return max(1, _DEFAULT_N_JOBS)


def _parallel_eval(
    f: Callable[[np.ndarray], float],
    points: list[np.ndarray],
    *,
    n_jobs: int | None = None,
) -> list[float]:
    """Evaluate ``f`` on each point in ``points``, optionally in parallel.

    Parameters
    ----------
    f : callable
        The objective function.
    points : list of np.ndarray
        List of input points at which to evaluate ``f``.
    n_jobs : int or None
        Number of threads.  If ``None``, the value of the environment
        variable ``ASHGF_N_JOBS`` is used (default 1).

    Returns
    -------
    list of float
        ``f(p)`` for each ``p`` in ``points``, in the same order.
    """
    n = len(points)
    nj = _get_n_jobs(n_jobs)

    if nj <= 1 or n < 4:
        return [f(p) for p in points]

    # Parallel path — use executor.map which preserves order and avoids
    # the overhead of as_completed + dict bookkeeping.
    with ThreadPoolExecutor(max_workers=nj) as executor:
        return list(executor.map(f, points))


__all__ = [
    "gaussian_smoothing",
    "gauss_hermite_derivative",
    "estimate_lipschitz_constants",
]


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------


def gaussian_smoothing(
    x: np.ndarray,
    f: Callable[[np.ndarray], float],
    sigma: float,
    directions: np.ndarray,
    *,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Central Gaussian Smoothing gradient estimator.

    g(x) ≈ (1/(2σ·M)) Σ_{i=1}^{M} [f(x+σ·d_i) - f(x-σ·d_i)] · d_i

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        Point at which to estimate the gradient.
    f : callable
        Objective function f: R^d → R.
    sigma : float
        Smoothing bandwidth.
    directions : np.ndarray, shape (M, d)
        Matrix of random directions (one per row).
    n_jobs : int or None
        Number of threads for parallel evaluation.

    Returns
    -------
    grad : np.ndarray, shape (d,)
        Estimated gradient.
    """
    M = len(directions)

    # Pre-scale directions
    sigma_dirs = sigma * directions  # (M, d)
    dim = len(x)

    # Build all perturbed points via broadcasting (no Python loop)
    # all_arr[0::2] = x + sigma_dirs, all_arr[1::2] = x - sigma_dirs
    all_arr = np.empty((2 * M, dim))
    all_arr[0::2] = x[None, :] + sigma_dirs
    all_arr[1::2] = x[None, :] - sigma_dirs
    points = list(all_arr)  # list of (dim,) views

    # Evaluate (parallel if n_jobs > 1)
    results = _parallel_eval(f, points, n_jobs=n_jobs)

    # ---- Vectorised gradient assembly ----
    # diff[j] = f(x + σ·d_j) - f(x - σ·d_j)
    results_arr = np.asarray(results)  # (2M,)
    diff = results_arr[0::2] - results_arr[1::2]  # (M,)
    grad = np.dot(diff, directions) / (2.0 * sigma * M)  # (d,) = (M,) @ (M, d)

    return grad


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


def gauss_hermite_derivative(
    x: np.ndarray,
    f: Callable[[np.ndarray], float],
    sigma: float,
    basis: np.ndarray,
    m: int,
    value_at_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate directional derivatives using Gauss-Hermite quadrature.

    For each basis direction b_i:
        D_i f(x) ≈ (2/(σ√π)) Σ_{k=1}^{m} w_k · p_k · f(x + σ·p_k·b_i)

    where (p_k, w_k) are the nodes and weights of the m-point
    Gauss-Hermite quadrature rule.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        Point at which to estimate.
    f : callable
        Objective function.
    sigma : float
        Smoothing bandwidth.
    basis : np.ndarray, shape (d, d)
        Orthonormal basis matrix (each row is a direction vector b_i).
    m : int
        Number of quadrature points (must be odd for central node at 0).
    value_at_x : float or None
        f(x). If None, it will be evaluated.

    Returns
    -------
    grad : np.ndarray, shape (d,)
        Estimated gradient (sum of directional derivatives times basis vectors).
    evals_matrix : np.ndarray, shape (d, m)
        Matrix of function evaluations: row i = evaluations along direction i.
    points : np.ndarray, shape (m,)
        Quadrature nodes (shared across all directions).
    derivatives : np.ndarray, shape (d,)
        Array of estimated directional derivatives D_i f(x).
    """
    dim = len(x)
    if value_at_x is None:
        value_at_x = f(x)

    p_nodes, p_w, _weights = _get_gauss_hermite(m)
    # p_w[mid] == 0 because p_nodes[mid] == 0
    sigma_p = sigma * p_nodes  # (m,)
    mid = m // 2

    # Pre-compute scale factor for the quadrature
    quad_scale = 2.0 / (sigma * np.sqrt(math.pi))

    # ---- Build all non-central perturbed points via broadcasting ----
    # Shape: (dim, m, dim) → select k != mid → (dim, m-1, dim) → flatten
    perturbed = x[None, None, :] + sigma_p[None, :, None] * basis[:, None, :]
    k_mask = np.ones(m, dtype=bool)
    k_mask[mid] = False
    flat_arr = perturbed[:, k_mask, :].reshape(-1, dim)
    flat_points = list(flat_arr)

    # ---- Evaluate all points (parallel if ASHGF_N_JOBS > 1) ----
    flat_results = _parallel_eval(f, flat_points)

    # ---- Build evaluation matrix: (d, m) ----
    # Fill central column with f(x); other entries from flat_results.
    evals_matrix = np.empty((dim, m))
    evals_matrix[:, mid] = value_at_x
    # Reshape flat results to (dim, m-1) and assign to non-mid columns
    flat_reshaped = np.asarray(flat_results).reshape(dim, m - 1)
    evals_matrix[:, k_mask] = flat_reshaped

    # ---- Compute directional derivatives (vectorised) ----
    # derivatives[i] = quad_scale · Σ_k p_w[k] · evals_matrix[i, k]
    # Since p_w[mid] == 0, value_at_x contributes nothing.
    derivatives = quad_scale * (evals_matrix @ p_w)  # (d,) = (d, m) @ (m,)

    # ---- Gradient reconstruction: ∇f = Σ_i D_i f · b_i = Bᵀ · D ----
    grad = basis.T @ derivatives  # (d,) = (d, d) @ (d,)

    # The ``points`` return is kept as the 1-D array of nodes for
    # compatibility with ``estimate_lipschitz_constants``.
    return grad, evals_matrix, p_nodes, derivatives


# ---------------------------------------------------------------------------
# Lipschitz-constant estimation
# ---------------------------------------------------------------------------


def estimate_lipschitz_constants(
    evaluations: np.ndarray,
    points: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Estimate directional Lipschitz constants from quadrature data.

    Implements the thesis formula (Eq. ``Lipschitz constants``)::

        L_j = max_{{i,k} in I} |F(x+σ·p_i·ξ_j) - F(x+σ·p_k·ξ_j)| / (σ·|p_i - p_k|)

    where the index set ``I`` excludes pairs that are symmetric about
    the central quadrature node::

        I = {{ {i,k} | |i - floor(m/2)| != |k - floor(m/2)| }}

    Parameters
    ----------
    evaluations : np.ndarray, shape (d, m)
        Matrix of function evaluations along each direction (row = direction,
        column = quadrature node).
    points : np.ndarray, shape (m,)
        Quadrature nodes (shared across all directions).
    sigma : float
        Smoothing bandwidth.

    Returns
    -------
    lipschitz : np.ndarray, shape (d,)
        Estimated Lipschitz constants per direction.
    """
    d, m = evaluations.shape
    mid = m // 2

    # ---- Build all unordered pairs (i, k) with i < k ----
    i_idx, k_idx = np.triu_indices(m, k=1)  # each shape (P,)

    # ---- Exclude symmetric pairs around the centre ----
    dist_i = np.abs(i_idx - mid)
    dist_k = np.abs(k_idx - mid)
    keep = dist_i != dist_k

    i_idx = i_idx[keep]
    k_idx = k_idx[keep]
    # P ~ m*(m-1)/2 - floor((m-1)/2)  valid pairs

    if len(i_idx) == 0:
        return np.zeros(d)

    # ---- Vectorised ratios across all directions and all valid pairs ----
    # diff_evals: shape (d, P)
    diff_evals = np.abs(evaluations[:, i_idx] - evaluations[:, k_idx])

    # diff_pts: shape (P,)
    diff_pts = sigma * np.abs(points[i_idx] - points[k_idx])

    # ratios: (d, P) / (P,) -> (d, P)
    ratios = diff_evals / diff_pts[None, :]

    # Max over valid pairs per direction -> (d,)
    lipschitz = np.max(ratios, axis=1)

    return lipschitz
