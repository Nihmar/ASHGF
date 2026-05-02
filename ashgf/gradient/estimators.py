from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    Notes
    -----
    Parallelism is only used when ``n_jobs > 1`` **and** there are at
    least 4 points to evaluate (to avoid thread-creation overhead for
    tiny batches).
    """
    n = len(points)
    nj = _get_n_jobs(n_jobs)

    if nj <= 1 or n < 4:
        # Sequential path — no overhead
        return [f(p) for p in points]

    # Parallel path — useful for expensive ``f`` (e.g., RL environments)
    results: list[float] = [0.0] * n
    with ThreadPoolExecutor(max_workers=nj) as executor:
        future_to_idx = {executor.submit(f, p): i for i, p in enumerate(points)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


__all__ = [
    "gaussian_smoothing",
    "gauss_hermite_derivative",
    "estimate_lipschitz_constants",
]


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
        Number of threads for parallel evaluation.  Default (None) uses
        the env var ``ASHGF_N_JOBS`` (default 1 = sequential).

    Returns
    -------
    grad : np.ndarray, shape (d,)
        Estimated gradient.
    """
    dim = len(x)
    M = len(directions)

    # Pre-scale directions
    sigma_dirs = sigma * directions  # (M, d)

    # Build flat list of all perturbed points
    points: list[np.ndarray] = []
    for i in range(M):
        d = sigma_dirs[i]
        points.append(x + d)
        points.append(x - d)

    # Evaluate (parallel if n_jobs > 1)
    results = _parallel_eval(f, points, n_jobs=n_jobs)

    # Assemble gradient
    grad = np.zeros(dim)
    for i in range(M):
        f_plus = results[2 * i]
        f_minus = results[2 * i + 1]
        grad += (f_plus - f_minus) * directions[i]

    grad /= 2 * sigma * M
    return grad


def gauss_hermite_derivative(
    x: np.ndarray,
    f: Callable[[np.ndarray], float],
    sigma: float,
    basis: np.ndarray,
    m: int,
    value_at_x: float | None = None,
) -> tuple[np.ndarray, dict, dict, np.ndarray]:
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
        Orthonormal basis matrix (each row is a direction).
    m : int
        Number of quadrature points (must be odd for central node at 0).
    value_at_x : float or None
        f(x). If None, it will be evaluated.

    Returns
    -------
    grad : np.ndarray, shape (d,)
        Estimated gradient (sum of directional derivatives times basis vectors).
    evaluations : dict
        Dict mapping direction index i → numpy array of m evaluations.
    points : dict
        Dict mapping direction index i → array of quadrature nodes.
    derivatives : np.ndarray, shape (d,)
        Array of estimated directional derivatives.
    """
    dim = len(x)
    if value_at_x is None:
        value_at_x = f(x)

    p_nodes, p_w, _weights = _get_gauss_hermite(m)
    # p_w[mid] == 0 because p_nodes[mid] == 0
    sigma_p = sigma * p_nodes  # (m,)
    mid = m // 2

    # Pre-allocate arrays to avoid per-direction list→array conversions
    evaluations: dict[int, np.ndarray] = {}
    points: dict[int, np.ndarray] = {}
    derivatives = np.empty(dim)

    # Pre-compute scale factor for the quadrature
    quad_scale = 2.0 / (sigma * np.sqrt(math.pi))

    # ---- Build flat list of all non-central perturbed points ----
    # Map each flat index to (direction_i, node_k)
    idx_map: list[tuple[int, int]] = []
    flat_points: list[np.ndarray] = []
    for i in range(dim):
        for k in range(m):
            if k != mid:
                flat_points.append(x + sigma_p[k] * basis[i])
                idx_map.append((i, k))

    # ---- Evaluate all points (parallel if ASHGF_N_JOBS > 1) ----
    flat_results = _parallel_eval(f, flat_points)

    # ---- Restructure results into per-direction arrays ----
    for i in range(dim):
        evals = np.empty(m)
        evals[mid] = value_at_x  # central node at 0
        evaluations[i] = evals
        points[i] = p_nodes

    for idx_flat, (i, k) in enumerate(idx_map):
        evaluations[i][k] = flat_results[idx_flat]

    # ---- Compute directional derivatives ----
    for i in range(dim):
        # D_i f(x) = quad_scale · Σ p_w[k] · evals[k]
        # Since p_w[mid] == 0, value_at_x contribution is zero.
        derivatives[i] = quad_scale * np.dot(p_w, evaluations[i])

    # Vectorised gradient reconstruction:  ∇f = Σ_i D_i f · b_i  =  Bᵀ · D
    grad = basis.T @ derivatives  # (d,) = (d, d) @ (d,)

    return grad, evaluations, points, derivatives


def estimate_lipschitz_constants(
    evaluations: dict[int, list[float] | np.ndarray],
    points: dict[int, np.ndarray],
    sigma: float,
) -> np.ndarray:
    """
    Estimate directional Lipschitz constants from quadrature data.

    For each direction i, compute
    max |f(x+σ·p_{k+1}·b_i) - f(x+σ·p_k·b_i)| / (σ·|p_{k+1} - p_k|)

    Parameters
    ----------
    evaluations : dict
        Direction index → list or array of m function evaluations.
    points : dict
        Direction index → array of m quadrature nodes.
    sigma : float
        Smoothing bandwidth.

    Returns
    -------
    lipschitz : np.ndarray
        Estimated Lipschitz constants per direction.
    """
    dim = len(evaluations)

    # All directions share the same quadrature nodes
    pts = points[0]

    # Build matrix of evaluations: shape (dim, m)
    evals_matrix = np.array([evaluations[i] for i in range(dim)])

    # |f(x+σ·p_{k+1}·b_i) - f(x+σ·p_k·b_i)|  → shape (dim, m-1)
    diff_evals = np.abs(np.diff(evals_matrix, axis=1))

    # σ·|p_{k+1} - p_k|  → shape (m-1,)
    diff_pts = sigma * np.abs(np.diff(pts))

    # Ratios: shape (dim, m-1) / (m-1,) → (dim, m-1)
    ratios = diff_evals / diff_pts[None, :]

    # Max per direction → (dim,)
    lipschitz = np.max(ratios, axis=1)

    return lipschitz
