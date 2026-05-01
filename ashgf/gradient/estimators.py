from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

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

    Returns
    -------
    grad : np.ndarray, shape (d,)
        Estimated gradient.
    """
    dim = len(x)
    M = len(directions)
    grad = np.zeros(dim)

    for i in range(M):
        d = directions[i].reshape(x.shape)
        f_plus = f(x + sigma * d)
        f_minus = f(x - sigma * d)
        grad += (f_plus - f_minus) * d.reshape(grad.shape)

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
        Dict mapping direction index i → list of m evaluations.
    points : dict
        Dict mapping direction index i → array of quadrature nodes.
    derivatives : np.ndarray, shape (d,)
        Array of estimated directional derivatives.
    """
    dim = len(x)
    if value_at_x is None:
        value_at_x = f(x)

    p_nodes, weights = np.polynomial.hermite.hermgauss(m)
    p_w = p_nodes * weights
    sigma_p = sigma * p_nodes

    evaluations: dict[int, list[float]] = {}
    points: dict[int, np.ndarray] = {}
    derivatives = np.zeros(dim)

    mid = m // 2

    for i in range(dim):
        temp: list[float] = []
        for k in range(m):
            if k == mid:
                temp.append(value_at_x)
            else:
                temp.append(f(x + sigma_p[k] * basis[i]))

        # Gauss-Hermite quadrature for derivative
        # D_i f(x) ≈ (2/(σ√π)) Σ w_k · p_k · f(x + σ·p_k·b_i)
        derivative = (2.0 / (sigma * np.sqrt(math.pi))) * np.sum(p_w * np.array(temp))

        points[i] = p_nodes
        evaluations[i] = temp
        derivatives[i] = derivative

    # Reconstruct gradient: ∇f = Σ D_i f · b_i
    grad = np.zeros(x.shape)
    for i in range(len(x)):
        grad += derivatives[i] * basis[i]

    return grad, evaluations, points, derivatives


def estimate_lipschitz_constants(
    evaluations: dict[int, list[float]],
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
        Direction index → list of m function evaluations.
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
    lipschitz = np.ones(dim)

    for i in range(dim):
        max_val = 0.0
        evals = evaluations[i]
        pts = points[i]
        for k in range(len(pts) - 1):
            val = abs(evals[k + 1] - evals[k]) / (sigma * abs(pts[k + 1] - pts[k]))
            if val > max_val:
                max_val = val
        lipschitz[i] = max_val

    return lipschitz
