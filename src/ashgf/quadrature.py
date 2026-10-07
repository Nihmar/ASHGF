"""Gauss-Hermite quadrature and Directional Gaussian Smoothing (DGS) estimators.

Implements the mathematical core of ASGF/ASHGF as specified by the thesis
(``thesis/chapters/stateoftheartalgorithms.tex`` and ``chapters/thealgorithm.tex``):

Directional Gaussian Smoothing derivative estimator
----------------------------------------------------
For a direction :math:`\\xi` (unit norm), the smoothed directional derivative is

.. math::
    \\mathcal{D}[G_\\sigma(0 \\mid x, \\xi)]
        = \\frac{1}{\\sigma}\\,\\mathbb{E}_{v \\sim \\mathcal{N}(0,1)}[v\\,F(x + \\sigma v \\xi)],

which is approximated with the M-point Gauss-Hermite rule
(equation "GHApproximationSmothedSection" in the thesis):

.. math::
    \\widetilde{\\mathcal{D}}^{M}[G_\\sigma(0 \\mid x, \\xi)]
        = \\frac{\\sqrt{2}}{\\sigma\\sqrt{\\pi}}
          \\sum_{k=1}^{M} w_k\\, p_k\\, F\\bigl(x + \\sqrt{2}\\,\\sigma\\, p_k\\, \\xi\\bigr),

where :math:`(p_k, w_k)_{k=1}^{M}` are the standard Gauss-Hermite nodes and weights, i.e.
:math:`\\int_{\\mathbb{R}} e^{-u^2} f(u) du \\approx \\sum_k w_k f(p_k)`. When :math:`M` is odd,
the central node :math:`p_c = 0` coincides with :math:`x`, so its evaluation reuses
:math:`f(x)` and no extra function evaluation is spent there.

Local Lipschitz constants
-------------------------
Per equation "Lipschitz constants" in the thesis, each direction :math:`j` gets

.. math::
    L_j = \\max_{\\{i,k\\} \\in I}
          \\left|
          \\frac{F(x + s\\, p_i\\, \\xi_j) - F(x + s\\, p_k\\, \\xi_j)}
               {s\\,(p_i - p_k)}
          \\right|,

with :math:`s = \\sqrt{2}\\,\\sigma` the actual offset scale of the evaluations above and
:math:`I = \\{\\{i,k\\} : |i - c| \\neq |k - c|\\}` the set of admissible index pairs (:math:`c`
is the center-node index). The denominator is therefore the *true distance* between the two
evaluation points along :math:`\\xi_j`; note that the thesis writes this denominator as
:math:`\\sigma(p_i-p_k)` while its own evaluation offsets carry a factor :math:`\\sqrt{2}`, which we
interpret here as a notational inconsistency (see ``comparisons/`` notes). Pairs symmetric about
the center (and the diagonal) are excluded, matching the intent of the frozen prototype.

Legacy rule
-----------
The frozen prototype instead evaluates at offsets :math:`\\sigma p_k` with prefactor
:math:`2/(\\sigma\\sqrt{\\pi})`. Both rules are exact for linear :math:`F` and consistent to
first order, but they quadrature different integrals. Set ``legacy_rule=True`` to reproduce
the prototype behavior; it exists solely for parity/comparison studies.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GHQuadrature:
    """An M-point Gauss-Hermite rule for :math:`\\int e^{-u^2} f(u) du`.

    Attributes
    ----------
    m:
        Number of nodes (odd).
    nodes:
        Ascending nodes :math:`p_k`, symmetric about zero.
    weights:
        Positive weights :math:`w_k`.
    center_index:
        Index of the central node (exactly zero when :math:`m` is odd).
    pairs:
        Admissible index pairs ``(i, k)`` with ``i < k`` for the local Lipschitz estimate,
        i.e. those whose distances from the center differ.
    """

    m: int
    nodes: NDArray[np.float64]
    weights: NDArray[np.float64]
    center_index: int
    pairs: tuple[tuple[int, int], ...]

    @classmethod
    def of_order(cls, m: int) -> GHQuadrature:
        if not isinstance(m, int) or m < 3 or m % 2 == 0:
            raise ValueError(f"Gauss-Hermite order must be an odd integer >= 3, got {m}")
        nodes, weights = np.polynomial.hermite.hermgauss(m)
        center = m // 2
        pairs = tuple(
            (i, k)
            for i in range(m)
            for k in range(i + 1, m)
            if abs(i - center) != abs(k - center)
        )
        return cls(m=m, nodes=nodes, weights=weights, center_index=center, pairs=pairs)

    @property
    def has_center_node(self) -> bool:
        """True when a node sits exactly at zero (always true for odd orders)."""
        return abs(self.nodes[self.center_index]) < 1e-14


@dataclass(frozen=True)
class DGSEstimate:
    """Result of one DGS gradient estimation step."""

    gradient: NDArray[np.float64]                 # full gradient estimate, shape (d,)
    directional_derivatives: NDArray[np.float64]  # per-direction estimates D̃_j, shape (d,)
    lipschitz_constants: NDArray[np.float64]      # local Lipschitz constants L_j, shape (d,)
    directional_values: NDArray[np.float64]       # raw evaluations V[k, j] = F(x + s p_k ξ_j)
    n_function_evaluations: int                   # objective calls made by this call


def get_quadrature_rule(m: int) -> GHQuadrature:
    """Convenience wrapper around :meth:`GHQuadrature.of_order` for algorithm configs."""
    return GHQuadrature.of_order(m)


def dgs_gradient_estimate(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    basis: NDArray[np.float64],
    sigma: float,
    quad: GHQuadrature,
    f_x: float | None = None,
    chunk_size: int | None = 256,
    legacy_rule: bool = False,
) -> DGSEstimate:
    """Estimate the DGS gradient of ``f`` at ``x`` along the directions of ``basis``.

    Parameters
    ----------
    f:
        Batch-evaluable objective: input ``(n, dim)``, output ``(n,)``.
    x:
        Point where the gradient is estimated, shape ``(dim,)``.
    basis:
        Orthonormal direction matrix, shape ``(dim, dim)``; row ``j`` is direction ``ξ_j``.
    sigma:
        Smoothing parameter (> 0).
    quad:
        Gauss-Hermite rule to use.
    f_x:
        Optional precomputed value ``f(x)`` reused at the central node (saves one FE).
    chunk_size:
        Number of directions processed per batched function call (memory control);
        ``None`` processes all directions at once.
    legacy_rule:
        Reproduce the frozen prototype's quadrature rule (offsets ``σ p_k``, prefactor
        ``2/(σ√π)``) instead of the thesis formula. Parity studies only.

    Returns
    -------
    DGSEstimate
        Gradient, per-direction derivatives, local Lipschitz constants, the raw
        per-direction evaluation matrix and the number of objective evaluations
        performed by this call.
    """
    x = np.asarray(x, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    dim = x.shape[0]
    if basis.shape != (dim, dim):
        raise ValueError(f"basis must have shape ({dim}, {dim}), got {basis.shape}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    scale = math.sqrt(2.0) * sigma if not legacy_rule else sigma
    coeff = (math.sqrt(2.0) / (sigma * math.sqrt(math.pi))) if not legacy_rule else (
        2.0 / (sigma * math.sqrt(math.pi))
    )
    q = coeff * quad.weights * quad.nodes  # length-m coefficient vector

    m = quad.m
    center = quad.center_index
    derivs = np.empty(dim)
    lip = np.zeros(dim)
    values_all = np.empty((m, dim))
    n_fes = 0

    if quad.has_center_node:
        if f_x is None:
            f_x = float(np.asarray(f(x[None, :]), dtype=np.float64)[0])
            n_fes += 1
        center_value = f_x
        # Offsets excluding the central node (it coincides with x and reuses f_x).
        off_idx = [k for k in range(m) if k != center]
    else:
        center_value = None
        off_idx = list(range(m))

    start = 0
    while start < dim:
        stop = dim if chunk_size is None else min(start + chunk_size, dim)
        B = basis[start:stop]  # (c, d), rows are directions ξ_j

        # Evaluation grid: point (r, j) is x + offset_r * ξ_j, shape (L, c, d).
        offsets = scale * quad.nodes[off_idx]
        grid = x[None, None, :] + offsets[:, None, None] * B[None, :, :]
        flat = grid.reshape(-1, dim)
        values_off = np.asarray(f(flat), dtype=np.float64).reshape(len(off_idx), -1)

        # Scatter into the full (m, c) value matrix, filling the center row from f_x.
        values = np.empty((m, stop - start))
        values[off_idx, :] = values_off
        if center_value is not None:
            values[center, :] = center_value

        # Per-direction derivative estimates: D̃_j = Σ_k q_k V[k, j].
        derivs[start:stop] = values.T @ q
        values_all[:, start:stop] = values

        # Local Lipschitz constants over admissible pairs I.
        for a, b in quad.pairs:
            denom = scale * abs(quad.nodes[a] - quad.nodes[b])
            slope = np.abs(values[a, :] - values[b, :]) / denom
            lip[start:stop] = np.maximum(lip[start:stop], slope)

        n_fes += flat.shape[0]
        start = stop

    gradient = basis.T @ derivs
    return DGSEstimate(
        gradient=gradient,
        directional_derivatives=derivs,
        lipschitz_constants=lip,
        directional_values=values_all,
        n_function_evaluations=n_fes,
    )
