"""ASGF - Adaptive Stochastic Gradient-Free optimization.

Implements Algorithm "ASGF" from ``thesis/chapters/thealgorithm.tex``:

* The gradient is estimated with Directional Gaussian Smoothing along an orthonormal
  basis of d directions using m Gauss-Hermite points per direction.
* After each iteration the first direction is pinned to the previously estimated
  (normalized) gradient and the remaining directions are completed at random, so the
  search keeps exploiting the most recent descent information.
* Local Lipschitz constants L_j are tracked per direction; the global constant follows
  the momentum recursion L_nabla <- (1-gamma_L) L_G + gamma_L L_nabla where L_G is the
  maximum local constant among historical directions (direction 0 once pinning is active,
  the maximum over all directions for the very first estimate).
* Step size lambda = sigma / L_nabla; sigma adapts via the threshold subroutine with a
  finite number of resets.

Deviations of the frozen prototype (documented, reproducible via parity studies):

* it quadratures with offsets ``sigma p_k`` and prefactor ``2/(sigma sqrt(pi))`` instead
  of the thesis' ``sqrt(2) sigma p_k`` and ``sqrt(2)/(sigma sqrt(pi))`` -- reproduced by
  setting ``legacy_rule=True``;
* it estimates L_j only over adjacent quadrature pairs instead of the full admissible
  pair set I;
* after assigning the pinned direction it re-orthonormalizes ALL rows with an SVD-based
  ``orth()``, which mixes the rows and loses exact pinning; we pin exactly via
  Gram-Schmidt completion instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..quadrature import GHQuadrature, dgs_gradient_estimate, get_quadrature_rule
from ._adaptive import AdaptiveSmoothingState
from ._common import (
    BaseRunConfig,
    RunResult,
    build_context,
    drive_run,
    gram_schmidt_complete,
    random_orthonormal_basis,
)


@dataclass(frozen=True)
class ASGFConfig(BaseRunConfig):
    """Configuration for :class:`ASGF`.

    Attributes
    ----------
    m:
        Number of Gauss-Hermite points per direction (odd, >= 3).
    sigma_zero:
        Initial smoothing radius; when None it is derived as ||x_0|| / 10, matching the
        frozen prototype's convention.
    a, b:
        Initial lower/upper thresholds of the adaptive subroutine.
    a_minus, a_plus, b_minus, b_plus:
        Threshold change rates.
    gamma_l:
        Momentum decay rate of L_nabla.
    gamma_sigma:
        Multiplicative smoothing-radius decay/increase factor.
    resets:
        Number of allowed full resets of (sigma, A, B, directions).
    rho:
        Reset trigger factor: reset when sigma < rho * sigma_zero.
    legacy_rule:
        Reproduce the frozen prototype's quadrature rule (parity studies only).
    """

    m: int = 5
    sigma_zero: float | None = None
    a: float = 0.1
    b: float = 0.9
    a_minus: float = 0.95
    a_plus: float = 1.02
    b_minus: float = 0.98
    b_plus: float = 1.01
    gamma_l: float = 0.9
    gamma_sigma: float = 0.9
    resets: int = 2
    rho: float = 0.01
    legacy_rule: bool = False


def _derive_sigma_zero(x0: NDArray[np.float64]) -> float:
    """Frozen-prototype convention: sigma_0 = ||x_0|| / 10."""
    return max(float(np.linalg.norm(x0)) / 10.0, 1e-12)


class ASGF:
    """Adaptive Stochastic Gradient-Free optimizer.

    Cost per iteration: dim*(m-1) function evaluations (+1 for the new iterate), since
    the central Gauss-Hermite node reuses f(x_i).
    """

    kind = "Adaptive Stochastic Gradient-Free"

    def __init__(self, config: ASGFConfig) -> None:
        self.config = config

    def run(self) -> RunResult:
        cfg = self.config
        name, f, rng = build_context(cfg)
        sign = 1.0 if cfg.maximize else -1.0
        quad: GHQuadrature = get_quadrature_rule(cfg.m)

        x0 = (
            np.array(cfg.x_init, dtype=np.float64)
            if cfg.x_init is not None
            else rng.standard_normal(cfg.dim)
        )
        state = AdaptiveSmoothingState(
            sigma_zero=cfg.sigma_zero if cfg.sigma_zero is not None else _derive_sigma_zero(x0),
            a=cfg.a,
            b=cfg.b,
            a_minus=cfg.a_minus,
            a_plus=cfg.a_plus,
            b_minus=cfg.b_minus,
            b_plus=cfg.b_plus,
            gamma_l=cfg.gamma_l,
            gamma_sigma=cfg.gamma_sigma,
            resets=cfg.resets,
            rho=cfg.rho,
        )
        basis = random_orthonormal_basis(rng, cfg.dim)
        prev_grad: NDArray[np.float64] | None = None
        sigmas: list[float] = []
        l_nablas: list[float] = []

        def step(i: int, x: NDArray[np.float64], value: float) -> tuple[NDArray[np.float64], float]:
            nonlocal basis, prev_grad
            est = dgs_gradient_estimate(f, x, basis, state.sigma, quad, f_x=value,
                                        legacy_rule=cfg.legacy_rule)
            # L_G: all directions before pinning exists, then the pinned one.
            if prev_grad is None:
                l_g = float(np.max(est.lipschitz_constants))
            else:
                l_g = float(est.lipschitz_constants[0])
            state.update_lipschitz_momentum(l_g)
            lr = state.learning_rate

            x_new = x + sign * lr * est.gradient
            v_new = float(f(x_new))

            # Parameter update: rebuild directions and adapt sigma.
            reset = state.adapt(est.directional_derivatives, est.lipschitz_constants)
            g_norm = float(np.linalg.norm(est.gradient))
            if reset:
                basis = random_orthonormal_basis(rng, cfg.dim)
            elif g_norm > 1e-12:
                first = est.gradient / g_norm
                seeds = np.vstack([first, rng.standard_normal((cfg.dim - 1, cfg.dim))])
                basis, _ = gram_schmidt_complete(rng, seeds)
            prev_grad = est.gradient
            sigmas.append(state.sigma)
            l_nablas.append(state.l_nabla)
            return x_new, v_new

        result = drive_run(cfg, name, f, rng, step, start_x=x0)
        result.extras["sigma"] = np.asarray(sigmas)
        result.extras["l_nabla"] = np.asarray(l_nablas)
        return result
