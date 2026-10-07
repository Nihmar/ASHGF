"""ASHGF - Adaptive Stochastic Historical Gradient-Free optimization.

Implements Algorithm "ASHGF" + "Parameter update" from
``thesis/chapters/thealgorithm.tex``:

* A FIFO buffer G of capacity k archives past gradient estimates; their span L_G is the
  historical subspace. For i < T_w all directions form a fresh random orthonormal basis;
  afterwards each iteration builds d candidate directions, drawing M ~ Binomial(d, alpha)
  of them from N(0, Cov(G)) and completing the rest isotropically, then
  orthonormalizes via Gram-Schmidt completion (the first M surviving directions stay in
  L_G, giving exact historical attribution).
* The DGS estimator uses m Gauss-Hermite points per direction (same m for historical and
  random directions, as prescribed by the thesis).
* L_nabla follows the momentum recursion with L_G = max over all local constants during
  warm-up and max over the M historical directions afterwards.
* lambda = sigma / L_nabla; sigma adapts through the threshold subroutine with r resets.
* The exploitation probability alpha is adapted after every post-warm-up iteration by
  comparing, per direction slot, the best observed quadrature value min_k F(x + s p_k xi_j):
  r_hat_G vs r_hat_perp, with the same rule as SGES.

Deviations of the frozen prototype (documented; parity studies in ``comparisons/``):

* legacy quadrature rule (offsets ``sigma p_k``, prefactor ``2/(sigma sqrt(pi))``);
  reproduced via ``legacy_rule=True``;
* its pair filter for L_j is broken by operator precedence but turns out numerically
  equivalent to the thesis' admissible set I (verified by test); we use I directly;
* during warm-up it sets M = d//2 and takes L_G over only half the directions, while the
  thesis says L_G is the maximum over ALL local constants until T_w -- we follow the
  thesis;
* its SVD-based ``orth()`` mixes rows after sampling, losing historical attribution;
  our Gram-Schmidt construction keeps the first M directions inside L_G exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..quadrature import GHQuadrature, dgs_gradient_estimate, get_quadrature_rule
from ._adaptive import AdaptiveSmoothingState
from ._common import (
    BaseRunConfig,
    HistoryBuffer,
    RunResult,
    build_context,
    drive_run,
    gram_schmidt_complete,
    random_orthonormal_basis,
    subspace_basis,
)
from .asgf import _derive_sigma_zero


@dataclass(frozen=True)
class ASHGFConfig(BaseRunConfig):
    """Configuration for :class:`ASHGF`.

    Attributes
    ----------
    m:
        Number of Gauss-Hermite points per direction (odd, >= 3).
    sigma_zero:
        Initial smoothing radius; None derives ||x_0|| / 10 (prototype convention).
    a, b, a_minus, a_plus, b_minus, b_plus, gamma_l, gamma_sigma, resets, rho:
        Parameters of the adaptive-smoothing subroutine (see ASGFConfig).
    k:
        Capacity of the gradient archive G.
    t_warmup:
        Warm-up iterations before historical sampling starts (thesis: T_w = k).
    alpha:
        Initial exploitation probability.
    kappa_1, kappa_2, delta:
        Adaptation bounds/factor for alpha (kappa_1 > kappa_2, delta > 1).
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
    resets: int = 10
    rho: float = 0.01
    k: int = 50
    t_warmup: int = 50
    alpha: float = 0.5
    kappa_1: float = 0.9
    kappa_2: float = 0.1
    delta: float = 1.1
    legacy_rule: bool = False


class ASHGF:
    """Adaptive Stochastic Historical Gradient-Free optimizer.

    Cost per iteration: dim*(m-1) function evaluations (+1 for the new iterate).
    """

    kind = "Adaptive Stochastic Historical Gradient-Free"

    def __init__(self, config: ASHGFConfig) -> None:
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
        buffer = HistoryBuffer(cfg.k)
        basis = random_orthonormal_basis(rng, cfg.dim)
        hist_count: dict[str, object] = {"current": None}  # M for current basis (None in warm-up)
        alpha_state = {"alpha": cfg.alpha}
        alphas: list[float] = []
        sigmas: list[float] = []

        def adapt_alpha(mins_per_slot: NDArray[np.float64]) -> None:
            n_hist = hist_count["current"] or 0
            if 0 < n_hist < cfg.dim:
                r_g = float(mins_per_slot[:n_hist].mean())
                r_perp = float(mins_per_slot[n_hist:].mean())
                if r_g < r_perp:
                    alpha_state["alpha"] = min(cfg.delta * alpha_state["alpha"], cfg.kappa_1)
                else:
                    alpha_state["alpha"] = max(alpha_state["alpha"] / cfg.delta, cfg.kappa_2)
            elif n_hist == 0:
                alpha_state["alpha"] = min(cfg.delta * alpha_state["alpha"], cfg.kappa_1)
            else:
                alpha_state["alpha"] = max(alpha_state["alpha"] / cfg.delta, cfg.kappa_2)
            alphas.append(alpha_state["alpha"])

        def step(i: int, x: NDArray[np.float64], value: float) -> tuple[NDArray[np.float64], float]:
            nonlocal basis
            est = dgs_gradient_estimate(f, x, basis, state.sigma, quad, f_x=value,
                                        legacy_rule=cfg.legacy_rule)

            # L_G selection: all directions in warm-up, historical ones afterwards.
            n_hist_now = hist_count["current"]
            if i < cfg.t_warmup or n_hist_now is None or n_hist_now == 0:
                l_g = float(np.max(est.lipschitz_constants))
            else:
                l_g = float(np.max(est.lipschitz_constants[:n_hist_now]))
            state.update_lipschitz_momentum(l_g)
            lr = state.learning_rate

            x_new = x + sign * lr * est.gradient
            v_new = float(f(x_new))

            buffer.append(est.gradient)

            # Exploitation-probability adaptation (post-warm-up only).
            if i >= cfg.t_warmup and hist_count["current"] is not None:
                adapt_alpha(est.directional_values.min(axis=0))

            # Parameter update: reset check, then rebuild next direction set.
            reset = state.adapt(est.directional_derivatives, est.lipschitz_constants)
            next_historical = (i + 1) >= cfg.t_warmup and not reset
            if reset or not next_historical:
                basis = random_orthonormal_basis(rng, cfg.dim)
                hist_count["current"] = None
            else:
                g_matrix = buffer.as_matrix()
                subspace = subspace_basis(g_matrix) if g_matrix is not None else None
                hist_mask = rng.random(cfg.dim) < alpha_state["alpha"]
                n_hist = int(hist_mask.sum())
                seeds = np.empty((cfg.dim, cfg.dim))
                idx = 0
                if n_hist > 0 and subspace is not None and subspace.shape[0] > 0:
                    z = rng.standard_normal((subspace.shape[0], n_hist))
                    seeds[idx : idx + n_hist] = (subspace.T @ z).T
                    idx += n_hist
                seeds[idx:] = rng.standard_normal((cfg.dim - idx, cfg.dim))
                new_basis, kept_mask = gram_schmidt_complete(rng, seeds)
                basis = new_basis
                hist_count["current"] = int(kept_mask[:n_hist].sum())
            sigmas.append(state.sigma)
            return x_new, v_new

        result = drive_run(cfg, name, f, rng, step, start_x=x0)
        result.extras["sigma"] = np.asarray(sigmas)
        if alphas:
            result.extras["alpha"] = np.asarray(alphas)
        return result

