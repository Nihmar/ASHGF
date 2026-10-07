"""SGES - Stochastic Gradient Estimation with Subspace exploitation.

Implements Algorithm "SGES" from ``thesis/chapters/thealgorithm.tex``:

* A FIFO buffer G of capacity k stores past gradient estimates; their span L_G is the
  "gradient subspace".
* For i < T_w (warm-up) all d directions are drawn iid from N(0, I_d). Afterwards each
  direction is drawn from L_G with probability alpha (exploitation) or isotropically
  with probability 1-alpha (exploration), then normalized.
* The exploitation probability is adapted after every post-warm-up iteration by
  comparing the best observed values along historical vs isotropic directions:
  r_hat_G = mean over historical slots of min(F+, F-), r_hat_perp defined analogously.
  If r_hat_G does not exist or r_hat_G < r_hat_perp then alpha <- min{delta*alpha, kappa_1};
  otherwise alpha <- max{alpha/delta, kappa_2}.

The frozen prototype deviates in two ways (documented, reproducible via parity harness):
it re-seeds its RNG inside the estimator on every call and overwrites the sampled
directions with fresh random ones, disabling guidance entirely. Our implementation
follows the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._common import (
    BaseRunConfig,
    HistoryBuffer,
    RunResult,
    build_context,
    central_difference_gradient,
    drive_run,
    subspace_basis,
)


@dataclass(frozen=True)
class SGESConfig(BaseRunConfig):
    """Configuration for :class:`SGES`.

    Attributes
    ----------
    sigma:
        Smoothing radius s.
    lr:
        Fixed step size lambda.
    k:
        Capacity of the gradient-history buffer G.
    alpha:
        Initial exploitation probability.
    kappa_1, kappa_2:
        Upper/lower bounds on alpha (kappa_1 > kappa_2).
    delta:
        Multiplicative adaptation factor (> 1).
    t_warmup:
        Number of warm-up iterations before exploitation starts; should satisfy
        t_warmup >= k so the buffer is full when guidance begins.
    """

    sigma: float = 1e-4
    lr: float = 1e-4
    k: int = 50
    alpha: float = 0.5
    kappa_1: float = 0.9
    kappa_2: float = 0.1
    delta: float = 1.1
    t_warmup: int = 50


class SGES:
    """Stochastic Gradient Estimation with Subspace-exploitation (SGES).

    Cost per iteration: 2d function evaluations (+1 for the new iterate).
    """

    kind = "SGES"

    def __init__(self, config: SGESConfig) -> None:
        self.config = config

    def run(self) -> RunResult:
        cfg = self.config
        name, f, rng = build_context(cfg)
        sign = 1.0 if cfg.maximize else -1.0
        buffer = HistoryBuffer(cfg.k)
        state = {"alpha": cfg.alpha}
        alphas: list[float] = []

        def sample_directions(i: int) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
            """Return (unit directions, historical-slot mask)."""
            hist_mask = np.zeros(cfg.dim, dtype=bool)
            dirs = rng.standard_normal((cfg.dim, cfg.dim))
            g_matrix = buffer.as_matrix()
            if i >= cfg.t_warmup and g_matrix is not None and len(buffer) >= 2:
                basis_g = subspace_basis(g_matrix)
                hist_mask = rng.random(cfg.dim) < state["alpha"]
                n_hist = int(hist_mask.sum())
                if n_hist > 0 and basis_g.shape[0] > 0:
                    z = rng.standard_normal((basis_g.shape[0], n_hist))
                    idx = np.flatnonzero(hist_mask)
                    dirs[idx] = (basis_g.T @ z).T
            norms = np.linalg.norm(dirs, axis=1)
            norms[norms == 0.0] = 1.0
            return dirs / norms[:, None], hist_mask


        def step(i: int, x: NDArray[np.float64], value: float) -> tuple[NDArray[np.float64], float]:
            dirs, hist_mask = sample_directions(i)
            grad, evals = central_difference_gradient(f, x, dirs, cfg.sigma)
            buffer.append(grad)

            # Adaptive exploitation probability (post-warm-up only).
            if i >= cfg.t_warmup:
                mins = np.minimum(evals[0::2], evals[1::2])  # best value per slot
                n_hist = int(hist_mask.sum())
                if 0 < n_hist < cfg.dim:
                    r = float(mins[hist_mask].mean())
                    r_perp = float(mins[~hist_mask].mean())
                    if r < r_perp:
                        state["alpha"] = min(cfg.delta * state["alpha"], cfg.kappa_1)
                    else:
                        state["alpha"] = max(state["alpha"] / cfg.delta, cfg.kappa_2)
                elif n_hist == 0:
                    state["alpha"] = min(cfg.delta * state["alpha"], cfg.kappa_1)
                else:
                    state["alpha"] = max(state["alpha"] / cfg.delta, cfg.kappa_2)
                alphas.append(state["alpha"])

            x_new = x + sign * cfg.lr * grad
            return x_new, float(f(x_new))

        result = drive_run(cfg, name, f, rng, step)
        if alphas:
            result.extras["alpha"] = np.asarray(alphas)
        return result
