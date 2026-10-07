"""ASEBO - Adaptive Subspace Exploration BOptimization.

Implements Algorithm "ASEBO" from ``thesis/chapters/thealgorithm.tex``:

* For the first l iterations (warm-up) n_t = d directions are sampled iid from N(0, I_d).
* Afterwards the rank r of the "active" subspace is chosen as the smallest integer such
  that the top r eigenvalues of the covariance accumulator Cov explain at least a
  fraction ``thresh`` of the total variance. Directions are then sampled per-slot with
  probability p from N(0, P_act) and with probability 1-p from N(0, P_perp), where
  P_act / P_perp project onto the active subspace and its orthogonal complement. Each
  direction is rescaled so that E||g||^2 = d (the thesis' chi-square marginal condition).
* The gradient estimator averages over n_t = r central differences.
* Cov follows the thesis recursion Cov_{t+1} = lambda Cov_t + (1-lambda) g g^T.
* The exploration/exploitation weight p is updated with the closed-form rule of the
  ASEBO paper, p* ~ sqrt(s_act (r+2)) / (sqrt(s_act (r+2)) + sqrt(s_perp (d-r+2))),
  with s_act, s_perp the squared projected gradient energies in the two subspaces.

The frozen prototype implements a different mechanism (exponentially decayed history of
gradients + PCA on it, mixed-covariance sampling, and an estimator missing the 1/n_t
normalization factor). Setting ``legacy_estimator`` reproduces only the normalization
deviation; full behavioral parity against the prototype is studied in ``comparisons/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._common import BaseRunConfig, RunResult, build_context, drive_run


@dataclass(frozen=True)
class ASEBOConfig(BaseRunConfig):
    """Configuration for :class:`ASEBO`.

    Attributes
    ----------
    sigma:
        Smoothing radius s.
    lr:
        Fixed step size lambda.
    lambd:
        Decay rate of the covariance accumulator (thesis lambda, in (0, 1)).
    thresh:
        Explained-variance threshold epsilon used to pick the active rank r.
    l_full:
        Number of warm-up iterations using full isotropic sampling.
    legacy_estimator:
        When True, omit the 1/n_t averaging factor of the gradient estimator, matching
        the frozen prototype (defaults follow the thesis).
    """

    sigma: float = 1e-4
    lr: float = 1e-4
    lambd: float = 0.1
    thresh: float = 1e-4
    l_full: int = 50
    legacy_estimator: bool = False


def _active_subspace(
    cov: NDArray[np.float64], thresh: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], int] | None:
    """Split R^d into active subspace and complement based on explained variance.

    Returns ``(u_act, u_perp, r)`` with rows orthonormal, or ``None`` when the covariance
    carries no variance (caller falls back to isotropic sampling).
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    lam_bar = eigvals / total
    r = int(np.searchsorted(np.cumsum(lam_bar), thresh) + 1)
    r = min(max(r, 1), cov.shape[0])
    return eigvecs[:, :r].T, eigvecs[:, r:].T, r


class ASEBO:
    """Adaptive Subspace Exploration BOptimization (ASEBO).

    Cost per iteration: 2*n_t function evaluations (+1 for the new iterate), with
    n_t = dim during warm-up and n_t = r (active rank) afterwards.
    """

    kind = "ASEBO"

    def __init__(self, config: ASEBOConfig) -> None:
        self.config = config

    @staticmethod
    def _optimal_p(s_act: float, s_perp: float, r: int, dim: int) -> float:
        """Closed-form exploration/exploitation weight from the ASEBO paper."""
        num = np.sqrt(max(s_act, 0.0) * (r + 2))
        den = num + np.sqrt(max(s_perp, 0.0) * (dim - r + 2))
        if den <= 0.0:
            return 0.5
        return float(np.clip(num / den, 1e-6, 1.0 - 1e-6))

    def run(self) -> RunResult:
        cfg = self.config
        d = cfg.dim
        name, f, rng = build_context(cfg)
        sign = 1.0 if cfg.maximize else -1.0
        cov = np.zeros((d, d))
        state = {"p": 0.5}
        ps: list[float] = []
        ranks: list[int] = []

        def estimate(i: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal cov
            split = _active_subspace(cov, cfg.thresh) if i >= cfg.l_full else None
            if split is None:
                dirs = rng.standard_normal((d, d))
                n_t = d
            else:
                u_act, u_perp, r = split
                n_t = r
                if r >= d:
                    # Active subspace is all of R^d: nothing left to explore.
                    n_a, n_b = n_t, 0
                else:
                    choose_active = rng.random(n_t) < state["p"]
                    n_a = int(choose_active.sum())
                    n_b = n_t - n_a
                dirs = np.empty((n_t, d))
                if n_a > 0:
                    z = rng.standard_normal((n_a, r))
                    dirs[:n_a] = (z @ u_act) * np.sqrt(d / r)
                if n_b > 0:
                    w = rng.standard_normal((n_b, d - r))
                    dirs[n_a:] = (w @ u_perp) * np.sqrt(d / (d - r))
                ranks.append(r)
            plus = f(x[None, :] + cfg.sigma * dirs)
            minus = f(x[None, :] - cfg.sigma * dirs)
            denom = 2.0 * cfg.sigma * n_t if not cfg.legacy_estimator else 2.0 * cfg.sigma
            grad = (plus - minus) @ dirs / denom
            cov = cfg.lambd * cov + (1.0 - cfg.lambd) * np.outer(grad, grad)
            if i >= cfg.l_full:
                # Update p for the next iteration using projected gradient energies.
                split_now = _active_subspace(cov, cfg.thresh)
                if split_now is not None:
                    u_act, u_perp, r = split_now
                    g_act = u_act @ grad
                    g_perp = u_perp @ grad
                    state["p"] = self._optimal_p(float(g_act @ g_act), float(g_perp @ g_perp), r, d)
                ps.append(state["p"])
            return grad

        def step(i: int, x: NDArray[np.float64], value: float) -> tuple[NDArray[np.float64], float]:
            grad = estimate(i, x)
            x_new = x + sign * cfg.lr * grad
            return x_new, float(f(x_new))

        result = drive_run(cfg, name, f, rng, step)
        if ps:
            result.extras["p"] = np.asarray(ps)
        if ranks:
            result.extras["active_rank"] = np.asarray(ranks, dtype=np.float64)
        return result
