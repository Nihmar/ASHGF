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
* By default the active subspace is computed from a *reduced spectrum*: the recursion
  unrolls to Cov_t = sum_j (1-lambda) lambda^(t-j) g_j g_j^T, so its range lies in the
  span of past gradients and older weights decay geometrically; keeping only the last
  K gradients (K chosen so the dropped tail is below double precision, K = 30 at the
  canonical lambda = 0.1) makes each spectral step O(d K^2) instead of O(d^3).
  Perpendicular directions are sampled as P_perp z with ambient Gaussians z, which has
  exactly the N(0, P_perp) marginal that drawing in an explicit complement basis gives.

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
    full_spectrum:
        When True, compute the active subspace from a full dense ``eigh`` of the
        covariance accumulator at every post-warmup step (exact but O(d^3) per
        iteration; kept as the reference path for A/B studies). Default False uses
        the numerically equivalent reduced spectrum over the decaying gradient window.
    legacy_estimator:
        When True, omit the 1/n_t averaging factor of the gradient estimator, matching
        the frozen prototype (defaults follow the thesis).
    """

    sigma: float = 1e-4
    lr: float = 1e-4
    lambd: float = 0.1
    thresh: float = 1e-4
    l_full: int = 50
    full_spectrum: bool = False
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


def _default_window_capacity(lambd: float) -> int:
    """Number of most recent gradients retained by the reduced-spectrum scheme.

    The dropped tail carries eigen-mass of order ||g||^2 * lambda^K / (1 - lambda);
    K is chosen so that this falls below double-precision relevance (~1e-13 relative).
    For the canonical lambda = 0.1 this yields 30.
    """
    if not 0.0 < lambd < 1.0:
        raise ValueError(f"lambd must lie in (0, 1), got {lambd}")
    k = int(np.ceil(np.log(1e-13) / np.log(lambd))) + 10
    return max(k, 30)


class DecayedGradientWindow:
    """Sliding window over recent gradients backing the reduced-spectrum scheme.

    Holds the last ``capacity`` gradient vectors (oldest first). Its restricted
    accumulator C~_t = sum_j (1-lambda) lambda^(t-j) g_j g_j^T over the window equals
    the true Cov_t up to the exponentially tiny dropped tail, and its non-zero
    spectrum lives entirely in the row space of the window matrix.
    """

    def __init__(self, lambd: float, capacity: int) -> None:
        self._lambd = float(lambd)
        self._capacity = int(capacity)
        self._buf: list[NDArray[np.float64]] = []

    @property
    def size(self) -> int:
        return len(self._buf)

    def push(self, grad: NDArray[np.float64]) -> None:
        self._buf.append(np.asarray(grad, dtype=np.float64))
        if len(self._buf) > self._capacity:
            del self._buf[: len(self._buf) - self._capacity]

    def spectrum(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Eigenpairs of the window-restricted accumulator, descending order.

        Returns ``(eigvals, eigvecs)`` where columns of ``eigvecs`` are the
        corresponding d-vectors, or ``None`` when the window is empty.
        """
        if not self._buf:
            return None
        G = np.stack(self._buf, axis=0)          # (n, d)
        Q, _ = np.linalg.qr(G.T)                # (d, k), cols spanning row space
        V = G @ Q                                # (n, k), row j = coeffs of g_j in Q
        ages = np.arange(len(self._buf) - 1, -1, -1, dtype=np.float64)
        weights = (1.0 - self._lambd) * self._lambd ** ages
        WV = V * weights[:, None]               # (n, k), weight each gradient row
        M = WV.T @ V                             # (k, k)
        vals, vecs = np.linalg.eigh(M)
        order = np.argsort(vals)[::-1]
        return vals[order], Q @ vecs[:, order]


def _split_from_spectrum(
    eigvals: NDArray[np.float64],
    eigvecs: NDArray[np.float64],
    thresh: float,
    dim: int,
) -> tuple[NDArray[np.float64], int] | None:
    """Active rank r and basis rows u_act from a descending spectrum.

    Same explained-variance rule as :func:`_active_subspace`; returns
    ``(u_act_rows, r)`` or ``None`` when the spectrum carries no variance.
    """
    total = float(eigvals.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    lam_bar = eigvals / total
    r = int(np.searchsorted(np.cumsum(lam_bar), thresh) + 1)
    r = min(max(r, 1), dim)
    return eigvecs[:, :r].T, r


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
        window = DecayedGradientWindow(cfg.lambd, _default_window_capacity(cfg.lambd))
        state = {"p": 0.5}
        ps: list[float] = []
        ranks: list[int] = []

        def sample_dirs(u_act: NDArray[np.float64], r: int) -> tuple[NDArray[np.float64], int]:
            """Draw the per-slot direction matrix for one post-warmup iteration."""
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
                # P_perp z via ambient projection: exactly N(0, P_perp) marginals,
                # identical in law to drawing coefficients in an explicit complement
                # basis, without ever completing that basis.
                z = rng.standard_normal((n_b, d))
                proj = z - (z @ u_act.T) @ u_act
                dirs[n_a:] = proj * np.sqrt(d / (d - r))
            ranks.append(r)
            return dirs, n_t

        def estimate(i: int, x: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal cov
            use_subspace = i >= cfg.l_full
            if not use_subspace or cfg.full_spectrum:
                # Exact dense path (also the only pre-warm-up behaviour).
                split = _active_subspace(cov, cfg.thresh) if use_subspace else None
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
            else:
                spec = window.spectrum()
                split = _split_from_spectrum(*spec, cfg.thresh, d) if spec is not None else None
                if split is None:
                    dirs = rng.standard_normal((d, d))
                    n_t = d
                else:
                    dirs, n_t = sample_dirs(split[0], split[1])

            plus = f(x[None, :] + cfg.sigma * dirs)
            minus = f(x[None, :] - cfg.sigma * dirs)
            denom = 2.0 * cfg.sigma * n_t if not cfg.legacy_estimator else 2.0 * cfg.sigma
            grad = (plus - minus) @ dirs / denom
            cov = cfg.lambd * cov + (1.0 - cfg.lambd) * np.outer(grad, grad)
            window.push(grad)
            if use_subspace:
                # Update p for the next iteration using projected gradient energies.
                if cfg.full_spectrum:
                    split_now = _active_subspace(cov, cfg.thresh)
                    if split_now is not None:
                        u_act, u_perp, r = split_now
                        g_act = u_act @ grad
                        g_perp = u_perp @ grad
                        state["p"] = self._optimal_p(
                            float(g_act @ g_act), float(g_perp @ g_perp), r, d
                        )
                else:
                    spec_now = window.spectrum()
                    split_now = (
                        _split_from_spectrum(*spec_now, cfg.thresh, d)
                        if spec_now is not None
                        else None
                    )
                    if split_now is not None:
                        u_act, r = split_now
                        g_act = u_act @ grad
                        s_act = float(g_act @ g_act)
                        s_perp = float(grad @ grad) - s_act
                        state["p"] = self._optimal_p(s_act, s_perp, r, d)
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
