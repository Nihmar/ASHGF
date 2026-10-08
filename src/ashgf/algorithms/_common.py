"""Shared infrastructure for all optimization algorithms.

Conventions
-----------
- One ``numpy.random.Generator`` per run, derived from the configuration seed; no global
  seeding anywhere.
- Every objective call goes through :class:`CountingFunction`, so function-evaluation
  (FE) accounting is exact.
- Algorithms minimize by default; set ``maximize=True`` for maximization problems
  (the frozen prototype used this for its RL environments, flipping the update sign).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..functions import CountingFunction, get_function

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRunConfig:
    """Configuration shared by every algorithm.

    Attributes
    ----------
    function:
        Benchmark registry name (or any batch-evaluable callable ``(n, d) -> (n,)``).
    dim:
        Problem dimension.
    maxiter:
        Maximum number of iterations.
    seed:
        Seed for the single run-level RNG.
    eps:
        Termination tolerance on ``||x_{i+1} - x_i||_2``.
    rng:
        Optional explicit random source (any object exposing ``standard_normal``
        and ``random``, e.g. ``np.random.Generator`` or ``np.random.RandomState``);
        takes precedence over ``seed``. Used for legacy-stream parity studies.
    x_init:
        Optional starting point; drawn from N(0, I_d) when omitted.
    maximize:
        When True the algorithm performs ascent (sign-flipped update) and tracks the
        maximum instead of the minimum.
    """

    function: str | Callable[[NDArray[np.float64]], NDArray[np.float64]]
    dim: int
    maxiter: int
    seed: int
    eps: float = 1e-8
    rng: np.random.Generator | np.random.RandomState | None = field(
        default=None, compare=False, repr=False
    )
    x_init: NDArray[np.float64] | None = None
    maximize: bool = False


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Structured record of an optimization run.

    Attributes
    ----------
    function:
        Name of the benchmark function that was optimized.
    dim:
        Problem dimension.
    iterates:
        Array of shape ``(n_iters + 1, dim)``; row ``i`` is ``x_i``.
    values:
        Array of shape ``(n_iters + 1,)``; entry ``i`` is ``F(x_i)``.
    fes_per_iterate:
        Cumulative function-evaluation count aligned with ``values``; entry ``i`` is
        the number of objective evaluations performed up to and including ``F(x_i)``.
        Enables More & Sugrue performance/data profiles directly from the trajectory.
    best_index:
        Index of the iterate with the best-so-far value (min or max per config).
    total_function_evaluations:
        Exact count of scalar objective evaluations performed during the run.
    wall_time_seconds:
        Measured runtime of the run loop.
    extras:
        Per-algorithm trajectories (sigma, alpha, L_nabla, ...), stored by name.
    """

    function: str
    dim: int
    iterates: NDArray[np.float64]
    values: NDArray[np.float64]
    fes_per_iterate: NDArray[np.int64]
    best_index: int
    total_function_evaluations: int
    wall_time_seconds: float
    extras: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def n_iterations(self) -> int:
        return len(self.values) - 1

    @property
    def best_value(self) -> float:
        return float(self.values[self.best_index])

    @property
    def best_iterate(self) -> NDArray[np.float64]:
        return self.iterates[self.best_index]


def make_objective(
    function: str | Callable[[NDArray[np.float64]], NDArray[np.float64]]
) -> tuple[str, CountingFunction]:
    """Resolve the configured objective into (name, counting wrapper)."""
    if isinstance(function, str):
        fn = get_function(function)
        return fn.name, CountingFunction(fn.evaluate_batched)
    return "custom", CountingFunction(function)


def random_orthonormal_basis(rng: np.random.Generator, dim: int) -> NDArray[np.float64]:
    """A Haar-uniform orthonormal basis of R^d as a matrix whose rows are unit vectors."""
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return Q


def central_difference_gradient(
    f: CountingFunction,
    x: NDArray[np.float64],
    directions: NDArray[np.float64],
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Monte-Carlo central-difference gradient estimate along given directions.

    Parameters
    ----------
    f:
        Counting, batch-evaluable objective.
    x:
        Point of evaluation, shape ``(d,)``.
    directions:
        Matrix of shape ``(M, d)``; row ``j`` is direction ``xi_j``.
    sigma:
        Smoothing radius.

    Returns
    -------
    (gradient, evaluations)
        Gradient estimate ``sum_j (F(x+s xi_j) - F(x-s xi_j)) xi_j / (2 s M)`` and the
        raw evaluation array of shape ``(2*M,)`` in interleaved ``(plus, minus)`` order.
    """
    plus = f(x[None, :] + sigma * directions)
    minus = f(x[None, :] - sigma * directions)
    diff = plus - minus
    grad = diff @ directions / (2.0 * sigma * directions.shape[0])
    evaluations = np.empty(2 * directions.shape[0])
    evaluations[0::2] = plus
    evaluations[1::2] = minus
    return grad, evaluations


class HistoryBuffer:
    """Fixed-capacity FIFO buffer of past gradient estimates (the archive G)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: deque[NDArray[np.float64]] = deque()

    def append(self, g: NDArray[np.float64]) -> None:
        self._buf.append(g.copy())
        if len(self._buf) > self.capacity:
            self._buf.popleft()

    def as_matrix(self) -> NDArray[np.float64] | None:
        """The buffered gradients stacked as a ``(len, d)`` matrix, or None if empty."""
        if not self._buf:
            return None
        return np.stack(list(self._buf), axis=0)

    def __len__(self) -> int:
        return len(self._buf)


def subspace_basis(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal basis (rows) of the span of the rows of ``matrix``, via SVD.

    A rank-revealing tolerance based on the largest singular value is used.
    """
    _, s, vh = np.linalg.svd(matrix, full_matrices=False)
    tol = s[0] * max(matrix.shape) * np.finfo(np.float64).eps if s.size else 0.0
    rank = int(np.sum(s > tol)) if s.size else 0
    return vh[:rank]  # rows are orthonormal and span the rows of ``matrix``


def gram_schmidt_complete(
    rng: np.random.Generator,
    seeds: NDArray[np.float64],
    block_size: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Orthonormalize seed vectors (rows), keeping order, then complete at random.

    Degenerate seeds (numerically dependent on previously kept ones) are dropped.
    The remaining slots are filled with Gaussian vectors orthogonalized against the
    already-kept basis, so the result always spans R^d exactly. Kept seeds occupy the
    leading rows of the returned basis, preserving their relative order.

    Implementation: blocked modified Gram-Schmidt. Each block projects all of its rows
    against the globally kept set in one batched matrix product and then
    orthonormalizes sequentially inside the block; completion draws fresh Gaussians
    once and extracts an orthonormal frame of the complement via a reduced QR. This is
    mathematically equivalent to fully sequential MGS up to floating-point rounding,
    but moves the O(d^3) work into a few GEMMs/LAPACK calls (~5x faster at d = 1000).

    Parameters
    ----------
    rng:
        Random source used for the completion draw.
    seeds:
        Matrix of shape ``(r, d)`` whose rows are the candidate directions, in priority
        order.
    block_size:
        Number of seeds processed per blocking stage (larger blocks trade memory for
        fewer Python-level iterations).

    Returns
    -------
    (basis, kept_mask)
        ``basis`` has shape ``(d, d)`` with orthonormal rows; ``kept_mask[j]`` tells whether
        seed ``j`` survived orthogonalization.
    """
    tol = 1e-12
    dim = seeds.shape[1]
    W = np.array(seeds, dtype=np.float64)
    n = len(W)
    kept_mask = np.zeros(n, dtype=bool)
    kept_rows: list[NDArray[np.float64]] = []

    b = max(1, min(int(block_size), n)) if n else 0
    start = 0
    while start < n:
        stop = min(start + b, n)
        block = W[start:stop].copy()
        if kept_rows:
            K = np.stack(kept_rows, axis=0)
            block -= (block @ K.T) @ K
        local_kept: list[NDArray[np.float64]] = []
        for j in range(len(block)):
            w = block[j]
            for u in local_kept:
                w -= float(w @ u) * u
            norm = np.linalg.norm(w)
            if norm > tol:
                wn = w / norm
                local_kept.append(wn)
                kept_mask[start + j] = True
                rest = block[j + 1 :]
                if rest.size:
                    rest -= (rest @ wn)[..., None] * wn[None, :]
        kept_rows.extend(local_kept)
        start = stop

    missing = dim - len(kept_rows)
    if missing > 0:
        Z = rng.standard_normal((missing, dim))
        if kept_rows:
            K = np.stack(kept_rows, axis=0)
            Z -= (Z @ K.T) @ K
        Q, _ = np.linalg.qr(Z.T, mode="reduced")
        kept_rows.extend(Q.T)

    return np.stack(kept_rows, axis=0), kept_mask


def build_context(
    config: BaseRunConfig,
) -> tuple[str, CountingFunction, np.random.Generator | np.random.RandomState]:
    """Resolve the objective wrapper and the single run-level RNG."""
    name, f = make_objective(config.function)
    rng = config.rng if config.rng is not None else np.random.default_rng(config.seed)
    return name, f, rng


def drive_run(
    config: BaseRunConfig,
    name: str,
    f: CountingFunction,
    rng: np.random.Generator,
    step_fn: Callable[[int, NDArray[np.float64], float], tuple[NDArray[np.float64], float]],
    start_x: NDArray[np.float64] | None = None,
) -> RunResult:
    """Generic iteration driver shared by all algorithms.

    Parameters
    ----------
    config:
        Run configuration.
    name, f:
        Objective name and counting wrapper (from :func:`build_context`).
    rng:
        The run-level random generator; used to draw ``x_0`` when neither ``config.x_init``
        nor ``start_x`` is given.
    step_fn:
        Callback ``(i, x_i, F(x_i)) -> (x_{i+1}, F(x_{i+1}))`` performing one iteration
        (including its own FE-counted evaluations).
    start_x:
        Explicit starting point (takes precedence over ``config.x_init``); lets callers
        that need ``x_0`` before the loop starts (e.g. to derive ``sigma_zero``) control
        the initialization themselves.
    """
    if start_x is not None:
        x = np.array(start_x, dtype=np.float64)
    elif config.x_init is not None:
        x = np.array(config.x_init, dtype=np.float64)
    else:
        x = rng.standard_normal(config.dim)
    assert x.shape == (config.dim,), "starting point must have shape (dim,)"

    values = [float(f(x))]
    iterates = [x.copy()]
    fes = [f.n_evaluations]
    best_index = 0

    t0 = time.perf_counter()
    for i in range(config.maxiter):
        x_new, v_new = step_fn(i, x, values[i])
        values.append(v_new)
        iterates.append(x_new)
        fes.append(f.n_evaluations)
        better = v_new < values[best_index] if not config.maximize else v_new > values[best_index]
        if better:
            best_index = len(values) - 1
        if np.linalg.norm(x_new - x) < config.eps:
            break
        x = x_new
    wall_time = time.perf_counter() - t0

    return RunResult(
        function=name,
        dim=config.dim,
        iterates=np.stack(iterates, axis=0),
        values=np.asarray(values, dtype=np.float64),
        fes_per_iterate=np.asarray(fes, dtype=np.int64),
        best_index=best_index,
        total_function_evaluations=f.n_evaluations,
        wall_time_seconds=wall_time,
    )
