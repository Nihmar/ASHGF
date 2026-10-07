"""Tests for the optimization algorithm layer.

Covers determinism, exact function-evaluation accounting, mathematical invariants
(step direction on quadratics), configuration validation, and the shared building
blocks (adaptive-smoothing state machine, Gram-Schmidt completion, subspace bases).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ashgf.algorithms._adaptive import AdaptiveSmoothingState
from ashgf.algorithms._common import (
    HistoryBuffer,
    gram_schmidt_complete,
    random_orthonormal_basis,
    subspace_basis,
)
from ashgf.algorithms.asebo import ASEBO, ASEBOConfig
from ashgf.algorithms.asgf import ASGF, ASGFConfig
from ashgf.algorithms.ashgf import ASHGF, ASHGFConfig
from ashgf.algorithms.random_search import GD, GDConfig
from ashgf.algorithms.sges import SGES, SGESConfig
from ashgf.quadrature import GHQuadrature

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _quadratic_objective(a: NDArray[np.float64], b: NDArray[np.float64]):
    """Batched evaluator of F(x) = x^T A x + b^T x."""

    def evaluate(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return (points * (points @ a)).sum(axis=-1) + points @ b

    return evaluate


@pytest.fixture(scope="module")
def quadratic_setup():
    rng = np.random.default_rng(123)
    dim = 4
    a_sym = rng.standard_normal((dim, dim))
    mat = a_sym + a_sym.T + dim * np.eye(dim)  # SPD
    b = rng.standard_normal(dim)
    x0 = rng.standard_normal(dim)
    return mat, b, x0


def _mk(config_cls, base: dict, **overrides):
    """Build a config merging base defaults with per-test overrides."""
    return config_cls(**{**base, **overrides})


_BASE = {"function": "sphere", "dim": 5, "seed": 11}

ALL_ALGORITHMS = [
    (GD, lambda **kw: _mk(GDConfig, {**_BASE, "maxiter": 20}, **kw)),
    (SGES, lambda **kw: _mk(SGESConfig, {**_BASE, "maxiter": 30, "k": 8, "t_warmup": 8}, **kw)),
    (ASEBO, lambda **kw: _mk(ASEBOConfig, {**_BASE, "maxiter": 30, "l_full": 8}, **kw)),
    (ASGF, lambda **kw: _mk(ASGFConfig, {**_BASE, "maxiter": 20}, **kw)),
    (ASHGF, lambda **kw: _mk(ASHGFConfig, {**_BASE, "maxiter": 30, "k": 8, "t_warmup": 8}, **kw)),
]


# ---------------------------------------------------------------------------
# Determinism and result structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alg,mkcfg", ALL_ALGORITHMS, ids=[a.__name__ for a, _ in ALL_ALGORITHMS])
def test_deterministic_runs(alg, mkcfg):
    r1 = alg(mkcfg()).run()
    r2 = alg(mkcfg()).run()
    assert np.array_equal(r1.iterates, r2.iterates)
    assert np.array_equal(r1.values, r2.values)
    assert r1.total_function_evaluations == r2.total_function_evaluations


@pytest.mark.parametrize("alg,mkcfg", ALL_ALGORITHMS, ids=[a.__name__ for a, _ in ALL_ALGORITHMS])
def test_result_structure_and_best_index(alg, mkcfg):
    cfg = mkcfg()
    r = alg(cfg).run()
    assert r.iterates.shape == (r.n_iterations + 1, cfg.dim)
    assert r.values.shape == (r.n_iterations + 1,)
    assert r.iterates[0].shape == (cfg.dim,)
    if not cfg.maximize:
        assert r.best_value == pytest.approx(r.values.min())
        assert r.values[r.best_index] == r.values.min()
    else:
        assert r.best_value == pytest.approx(r.values.max())
    assert 0 <= r.best_index <= r.n_iterations
    assert r.wall_time_seconds >= 0.0


def test_maximization_mode_tracks_maximum():
    cfg = GDConfig(function="sphere", dim=4, maxiter=15, seed=3, maximize=True)
    r = GD(cfg).run()
    # Ascent on the sphere: values should grow well above the start.
    assert r.values[-1] > r.values[0]
    assert r.best_value == pytest.approx(r.values.max())


def test_x_init_is_respected():
    x0 = np.full(3, 2.0)
    cfg = GDConfig(function="sphere", dim=3, maxiter=1, seed=1, x_init=x0)
    r = GD(cfg).run()
    assert np.array_equal(r.iterates[0], x0)
    assert r.values[0] == pytest.approx(np.sum(x0**2))


def test_bad_x_init_shape_raises():
    cfg = GDConfig(function="sphere", dim=3, maxiter=1, seed=1, x_init=np.zeros(2))
    with pytest.raises(AssertionError):
        GD(cfg).run()


# ---------------------------------------------------------------------------
# Function-evaluation accounting
# ---------------------------------------------------------------------------


def test_gd_fe_accounting():
    dim, m, iters = 6, 6, 25
    cfg = GDConfig(function="sphere", dim=dim, maxiter=iters, seed=5, m_directions=m, eps=1e-30)
    r = GD(cfg).run()
    assert r.n_iterations == iters
    assert r.total_function_evaluations == 1 + iters * (2 * m + 1)


def test_sges_fe_accounting():
    dim, iters = 5, 25
    cfg = SGESConfig(function="sphere", dim=dim, maxiter=iters, seed=5, k=8, t_warmup=8, eps=1e-30)
    r = SGES(cfg).run()
    assert r.n_iterations == iters
    assert r.total_function_evaluations == 1 + iters * (2 * dim + 1)


def test_asgf_fe_accounting():
    dim, m, iters = 5, 5, 20
    cfg = ASGFConfig(function="sphere", dim=dim, maxiter=iters, seed=5, m=m, eps=1e-30)
    r = ASGF(cfg).run()
    assert r.n_iterations == iters
    # Central node reuses f(x_i): dim*(m-1) estimator calls + 1 new iterate per step.
    assert r.total_function_evaluations == 1 + iters * (dim * (m - 1) + 1)


def test_ashgf_fe_accounting():
    dim, m, iters = 5, 7, 20
    base = {"function": "sphere", "dim": dim, "maxiter": iters, "seed": 5, "m": m, "eps": 1e-30}
    cfg = ASHGFConfig(k=8, t_warmup=8, **base)
    r = ASHGF(cfg).run()
    assert r.n_iterations == iters
    assert r.total_function_evaluations == 1 + iters * (dim * (m - 1) + 1)


def test_asebo_fe_accounting_from_active_rank():
    dim, l_full, iters = 5, 8, 25
    cfg = ASEBOConfig(function="sphere", dim=dim, maxiter=iters, seed=5, l_full=l_full, eps=1e-30)
    r = ASEBO(cfg).run()
    assert r.n_iterations == iters
    n_t = [dim] * l_full + [int(v) for v in r.extras["active_rank"]]
    expected = 1 + sum(2 * n + 1 for n in n_t)
    assert r.total_function_evaluations == expected
    # Active rank always within [1, dim]; never exceeds full dimension.
    assert all(1 <= n <= dim for n in r.extras["active_rank"])


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alg,mkcfg",
    [(ASGF, lambda **kw: ASGFConfig(**kw)), (ASHGF, lambda **kw: ASHGFConfig(**kw))],
)
def test_dgs_step_is_exactly_antiparallel_to_true_gradient(alg, mkcfg, quadratic_setup):
    """DGS along an orthonormal basis is exact for quadratics, so one step must move
    exactly opposite to the true gradient (only the magnitude may differ)."""
    mat, b, x0 = quadratic_setup
    cfg = mkcfg(function=_quadratic_objective(mat, b), dim=len(x0), maxiter=1, seed=42, x_init=x0)
    r = alg(cfg).run()
    step_dir = r.iterates[1] - r.iterates[0]
    true_grad = 2 * mat @ x0 + b
    cos = float(step_dir @ true_grad / (np.linalg.norm(step_dir) * np.linalg.norm(true_grad)))
    assert cos < -1.0 + 1e-9


@pytest.mark.parametrize("alg,mkcfg", ALL_ALGORITHMS, ids=[a.__name__ for a, _ in ALL_ALGORITHMS])
def test_all_algorithms_make_progress_on_sphere(alg, mkcfg):
    cfg = mkcfg(dim=8, maxiter=200)
    r = alg(cfg).run()
    assert r.best_value < r.values[0]


def test_alpha_stays_within_bounds():
    common = {"k": 8, "t_warmup": 8, "kappa_1": 0.8, "kappa_2": 0.2}
    pairs = [(SGES, SGESConfig), (ASHGF, ASHGFConfig)]
    for cls, cfg_cls in pairs:
        cfg = cfg_cls(function="sphere", dim=6, maxiter=60, seed=9, **common)
        r = cls(cfg).run()
        alpha = r.extras["alpha"]
        assert len(alpha) > 0
        assert np.all(alpha >= 0.2 - 1e-12) and np.all(alpha <= 0.8 + 1e-12)


def test_sigma_trajectory_recorded_and_positive():
    configs = [
        (ASGF, ASGFConfig(function="sphere", dim=5, maxiter=30, seed=2)),
        (ASHGF, ASHGFConfig(function="sphere", dim=5, maxiter=30, seed=2, k=8, t_warmup=8)),
    ]
    for cls, cfg in configs:
        r = cls(cfg).run()
        sigma = r.extras["sigma"]
        assert len(sigma) == r.n_iterations
        assert np.all(sigma > 0)


# ---------------------------------------------------------------------------
# AdaptiveSmoothingState unit tests
# ---------------------------------------------------------------------------


def _state(**overrides):
    params = {"sigma_zero": 1.0, "resets": 2}
    params.update(overrides)
    return AdaptiveSmoothingState(**params)


def test_adapt_reset_branch():
    st = _state(resets=1)
    st.sigma = 0.5 * 0.01  # below rho*sigma_zero with default rho=0.01
    assert st.adapt(np.array([1.0]), np.array([1.0])) is True
    assert st.sigma == pytest.approx(1.0)
    assert st.resets_left == 0
    assert st.a == pytest.approx(0.1)
    assert st.b == pytest.approx(0.9)


def test_adapt_reset_exhausted_falls_through():
    st = _state(resets=0)
    st.sigma = 1e-6
    assert st.adapt(np.array([1.0]), np.array([1.0])) is False
    assert st.sigma != 1.0  # no reset happened; threshold logic applied instead


def test_adapt_threshold_branches():
    # ratio far below A -> shrink sigma, lower A
    st = _state()
    assert st.adapt(np.array([0.01]), np.array([1.0])) is False
    assert st.sigma == pytest.approx(0.9)
    assert st.a == pytest.approx(0.1 * 0.95)

    # ratio above B -> grow sigma, raise B
    st = _state()
    assert st.adapt(np.array([10.0]), np.array([1.0])) is False
    assert st.sigma == pytest.approx(1.0 / 0.9)
    assert st.b == pytest.approx(0.9 * 1.01)

    # ratio inside [A, B] -> tighten both thresholds only
    st = _state()
    assert st.adapt(np.array([0.5]), np.array([1.0])) is False
    assert st.sigma == pytest.approx(1.0)
    assert st.a == pytest.approx(0.1 * 1.02)
    assert st.b == pytest.approx(0.9 * 0.98)


def test_adapt_degenerate_lipschitz_pushes_sigma_up():
    st = _state()
    # L_j = 0 but nonzero derivative: inconsistent -> treat ratio as inf.
    assert st.adapt(np.array([0.4]), np.array([0.0])) is False
    assert st.sigma == pytest.approx(1.0 / 0.9)


def test_learning_rate_floor():
    st = _state()
    st.l_nabla = 0.0
    assert st.learning_rate == pytest.approx(st.sigma / 1e-30)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def test_random_basis_is_orthonormal():
    rng = np.random.default_rng(0)
    q = random_orthonormal_basis(rng, 7)
    assert q.shape == (7, 7)
    assert q @ q.T == pytest.approx(np.eye(7), abs=1e-12)


def test_subspace_basis_tall_matrix():
    rng = np.random.default_rng(1)
    d, n = 4, 30
    g = rng.standard_normal((n, d))
    basis = subspace_basis(g)
    assert basis.ndim == 2 and basis.shape[1] == d
    assert basis.shape[0] == d  # full rank expected
    assert basis @ basis.T == pytest.approx(np.eye(d), abs=1e-10)
    # Every gradient lies in the span of the basis rows.
    u, s, vh = np.linalg.svd(basis, full_matrices=False)
    proj = vh.T @ vh
    resid = np.linalg.norm(proj @ g.T - g.T)
    assert resid < 1e-8


def test_subspace_basis_rank_deficient():
    rng = np.random.default_rng(2)
    d = 5
    v = rng.standard_normal(d)
    g = np.tile(v, (10, 1))  # exactly rank-1
    basis = subspace_basis(g)
    assert basis.shape == (1, d)
    assert np.linalg.norm(basis[0]) == pytest.approx(1.0, abs=1e-10)
    # Basis aligned with the dominant direction up to sign.
    assert abs(float(basis[0] @ v / np.linalg.norm(v))) > 1.0 - 1e-6


def test_gram_schmidt_complete_orthonormal_and_spanning():
    rng = np.random.default_rng(3)
    seeds = rng.standard_normal((3, 6))
    basis, kept = gram_schmidt_complete(rng, seeds)
    assert basis.shape == (6, 6)
    assert basis @ basis.T == pytest.approx(np.eye(6), abs=1e-10)
    assert len(kept) == 3 and int(kept.sum()) == 3
    # The first surviving seed defines the first row exactly (up to nothing: it is kept
    # unmodified); later rows are orthogonalized away from earlier ones.
    assert np.allclose(basis[0], seeds[0] / np.linalg.norm(seeds[0]), atol=1e-12)


def test_gram_schmidt_drops_degenerate_seeds():
    rng = np.random.default_rng(4)
    e0 = np.zeros(4)
    e0[0] = 1.0
    seeds = np.vstack([e0, e0.copy(), e0.copy(), rng.standard_normal(4)])
    basis, kept = gram_schmidt_complete(rng, seeds)
    assert kept.tolist() == [True, False, False, True]
    assert basis @ basis.T == pytest.approx(np.eye(4), abs=1e-10)
    # First surviving seed defines the first row (up to sign).
    assert abs(float(basis[0] @ e0)) == pytest.approx(1.0, abs=1e-10)


def test_history_buffer_fifo_capacity():
    buf = HistoryBuffer(capacity=3)
    for i in range(5):
        buf.append(np.full(2, float(i)))
    assert len(buf) == 3
    m = buf.as_matrix()
    assert m is not None and m.shape == (3, 2)
    assert np.array_equal(m[:, 0], [2.0, 3.0, 4.0])

    empty = HistoryBuffer(capacity=2)
    assert empty.as_matrix() is None


# ---------------------------------------------------------------------------
# Audit claim: prototype's broken pair filter equals the thesis' set I
# ---------------------------------------------------------------------------


def test_prototype_pair_filter_equivalent_to_thesis_set_i():
    """The frozen ashgf.py builds an ordered-pair buffer via a precedence-broken
    condition; we verify that maximizing the secant slope over that (ordered) buffer
    gives exactly the same value as over the thesis' admissible unordered set I."""
    quad = GHQuadrature.of_order(5)
    rng = np.random.default_rng(7)
    values = rng.standard_normal((quad.m, 9))  # per-node evaluations along 9 directions
    scale = 1.0

    def slope(i: int, k: int) -> NDArray[np.float64]:
        return np.abs(values[i] - values[k]) / (scale * abs(quad.nodes[i] - quad.nodes[k]))

    # Prototype logic: every ordered pair (i, j) with |i-c| != |j-c| enters the buffer.
    c = quad.center_index
    proto_pairs = [(i, j) for i in range(quad.m) for j in range(quad.m) if abs(i - c) != abs(j - c)]
    proto_max = max(slope(i, j).max() for i, j in proto_pairs)
    thesis_max = max(slope(a, b).max() for a, b in quad.pairs)
    assert proto_max == pytest.approx(thesis_max)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,mk", [
    (ASGF, lambda **kw: ASGFConfig(function="sphere", dim=3, maxiter=1, seed=0, **kw)),
    (ASHGF, lambda **kw: ASHGFConfig(function="sphere", dim=3, maxiter=1, seed=0, **kw)),
])
def test_even_quadrature_order_rejected(cls, mk):
    with pytest.raises(ValueError):
        cls(mk(m=4)).run()


def test_custom_callable_objective_supported():
    def f(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(points**2, axis=-1) + 3.0

    cfg = GDConfig(function=f, dim=2, maxiter=3, seed=0)
    r = GD(cfg).run()
    assert r.function == "custom"
    assert r.values[0] == pytest.approx(float(f(np.array([r.iterates[0]]))[0]))
