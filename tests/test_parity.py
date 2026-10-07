"""Parity tests between the new implementation and the frozen prototype.

These tests drive the unmodified modules in ``original_python_code/`` through
the parity adapter and check structure, exact function-evaluation accounting,
determinism, documented deviations, and — where random streams can be aligned —
bit-close trajectory agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashgf.algorithms.asebo import ASEBO, ASEBOConfig
from ashgf.algorithms.asgf import ASGF, ASGFConfig
from ashgf.algorithms.ashgf import ASHGF, ASHGFConfig
from ashgf.algorithms.random_search import GD, GDConfig
from ashgf.parity import ORIGINAL_ALGORITHMS, run_original

_ORIGINAL_AVAILABLE = True
try:  # pragma: no cover - environment dependent
    import sklearn  # noqa: F401
except ImportError:
    _ORIGINAL_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not _ORIGINAL_AVAILABLE, reason="scikit-learn not installed"),
]


# ---------------------------------------------------------------------------
# Smoke runs of every frozen algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ORIGINAL_ALGORITHMS)
def test_original_runs_smoke(algorithm):
    rec = run_original(algorithm, function="sphere", dim=5, it=20, seed=7)
    assert rec.n_iterations >= 1
    assert len(rec.values) == rec.n_iterations + 1
    assert rec.best_value <= rec.values[0]
    assert rec.total_function_evaluations > 0
    assert rec.wall_time_seconds >= 0.0


@pytest.mark.parametrize("algorithm", ORIGINAL_ALGORITHMS)
def test_original_deterministic_given_seed(algorithm):
    kwargs = {"function": "sphere", "dim": 5, "it": 15, "seed": 99}
    r1 = run_original(algorithm, **kwargs)
    r2 = run_original(algorithm, **kwargs)
    assert r1.values == r2.values
    assert r1.total_function_evaluations == r2.total_function_evaluations


# ---------------------------------------------------------------------------
# Exact FE accounting against analytic formulas
# ---------------------------------------------------------------------------


def test_gd_fe_accounting_matches_prototype_formula():
    dim, iters = 6, 30
    rec = run_original(
        "gd", function="sphere", dim=dim, it=iters, seed=5, eps=1e-30
    )
    assert rec.n_iterations == iters
    assert rec.total_function_evaluations == 1 + iters * (2 * dim + 1)


def test_sges_guidance_phase_crashes_under_modern_numpy():
    """The frozen SGES calls ``int(np.random.choice(...))`` in its guidance phase,
    which raises under NumPy >= 2. The run therefore aborts at the first
    post-warm-up iteration (i = t); only the t - 1 warm-up iterations complete."""
    dim, t, iters = 5, 8, 25
    rec = run_original(
        "sges", function="sphere", dim=dim, it=iters, seed=5, eps=1e-30,
        ctor_params={"t": t},
    )
    assert not rec.terminated_normally
    assert rec.n_iterations == t - 1
    assert rec.total_function_evaluations == 1 + (t - 1) * (2 * dim + 1)


def test_sges_warmup_only_run_is_clean_and_fully_accounted():
    """When every iteration is plain warm-up (it < t) the run terminates normally
    and each iteration costs exactly 2d + 1 evaluations."""
    dim, iters = 5, 20
    rec = run_original("sges", function="sphere", dim=dim, it=iters, seed=5, eps=1e-30)
    assert rec.terminated_normally
    assert rec.n_iterations == iters
    # Guidance draws extra RNG values but never extra objective calls.
    assert rec.total_function_evaluations == 1 + iters * (2 * dim + 1)


@pytest.mark.parametrize("algorithm", ["asgf", "ashgf"])
def test_asgf_ashf_fe_accounting_matches_prototype_formula(algorithm):
    dim, m, iters = 5, 5, 20
    rec = run_original(
        algorithm, function="sphere", dim=dim, it=iters, seed=5, eps=1e-30,
        data_overrides={"m": m},
    )
    assert rec.n_iterations == iters
    # Central GH node reuses f(x_i): d*(m-1) estimator calls + 1 new iterate.
    assert rec.total_function_evaluations == 1 + iters * (dim * (m - 1) + 1)


# ---------------------------------------------------------------------------
# Documented deviation: ASEBO warm-up cost
# ---------------------------------------------------------------------------


def test_asebo_warmup_sampling_deviation():
    """The prototype samples 100 directions during its warm-up regardless of dim,
    while the thesis prescribes n_t = d. Quantify the resulting FE gap."""
    dim, k, iters = 5, 10, 15
    old = run_original(
        "asebo", function="sphere", dim=dim, it=iters, seed=3, eps=1e-30,
        ctor_params={"k": k},
    )
    # Prototype: i < k -> 100 directions; i == k -> forced 100; i > k -> at least 10.
    lower_bound = 1 + k * (2 * 100 + 1) + (iters - k) * (2 * 10 + 1)
    assert old.total_function_evaluations >= lower_bound

    cfg = ASEBOConfig(
        function="sphere", dim=dim, maxiter=iters, seed=3, l_full=k, eps=1e-30
    )
    new = ASEBO(cfg).run()
    canonical_upper = 1 + k * (2 * dim + 1) + (iters - k) * (2 * dim + 1)
    assert new.total_function_evaluations <= canonical_upper
    assert old.total_function_evaluations > 5 * new.total_function_evaluations


# ---------------------------------------------------------------------------
# Bit-close trajectory parity for GD with aligned legacy streams
# ---------------------------------------------------------------------------


def test_gd_exact_trajectory_parity_with_shared_stream():
    """With a fixed x_init and both sides drawing from the same MT19937 stream
    (np.random.RandomState), one iteration of each implementation consumes the
    identical direction matrix; trajectories must agree up to roundoff."""
    dim, iters, seed = 8, 150, 1234
    rng0 = np.random.default_rng(0)
    x0 = rng0.standard_normal(dim)

    old = run_original(
        "gd",
        function="sphere",
        dim=dim,
        it=iters,
        seed=seed,
        x_init=x0,
        eps=1e-8,
        ctor_params={"lr": 1e-4, "sigma": 1e-4},
    )
    cfg = GDConfig(
        function="sphere",
        dim=dim,
        maxiter=iters,
        seed=seed,
        sigma=1e-4,
        lr=1e-4,
        eps=1e-8,
        x_init=x0,
        rng=np.random.RandomState(seed),
    )
    new = GD(cfg).run()

    assert old.n_iterations == new.n_iterations
    assert old.total_function_evaluations == new.total_function_evaluations
    assert np.allclose(old.values, new.values, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Legacy-rule flag sanity on the new ASGF
# ---------------------------------------------------------------------------


def test_legacy_rule_flag_changes_asgf_trajectory():
    base = {"function": "sphere", "dim": 5, "maxiter": 12, "seed": 11}
    r_new = ASGF(ASGFConfig(**base)).run()
    r_old_style = ASGF(ASGFConfig(**base, legacy_rule=True)).run()
    assert not np.array_equal(r_new.iterates[:, 1:], r_old_style.iterates[:, 1:])


def test_legacy_rule_flag_changes_ashf_trajectory():
    base = {"function": "sphere", "dim": 5, "maxiter": 12, "seed": 11, "t_warmup": 4}
    r_new = ASHGF(ASHGFConfig(**base)).run()
    r_old_style = ASHGF(ASHGFConfig(**base, legacy_rule=True)).run()
    assert not np.array_equal(r_new.iterates[:, 1:], r_old_style.iterates[:, 1:])
