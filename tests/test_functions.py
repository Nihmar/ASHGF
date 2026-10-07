"""Tests for the benchmark function suite."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashgf.functions import (
    FUNCTION_NAMES,
    LEGACY_VARIANTS,
    THESIS_STARRED_FUNCTIONS,
    CountingFunction,
    get_function,
)


def _random_points(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n, d))


@pytest.mark.parametrize("name", FUNCTION_NAMES)
def test_batch_matches_rowwise(name: str) -> None:
    """Batch evaluation must equal independent row-wise evaluation (internal consistency)."""
    rng = np.random.default_rng(1234)
    fn = get_function(name)
    X = _random_points(rng, 5, 7)
    batched = fn.evaluate(X)
    rowwise = np.array([fn.evaluate(row) for row in X])
    assert batched.shape == (5,)
    assert np.allclose(batched, rowwise, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", FUNCTION_NAMES)
def test_single_and_batch_agree(name: str) -> None:
    rng = np.random.default_rng(99)
    fn = get_function(name)
    x = rng.uniform(-1.0, 1.0, size=6)
    single = fn.evaluate(x)
    batched = fn.evaluate(x[None, :])
    assert isinstance(single, float)
    assert math.isfinite(single)
    assert single == pytest.approx(float(batched[0]), rel=1e-12, abs=1e-12)


def test_known_values_at_origin() -> None:
    checks = {
        "sphere": 0.0,
        "power": 0.0,
        "perturbed_quadratic": 0.0,
        "ackley": 0.0,
        "griewank": 0.0,
        "zakharov": 0.0,
        "rastrigin": 0.0,
    }
    for name, expected in checks.items():
        x = np.zeros(8)
        assert get_function(name).evaluate(x) == pytest.approx(expected, abs=1e-12)


def test_rosenbrock_family_zero_at_ones() -> None:
    for name in (
        "generalized_rosenbrock",
        "extended_rosenbrock",
        "generalized_white_and_holst",
        "extended_white_and_holst",
        "cube",
        "nonscomp",
    ):
        x = np.ones(10)
        assert get_function(name).evaluate(x) == pytest.approx(0.0, abs=1e-12)


def test_levy_zero_at_ones() -> None:
    assert get_function("levy").evaluate(np.ones(9)) == pytest.approx(0.0, abs=1e-12)


def test_odd_dimension_pair_splitting_drops_last_entry() -> None:
    """Pair-based functions on odd dimensions must ignore the final coordinate."""
    fn = get_function("extended_rosenbrock")
    even = np.linspace(-0.5, 0.5, 8)
    with_last = np.append(even, 7.0)  # wildly different last entry
    assert fn.evaluate(with_last) == pytest.approx(fn.evaluate(even), rel=1e-12, abs=1e-12)


# --- Hand-computed spot checks guarding transcription errors -----------------


def test_eg2_hand_computed() -> None:
    a, b, c = 0.3, -0.7, 0.9
    x = np.array([a, b, c])
    # Terms run over i = 0..n-2: sin(x_0 + x_i^2 - 1); plus 0.5 sin(x_n^2).
    expected = math.sin(a + a**2 - 1) + math.sin(a + b**2 - 1) + 0.5 * math.sin(c**2)
    assert get_function("eg2").evaluate(x) == pytest.approx(expected, rel=1e-12)


def test_arwhead_hand_computed() -> None:
    a, b, c = 0.2, 1.1, -0.4
    x = np.array([a, b, c])
    expected = (-4 * a + 3) + (-4 * b + 3) + (a**2 + c**2) ** 2 + (b**2 + c**2) ** 2
    assert get_function("arwhead").evaluate(x) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_nondquar_hand_computed() -> None:
    a, b, c = 0.5, -0.2, 0.8
    x = np.array([a, b, c])
    expected = (a - b) ** 2 + (a + b + c) ** 4 + (b + c) ** 2
    assert get_function("nondquar").evaluate(x) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_bdqrtic_hand_computed() -> None:
    x = np.array([0.1, -0.3, 0.6, 0.2, 0.9])
    xs = x**2
    t1 = sum((-4 * xi + 3) ** 2 for xi in x[:-3])
    t2 = 0.0
    for i in range(len(x) - 3):
        s = xs[i] + 2 * xs[i + 1] + 3 * xs[i + 2] + 4 * xs[i + 3] + 5 * xs[-1]
        t2 += s**2
    assert get_function("bdqrtic").evaluate(x) == pytest.approx(t1 + t2, rel=1e-12, abs=1e-12)


def test_fletcbv3_hand_computed() -> None:
    x = np.array([0.4, -0.6, 0.2])
    d = len(x)
    p, h, c = 1e-8, 1 / (d + 1), 1.0
    t1 = 0.5 * p * (x[0] ** 2 + x[-1] ** 2)
    t2 = sum((p / 2) * (x[i] - x[i + 1]) ** 2 for i in range(d - 1))
    t3 = sum(p * (h**2 + 2) / h**2 * xi + c * p / h**2 * math.cos(xi) for xi in x)
    assert get_function("fletcbv3").evaluate(x) == pytest.approx(t1 + t2 + t3, rel=1e-9, abs=1e-14)


# --- Legacy variants ----------------------------------------------------------


@pytest.mark.parametrize("name", list(LEGACY_VARIANTS))
def test_legacy_variant_registered_and_differs(name: str) -> None:
    base_name = name[: -len("_original")]
    rng = np.random.default_rng(7)
    X = _random_points(rng, 4, 8)
    legacy_vals = get_function(name).evaluate(X)
    canonical_vals = get_function(base_name).evaluate(X)
    # They must be real, distinct functions on generic inputs.
    assert not np.allclose(legacy_vals, canonical_vals)


def test_trid_sign_convention() -> None:
    """Canonical trid subtracts the cross term; the original adds it."""
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, size=6)
    base = float(np.sum((x - 1.0) ** 2))
    cross = float(np.sum(x[1:] * x[:-1]))
    assert get_function("trid").evaluate(x) == pytest.approx(base - cross, rel=1e-12)
    assert get_function("trid_original").evaluate(x) == pytest.approx(base + cross, rel=1e-12)


# --- API -----------------------------------------------------------------------


def test_unknown_function_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown function"):
        get_function("does_not_exist")


def test_counting_function_counts_batches() -> None:
    fn = CountingFunction(get_function("sphere").evaluate_batched)
    single = fn(np.zeros(5))
    assert isinstance(single, float)
    assert fn.n_evaluations == 1
    fn(np.zeros((4, 5)))
    assert fn.n_evaluations == 5
    fn(np.ones((2, 3, 5)))
    assert fn.n_evaluations == 11


def test_thesis_starred_subset_is_valid() -> None:
    assert all(name in FUNCTION_NAMES for name in THESIS_STARRED_FUNCTIONS)
