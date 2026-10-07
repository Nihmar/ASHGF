"""Tests for the Gauss-Hermite DGS gradient estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import qr

from ashgf.functions import CountingFunction, get_function
from ashgf.quadrature import GHQuadrature, dgs_gradient_estimate


def _random_basis(rng: np.random.Generator, dim: int) -> np.ndarray:
    """A random orthonormal basis (rows are unit directions)."""
    Q, _ = qr(rng.standard_normal((dim, dim)))
    return Q


# --- Quadrature rule ----------------------------------------------------------


@pytest.mark.parametrize("m", [3, 5, 7])
def test_hermgauss_exact_for_polynomials(m: int) -> None:
    """The rule integrates e^{-u^2} p(u) exactly for deg(p) <= 2m - 1."""
    quad = GHQuadrature.of_order(m)
    rng = np.random.default_rng(0)
    for degree in range(2 * m):
        coeffs = rng.standard_normal(degree + 1)  # ascending powers
        # Exact integral: odd moments vanish; int e^{-u^2} u^{2j} du = sqrt(pi) (2j)! / (4^j j!).
        from math import factorial

        exact = 0.0
        for j in range(degree // 2 + 1):
            if 2 * j >= len(coeffs):
                break
            exact += coeffs[2 * j] * math.sqrt(math.pi) * factorial(2 * j) / (4.0**j * factorial(j))
        poly_nodes = np.polynomial.polynomial.polyval(quad.nodes, coeffs)
        quadrature = float(np.dot(quad.weights, poly_nodes))
        assert quadrature == pytest.approx(exact, rel=1e-10, abs=1e-12)


def test_quadrature_rejects_invalid_orders() -> None:
    with pytest.raises(ValueError):
        GHQuadrature.of_order(4)
    with pytest.raises(ValueError):
        GHQuadrature.of_order(1)


def test_lipschitz_pair_set_m5() -> None:
    """For m=5 the admissible pair set excludes only center-symmetric pairs."""
    quad = GHQuadrature.of_order(5)
    expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)}
    assert set(quad.pairs) == expected
    assert (0, 4) not in set(quad.pairs)
    assert (1, 3) not in set(quad.pairs)


# --- Estimator correctness -----------------------------------------------------


def _gradient_of(name: str, x: np.ndarray) -> np.ndarray:
    """Analytic gradients of the test objectives used below."""
    if name == "sphere":
        return 2.0 * x
    if name == "power":
        idx = np.arange(1, x.shape[-1] + 1)
        return 2.0 * idx * x
    if name == "perturbed_quadratic":
        idx = np.arange(1, x.shape[-1] + 1)
        return 2.0 * idx * x + (np.sum(x) / 50.0) * np.ones_like(x)
    if name == "raydan_2":
        return np.exp(x) - 1.0
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["sphere", "power", "perturbed_quadratic"])
def test_estimator_exact_on_quadratics(name: str) -> None:
    """Quadratic objectives have vanishing third derivative: the estimate is exact at any sigma."""
    rng = np.random.default_rng(42)
    dim = 8
    x = rng.standard_normal(dim)
    basis = _random_basis(rng, dim)
    fn = get_function(name).evaluate_batched
    est = dgs_gradient_estimate(fn, x, basis, sigma=0.05, quad=GHQuadrature.of_order(5))
    assert np.allclose(est.gradient, _gradient_of(name, x), rtol=1e-10, atol=1e-10)


def test_error_scales_as_sigma_squared() -> None:
    """On a smooth non-quadratic objective the error must shrink ~4x when sigma halves."""
    rng = np.random.default_rng(7)
    dim = 5
    x = rng.uniform(-1.0, 1.0, size=dim)
    basis = _random_basis(rng, dim)
    fn = get_function("raydan_2").evaluate_batched
    true_grad = _gradient_of("raydan_2", x)

    errors = {}
    for sigma in (0.2, 0.1):
        est = dgs_gradient_estimate(fn, x, basis, sigma=sigma, quad=GHQuadrature.of_order(5))
        errors[sigma] = np.linalg.norm(est.gradient - true_grad)

    ratio = errors[0.2] / errors[0.1]
    assert errors[0.1] < errors[0.2]
    assert 2.0 <= ratio <= 10.0  # O(sigma^2) regime, loose band for finite-sigma effects


def test_linear_objective_gives_exact_derivatives_and_lipschitz() -> None:
    """For F(x) = a.x both the derivatives and every secant slope are exact."""
    rng = np.random.default_rng(11)
    dim = 6
    a = rng.standard_normal(dim)
    x = rng.standard_normal(dim)
    basis = _random_basis(rng, dim)

    def f(points: np.ndarray) -> np.ndarray:
        return points @ a

    quad = GHQuadrature.of_order(5)
    est = dgs_gradient_estimate(f, x, basis, sigma=0.3, quad=quad)
    proj = basis @ a  # components of a along each direction ξ_j
    assert np.allclose(est.directional_derivatives, proj, rtol=1e-12, atol=1e-12)
    assert np.allclose(est.lipschitz_constants, np.abs(proj), rtol=1e-12, atol=1e-12)
    assert np.allclose(est.gradient, a, rtol=1e-10, atol=1e-10)


def test_legacy_rule_matches_thesis_on_quadratics_but_differs_elsewhere() -> None:
    rng = np.random.default_rng(5)
    dim = 5
    x = rng.standard_normal(dim)
    basis = _random_basis(rng, dim)
    quad = GHQuadrature.of_order(5)

    quad_fn = get_function("sphere").evaluate_batched
    thesis = dgs_gradient_estimate(quad_fn, x, basis, sigma=0.1, quad=quad)
    legacy = dgs_gradient_estimate(quad_fn, x, basis, sigma=0.1, quad=quad, legacy_rule=True)
    assert np.allclose(thesis.gradient, legacy.gradient, rtol=1e-12, atol=1e-12)

    smooth_fn = get_function("raydan_2").evaluate_batched
    thesis = dgs_gradient_estimate(smooth_fn, x, basis, sigma=0.1, quad=quad)
    legacy = dgs_gradient_estimate(smooth_fn, x, basis, sigma=0.1, quad=quad, legacy_rule=True)
    assert not np.allclose(thesis.gradient, legacy.gradient, rtol=1e-9, atol=1e-9)


# --- Function evaluation accounting --------------------------------------------


def test_fe_count_with_and_without_reused_center_value() -> None:
    dim, m = 7, 5
    x = np.zeros(dim)
    basis = np.eye(dim)
    counter = CountingFunction(get_function("sphere").evaluate_batched)
    est = dgs_gradient_estimate(counter, x, basis, sigma=0.1, quad=GHQuadrature.of_order(m))
    assert est.n_function_evaluations == dim * (m - 1) + 1
    assert counter.n_evaluations == est.n_function_evaluations

    counter2 = CountingFunction(get_function("sphere").evaluate_batched)
    est2 = dgs_gradient_estimate(
        counter2, x, basis, sigma=0.1, quad=GHQuadrature.of_order(m), f_x=float(x @ x)
    )
    assert est2.n_function_evaluations == dim * (m - 1)
    assert np.allclose(est.gradient, est2.gradient, rtol=1e-12, atol=1e-12)


def test_chunking_does_not_change_the_result() -> None:
    rng = np.random.default_rng(1)
    dim = 12
    x = rng.standard_normal(dim)
    basis = _random_basis(rng, dim)
    fn = get_function("ackley").evaluate_batched
    full = dgs_gradient_estimate(fn, x, basis, sigma=0.07, quad=GHQuadrature.of_order(5))
    chunked = dgs_gradient_estimate(
        fn, x, basis, sigma=0.07, quad=GHQuadrature.of_order(5), chunk_size=4
    )
    assert np.allclose(full.gradient, chunked.gradient, rtol=1e-12, atol=1e-12)
    assert np.allclose(full.lipschitz_constants, chunked.lipschitz_constants, rtol=1e-12,
                       atol=1e-12)
    assert full.n_function_evaluations == chunked.n_function_evaluations


def test_invalid_arguments_raise() -> None:
    x = np.zeros(4)
    basis = np.eye(4)
    fn = get_function("sphere").evaluate_batched
    with pytest.raises(ValueError):
        dgs_gradient_estimate(fn, x, np.eye(3), sigma=0.1, quad=GHQuadrature.of_order(5))
    with pytest.raises(ValueError):
        dgs_gradient_estimate(fn, x, basis, sigma=-1.0, quad=GHQuadrature.of_order(5))
