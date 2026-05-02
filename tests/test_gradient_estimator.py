"""Tests for gradient estimators."""

import numpy as np
import pytest

from ashgf.gradient.estimators import (
    estimate_lipschitz_constants,
    gauss_hermite_derivative,
    gaussian_smoothing,
)


class TestGaussianSmoothing:
    """Tests for the central Gaussian smoothing estimator."""

    def test_sphere_gradient(self):
        """For f(x) = ||x||^2, grad f(x) = 2x.
        The Gaussian smoothing estimator converges as O(sqrt(d/M)).
        With M >> d, the relative error should be small.
        """
        rng = np.random.default_rng(42)
        dim = 10
        x = rng.standard_normal(dim)

        def f(x):
            return float(x @ x)

        sigma = 0.01
        M = 2000
        directions = rng.standard_normal((M, dim))

        grad_est = gaussian_smoothing(x, f, sigma, directions)
        grad_true = 2 * x

        rel_error = np.linalg.norm(grad_est - grad_true) / np.linalg.norm(grad_true)
        assert rel_error < 0.2, f"Relative error {rel_error:.4f} too high"

    def test_linear_function(self):
        """For f(x) = a @ x, the Gaussian smoothing yields an empirical
        covariance estimate that converges to the true gradient as M grows.
        """
        rng = np.random.default_rng(42)
        dim = 5
        a = rng.standard_normal(dim)

        def f(x):
            return float(np.dot(a, x))

        x = rng.standard_normal(dim)
        sigma = 0.1
        M = 5000
        directions = rng.standard_normal((M, dim))

        grad_est = gaussian_smoothing(x, f, sigma, directions)
        rel_error = np.linalg.norm(grad_est - a) / np.linalg.norm(a)
        assert rel_error < 0.1, f"Relative error {rel_error:.4f} too high"

    def test_output_shape(self):
        """Gradient estimate should have same shape as input."""
        rng = np.random.default_rng(42)
        dim = 10

        def f(x):
            return float(x @ x)

        x = rng.standard_normal(dim)
        directions = rng.standard_normal((dim, dim))
        grad = gaussian_smoothing(x, f, 0.01, directions)
        assert grad.shape == x.shape


class TestGaussHermiteDerivative:
    """Tests for Gauss-Hermite quadrature gradient estimation."""

    def test_sphere_gradient(self):
        """For f(x) = ||x||^2, the quadrature should recover the exact gradient."""
        from scipy.stats import special_ortho_group

        rng = np.random.default_rng(42)
        dim = 10
        x = rng.standard_normal(dim)

        def f(x):
            return float(x @ x)

        sigma = 0.1
        basis = special_ortho_group.rvs(dim)
        m = 5  # odd number of quadrature points

        grad_est, _, _, derivatives = gauss_hermite_derivative(
            x, f, sigma, basis, m, f(x)
        )
        grad_true = 2 * x

        # On a quadratic, the estimator should be very accurate
        rel_error = np.linalg.norm(grad_est - grad_true) / np.linalg.norm(grad_true)
        assert rel_error < 1e-6, f"Relative error {rel_error:.2e} too high"

    def test_output_shapes(self):
        """Check shapes of all returned arrays.

        .. note::
           Since the estimators module was refactored to use
           matrix-based returns, ``evals_matrix`` is ``(dim, m)``
           and ``p_nodes`` is ``(m,)``.
        """
        from scipy.stats import special_ortho_group

        rng = np.random.default_rng(42)
        dim = 10
        x = rng.standard_normal(dim)

        def f(x):
            return float(np.sum(x**2))

        sigma = 0.1
        basis = special_ortho_group.rvs(dim)
        m = 5

        grad, evals_matrix, p_nodes, derivatives = gauss_hermite_derivative(
            x, f, sigma, basis, m
        )

        assert grad.shape == (dim,)
        assert derivatives.shape == (dim,)
        assert evals_matrix.shape == (dim, m)
        assert p_nodes.shape == (m,)


class TestEstimateLipschitzConstants:
    """Tests for the Lipschitz constant estimator."""

    def test_sphere_lipschitz(self):
        """For sphere function, directional Lipschitz should be positive and
        approximately 2 along each direction (within a generous tolerance due
        to finite-difference noise)."""
        from scipy.stats import special_ortho_group

        rng = np.random.default_rng(42)
        dim = 10
        x = rng.standard_normal(dim)

        def f(x):
            return float(x @ x)

        sigma = 0.1
        basis = special_ortho_group.rvs(dim)
        m = 5

        _, evals_matrix, p_nodes, _ = gauss_hermite_derivative(
            x, f, sigma, basis, m, f(x)
        )
        lipschitz = estimate_lipschitz_constants(evals_matrix, p_nodes, sigma)

        assert lipschitz.shape == (dim,)
        assert np.all(lipschitz > 0)
        # For sphere, Lipschitz is approximately 2 per direction
        mean_lip = np.mean(lipschitz)
        assert 1.0 < mean_lip < 4.0, f"Mean Lipschitz {mean_lip:.2f} out of range"
