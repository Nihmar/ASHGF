"""Tests for direction sampling functions."""

import numpy as np
import pytest

from ashgf.gradient.sampling import (
    compute_directions,
    compute_directions_ashgf,
    compute_directions_sges,
)


class TestComputeDirections:
    """Tests for compute_directions."""

    def test_shape(self):
        """Should return a (dim, dim) array."""
        dim = 10
        dirs = compute_directions(dim)
        assert dirs.shape == (dim, dim)

    def test_statistics(self):
        """Entries should be approximately N(0, 1)."""
        dim = 1000
        dirs = compute_directions(dim)
        # Mean ≈ 0, Std ≈ 1
        assert abs(np.mean(dirs)) < 0.1
        assert abs(np.std(dirs) - 1.0) < 0.1


class TestComputeDirectionsSGES:
    """Tests for compute_directions_sges."""

    def test_shape(self):
        """Should return (dim, dim) directions and an int choices."""
        dim = 10
        G = [np.random.randn(dim) for _ in range(20)]
        dirs, choices = compute_directions_sges(dim, G, alpha=0.5)
        assert dirs.shape == (dim, dim)
        assert isinstance(choices, (int, np.integer))
        assert 0 <= choices <= dim

    def test_normalized(self):
        """All directions should have unit norm (or close)."""
        dim = 10
        G = [np.random.randn(dim) for _ in range(20)]
        dirs, _ = compute_directions_sges(dim, G, alpha=0.5)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_alpha_zero_all_gradient(self):
        """With alpha=0, all directions come from the gradient subspace
        (choices == dim)."""
        dim = 10
        G = [np.random.randn(dim) for _ in range(50)]
        dirs, choices = compute_directions_sges(dim, G, alpha=0.0)
        assert choices == dim

    def test_alpha_one_all_random(self):
        """With alpha=1, all directions are random (choices == 0)."""
        dim = 10
        G = [np.random.randn(dim) for _ in range(50)]
        dirs, choices = compute_directions_sges(dim, G, alpha=1.0)
        assert choices == 0


class TestComputeDirectionsASHGF:
    """Tests for compute_directions_ashgf."""

    def test_delegates_to_sges(self):
        """Should behave like compute_directions_sges."""
        dim = 10
        G = [np.random.randn(dim) for _ in range(20)]
        dirs1, M1 = compute_directions_ashgf(dim, G, alpha=0.5)
        dirs2, choices2 = compute_directions_sges(dim, G, alpha=0.5)
        assert M1 == choices2
        assert dirs1.shape == dirs2.shape
