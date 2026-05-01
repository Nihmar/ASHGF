"""Tests for test functions in the registry."""

import numpy as np
import pytest

from ashgf.functions import get_function, list_functions


class TestSphere:
    """Tests for the sphere function f(x) = ||x||^2."""

    def test_known_minimum(self):
        """f(0) should equal 0."""
        f = get_function("sphere")
        x = np.zeros(10)
        assert f(x) == 0.0

    def test_positive_definite(self):
        """f(x) > 0 for any non-zero x."""
        f = get_function("sphere")
        x = np.ones(10)
        assert f(x) > 0

    def test_scalar_output(self):
        """Output must be a scalar float."""
        f = get_function("sphere")
        result = f(np.random.randn(10))
        assert isinstance(result, float)
        assert np.ndim(result) == 0

    def test_dim_10(self):
        """Should work with any dimension."""
        f = get_function("sphere")
        for d in [1, 5, 10, 50]:
            result = f(np.zeros(d))
            assert result == 0.0


class TestRastrigin:
    """Tests for the Rastrigin function."""

    def test_known_minimum(self):
        """f(0) should equal 0."""
        f = get_function("rastrigin")
        x = np.zeros(10)
        assert f(x) == 0.0

    def test_symmetry(self):
        """f(x) = f(-x) for Rastrigin."""
        f = get_function("rastrigin")
        x = np.random.randn(10)
        assert np.isclose(f(x), f(-x))


class TestAckley:
    """Tests for the Ackley function."""

    def test_known_minimum(self):
        """f(0) ≈ 0 (within tolerance)."""
        f = get_function("ackley")
        x = np.zeros(10)
        assert np.isclose(f(x), 0.0, atol=1e-14)


class TestLevy:
    """Tests for the Levy function."""

    def test_known_minimum(self):
        """f(1) should equal 0."""
        f = get_function("levy")
        x = np.ones(10)
        assert np.isclose(f(x), 0.0, atol=1e-12)


class TestRegistry:
    """Tests for the function registry."""

    def test_all_functions_listed(self):
        """list_functions should return a non-empty list."""
        funcs = list_functions()
        assert len(funcs) > 0
        assert "sphere" in funcs
        assert "rastrigin" in funcs

    def test_get_unknown_function_raises(self):
        """get_function with an unknown name should raise KeyError."""
        with pytest.raises(KeyError):
            get_function("nonexistent_function_xyz")

    @pytest.mark.parametrize(
        "name",
        [
            "sphere",
            "rastrigin",
            "ackley",
            "griewank",
            "levy",
            "schwefel",
            "sum_of_different_powers",
            "trid",
            "zakharov",
            "extended_rosenbrock",
            "generalized_rosenbrock",
        ],
    )
    def test_function_returns_float(self, name):
        """Each function should return a float for a random input."""
        f = get_function(name)
        x = np.random.randn(10)
        result = f(x)
        assert isinstance(result, float)
        assert np.isfinite(result)
