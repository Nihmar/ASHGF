"""Shared fixtures for the ASHGF test suite."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Return a seeded NumPy random generator."""
    return np.random.default_rng(2003)


@pytest.fixture
def dim():
    """Default problem dimension for tests."""
    return 10


@pytest.fixture
def x0(dim, rng):
    """Random initial point."""
    return rng.standard_normal(dim)


@pytest.fixture
def sphere_func():
    """Return the sphere function f(x) = ||x||^2."""
    from ashgf.functions.classic import sphere

    return sphere


@pytest.fixture
def rastrigin_func():
    """Return the Rastrigin function."""
    from ashgf.functions.classic import rastrigin

    return rastrigin


@pytest.fixture
def ackley_func():
    """Return the Ackley function."""
    from ashgf.functions.classic import ackley

    return ackley


@pytest.fixture
def levy_func():
    """Return the Levy function."""
    from ashgf.functions.classic import levy

    return levy
