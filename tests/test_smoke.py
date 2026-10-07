"""Smoke tests for the ashgf package."""

import ashgf


def test_package_imports():
    assert hasattr(ashgf, "__version__")
    assert isinstance(ashgf.__version__, str)
