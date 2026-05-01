"""
ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization.

A package for derivative-free optimization implementing algorithms
based on Gaussian smoothing and directional derivative estimation.
"""

__version__ = "0.2.0"
__author__ = "ASHGF Contributors"

from ashgf.algorithms.base import BaseOptimizer

from ashgf.functions import get_function, list_functions

__all__ = [
    "BaseOptimizer",
    "get_function",
    "list_functions",
    "__version__",
]
