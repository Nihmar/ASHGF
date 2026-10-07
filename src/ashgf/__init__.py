"""ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization.

Production-quality reimplementation of the algorithms described in the master's thesis
"Adaptive Stochastic Historical Gradient-Free Optimization" (Unipd), together with the
state-of-the-art baselines it is compared against (random search, ES-style sampling, SGES,
ASEBO, ASGF). The thesis formulas are the specification; deviations of the frozen original
prototype are documented in ``comparisons/``.
"""

__version__ = "0.1.0"

from ashgf.functions import (
    FUNCTION_NAMES,
    LEGACY_VARIANTS,
    THESIS_STARRED_FUNCTIONS,
    CountingFunction,
    Function,
    get_function,
)
from ashgf.quadrature import DGSEstimate, GHQuadrature, dgs_gradient_estimate

__all__ = [
    "FUNCTION_NAMES",
    "LEGACY_VARIANTS",
    "THESIS_STARRED_FUNCTIONS",
    "CountingFunction",
    "Function",
    "get_function",
    "DGSEstimate",
    "GHQuadrature",
    "dgs_gradient_estimate",
]
