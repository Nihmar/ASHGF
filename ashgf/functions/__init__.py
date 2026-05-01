"""
Function registry for optimization test functions.

Provides a central registry mapping function names to callable objects,
supporting both analytical test functions and RL environments.
"""

from __future__ import annotations

import logging
from typing import Callable

from ashgf.functions.benchmark import (
    almost_perturbed_quadratic,
    arwhead,
    bdexp,
    bdqrtic,
    biggsb1,
    broyden_tridiagonal,
    cube,
    diagonal_1,
    diagonal_2,
    diagonal_3,
    diagonal_4,
    diagonal_5,
    diagonal_7,
    diagonal_8,
    diagonal_9,
    dixon3dq,
    dqdrtic,
    edensch,
    eg2,
    engval1,
    extended_quadratic_penalty_qp1,
    extended_quadratic_penalty_qp2,
    fh3,
    fletcbv3,
    fletchcr,
    generalized_quartic,
    genhumps,
    hager,
    himmelbg,
    himmelh,
    indef,
    liarwhd,
    mccormck,
    nondia,
    nondquar,
    nonscomp,
    perturbed_quadratic,
    perturbed_quadratic_diagonal,
    power,
    quadratic_qf1,
    quadratic_qf2,
    quartc,
    raydan_1,
    raydan_2,
    sinquad,
    tridia,
    vardim,
)
from ashgf.functions.classic import (
    ackley,
    cosine,
    griewank,
    levy,
    rastrigin,
    schwefel,
    sincos,
    sine,
    sphere,
    sum_of_different_powers,
    trid,
    zakharov,
)
from ashgf.functions.extended import (
    extended_baele,
    extended_bd1,
    extended_cliff,
    extended_denschnb,
    extended_denschnf,
    extended_feudenstein_and_roth,
    extended_hiebert,
    extended_himmelblau,
    extended_maratos,
    extended_penalty,
    extended_psc1,
    extended_quadratic_exponential_ep1,
    extended_rosenbrock,
    extended_tridiagonal_1,
    extended_tridiagonal_2,
    extended_trigonometric,
    extended_white_and_holst,
    generalized_rosenbrock,
    generalized_white_and_holst,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, Callable[[np.ndarray], float]] = {}

import numpy as np


def _populate_registry() -> None:
    """Build the internal registry from all imported function symbols."""
    import inspect
    import sys

    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module, inspect.isfunction):
        if not name.startswith("_"):
            _registry[name] = obj

    # Also register RL environments if available
    try:
        from ashgf.functions.rl import RLEnvironmentCartPole, RLEnvironmentPendulum

        _registry["RLenvironmentCartPole"] = lambda x: RLEnvironmentCartPole(
            seed_env=0
        )(x)
        _registry["RLenvironmentPendulum"] = lambda x: RLEnvironmentPendulum(
            seed_env=0
        )(x)
    except ImportError:
        logger.debug("RL functions not available (gymnasium not installed)")


_populate_registry()


def get_function(name: str, **kwargs) -> Callable[[np.ndarray], float]:
    """
    Retrieve a test function by name.

    Parameters
    ----------
    name : str
        The registered name of the function (case-sensitive).

    Returns
    -------
    callable
        A callable f(x) that evaluates the function on a numpy array.

    Raises
    ------
    KeyError
        If the function name is not found in the registry.
    """
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown function '{name}'. Available functions: {available}")
    return _registry[name]


def list_functions() -> list[str]:
    """Return a sorted list of all registered function names."""
    return sorted(_registry.keys())
