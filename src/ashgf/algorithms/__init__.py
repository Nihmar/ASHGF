"""Optimization algorithms.

Public API (names aligned with the thesis tables and the frozen prototype):

- :class:`GD` / :class:`RandomSearch`: central-difference random search baseline.
- :class:`SGES`: stochastic gradient estimation with subspace exploitation.
- :class:`ASEBO`: adaptive subspace-exploration Bayesian optimization.
- :class:`ASGF`: adaptive stochastic gradient-free optimizer.
- :class:`ASHGF`: adaptive stochastic *historical* gradient-free optimizer.

Every algorithm exposes ``run() -> RunResult`` and is fully deterministic given its
configuration seed.
"""

from ._adaptive import AdaptiveSmoothingState
from ._common import BaseRunConfig, HistoryBuffer, RunResult
from .asebo import ASEBO, ASEBOConfig
from .asgf import ASGF, ASGFConfig
from .ashgf import ASHGF, ASHGFConfig
from .random_search import GD, GDConfig, RandomSearch, RandomSearchConfig
from .sges import SGES, SGESConfig

__all__ = [
    "AdaptiveSmoothingState",
    "BaseRunConfig",
    "HistoryBuffer",
    "RunResult",
    "ASEBO",
    "ASEBOConfig",
    "ASGF",
    "ASGFConfig",
    "ASHGF",
    "ASHGFConfig",
    "GD",
    "GDConfig",
    "RandomSearch",
    "RandomSearchConfig",
    "SGES",
    "SGESConfig",
]
