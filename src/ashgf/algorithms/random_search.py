"""Central-difference random search (the baseline labelled "GD" everywhere).

Despite its name, this is NOT gradient descent: each iteration samples M = d fresh
Gaussian directions xi_j ~ N(0, I_d), estimates the gradient by central differences
along them, and takes a fixed-size step. The frozen prototype calls it ``GD`` and the
thesis table lists it as the "Vanilla Gradient Descent" baseline; we keep the name for
compatibility (see AGENTS.md naming rules) while documenting what it actually is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._common import (
    BaseRunConfig,
    RunResult,
    build_context,
    central_difference_gradient,
    drive_run,
)


@dataclass(frozen=True)
class RandomSearchConfig(BaseRunConfig):
    """Configuration for :class:`RandomSearch`.

    Attributes
    ----------
    sigma:
        Smoothing radius s used in the central differences.
    lr:
        Fixed step size lambda.
    m_directions:
        Number of sampled directions per iteration (default: full dimension).
    """

    sigma: float = 1e-4
    lr: float = 1e-4
    m_directions: int | None = None


class RandomSearch:
    """Gradient-free central-difference random search (a.k.a. "GD").

    Each iteration draws an M x d matrix of iid standard-normal directions and sets

        grad_hat = sum_j (F(x + s xi_j) - F(x - s xi_j)) xi_j / (2 s M)
        x_{i+1}  = x_i - lambda grad_hat

    Cost per iteration: 2M function evaluations (+1 for the new iterate).
    """

    kind = "Vanilla Gradient Descent"

    def __init__(self, config: RandomSearchConfig) -> None:
        self.config = config

    def run(self) -> RunResult:
        cfg = self.config
        m = cfg.m_directions if cfg.m_directions is not None else cfg.dim
        name, f, rng = build_context(cfg)
        sign = 1.0 if cfg.maximize else -1.0

        def step(i: int, x: NDArray[np.float64], value: float) -> tuple[NDArray[np.float64], float]:
            directions = rng.standard_normal((m, cfg.dim))
            grad, _ = central_difference_gradient(f, x, directions, cfg.sigma)
            x_new = x + sign * cfg.lr * grad
            return x_new, float(f(x_new))

        return drive_run(cfg, name, f, rng, step)


# Public alias matching the thesis/original-code naming.
GD = RandomSearch
GDConfig = RandomSearchConfig
