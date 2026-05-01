"""Vanilla Gradient Descent with Gaussian smoothing."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.base import BaseOptimizer
from ashgf.gradient.estimators import gaussian_smoothing
from ashgf.gradient.sampling import compute_directions

logger = logging.getLogger(__name__)

__all__ = ["GD"]


class GD(BaseOptimizer):
    """Vanilla Gradient Descent using Central Gaussian Smoothing.

    Uses a fixed learning rate and fixed sigma for gradient estimation
    via the central finite-difference Gaussian smoothing formula.

    Parameters
    ----------
    lr : float
        Fixed learning rate.
    sigma : float
        Smoothing bandwidth.
    seed : int
        Random seed for reproducibility.
    eps : float
        Convergence threshold.
    """

    kind = "GD"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        seed: int = 2003,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(seed=seed, eps=eps)
        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {lr}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.lr = lr
        self.sigma = sigma

    def _get_step_size(self) -> float:
        return self.lr

    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Estimate gradient via Gaussian smoothing with dim random directions."""
        dim = len(x)
        directions = compute_directions(dim)
        return gaussian_smoothing(x, f, self.sigma, directions)
