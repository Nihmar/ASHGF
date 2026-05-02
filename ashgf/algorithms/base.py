"""Base class for all optimization algorithms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import numpy.linalg as la

logger = logging.getLogger(__name__)

__all__ = ["BaseOptimizer"]


class BaseOptimizer(ABC):
    """Abstract base class for derivative-free optimizers.

    Implements the Template Method pattern: subclasses provide
    ``grad_estimator`` and optionally override ``_post_iteration``.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    eps : float
        Convergence threshold on step size ‖x_{k+1} - x_k‖.
    """

    kind: str = "Base"

    def __init__(self, seed: int = 2003, eps: float = 1e-8) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.seed = seed
        self.eps = eps
        self._rng: np.random.Generator | None = None

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        dim: int = 100,
        max_iter: int = 1000,
        x_init: np.ndarray | None = None,
        debug: bool = True,
        log_interval: int = 25,
        maximize: bool = False,
    ) -> tuple[list[tuple[np.ndarray, float]], list[float]]:
        """Run the optimization loop.

        Parameters
        ----------
        f : callable
            Objective function f: R^d → R.
        dim : int
            Problem dimension.
        max_iter : int
            Maximum number of iterations.
        x_init : np.ndarray or None
            Initial point. If None, use N(0, I) random vector.
        debug : bool
            If True, emit log messages during optimization.
        log_interval : int
            Print/log progress every ``log_interval`` iterations.
        maximize : bool
            If True, maximize f instead of minimizing (for RL).

        Returns
        -------
        best_values : list of (x, f(x))
            Sequence of best points found.
        all_values : list of float
            Sequence of function values at each iteration.
        """
        # Seed the global NumPy RNG for backward compatibility
        np.random.seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = np.copy(x_init)

        # Storage
        steps: dict[int, tuple[np.ndarray, float]] = {}
        current_val = f(x)
        steps[0] = (x, current_val)

        best_value = current_val
        best_values: list[tuple[np.ndarray, float]] = [(x.copy(), best_value)]

        if debug:
            logger.info(
                "algorithm=%-6s dim=%-4d init_val=%.6e max_iter=%d",
                self.kind,
                dim,
                current_val,
                max_iter,
            )

        # ---- Hook: pre-iteration setup ----
        self._setup(f, dim, x)

        actual_iter = 0
        for i in range(1, max_iter + 1):
            actual_iter = i
            try:
                if debug and i % log_interval == 0:
                    logger.info(
                        "iter=%5d  f(x)=%.6e  best=%.6e",
                        i,
                        steps[i - 1][1],
                        best_value,
                    )

                # 1. Estimate gradient
                grad = self.grad_estimator(x, f)

                # Guard against NaN/inf in gradient
                if not np.all(np.isfinite(grad)):
                    logger.warning(
                        "iter=%d: gradient contains NaN/inf — terminating", i
                    )
                    break

                # 2. Update x
                step_size = self._get_step_size()
                if maximize:
                    x = x + step_size * grad
                else:
                    x = x - step_size * grad

                # Guard against NaN/inf in x
                if not np.all(np.isfinite(x)):
                    logger.warning("iter=%d: x contains NaN/inf — terminating", i)
                    break

                current_val = f(x)

                # Guard against NaN/inf in function value
                if not np.isfinite(current_val):
                    logger.warning("iter=%d: f(x) = %s — terminating", i, current_val)
                    steps[i] = (x.copy(), current_val)
                    break

                steps[i] = (x.copy(), current_val)

                # 3. Track best
                if (maximize and current_val > best_value) or (
                    not maximize and current_val < best_value
                ):
                    best_value = current_val
                    best_values.append((x.copy(), best_value))

                # 4. Check convergence
                if la.norm(x - steps[i - 1][0]) < self.eps:
                    logger.info("Converged at iteration %d (step < eps)", i)
                    break

                # 5. Hook: post-iteration (e.g., parameter adaptation)
                self._post_iteration(i, x, grad, steps[i - 1][1])

            except Exception:
                logger.exception("Error at iteration %d", i)
                break

        if debug:
            last_val = steps[actual_iter][1]
            logger.info(
                "final  f(x)=%.6e  iter=%d  best=%.6e",
                last_val,
                actual_iter,
                best_value,
            )

        all_values = [steps[j][1] for j in range(actual_iter)]
        return best_values, all_values

    # ------------------------------------------------------------------
    # Abstract / overridable methods
    # ------------------------------------------------------------------

    @abstractmethod
    def grad_estimator(
        self, x: np.ndarray, f: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Estimate the gradient ∇f(x)."""
        ...

    def _get_step_size(self) -> float:
        """Return the step size (learning rate) for the current iteration.

        Override in subclasses that use adaptive learning rates.
        """
        return 1.0

    def _setup(self, f: Callable[[np.ndarray], float], dim: int, x: np.ndarray) -> None:
        """Hook called once before the main loop. Override for init logic."""
        pass

    def _post_iteration(
        self, iteration: int, x: np.ndarray, grad: np.ndarray, f_val: float
    ) -> None:
        """Hook called after each iteration. Override for adaptive logic."""
        pass
