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
        patience: int | None = None,
        ftol: float | None = None,
    ) -> tuple[list[tuple[np.ndarray, float]], list[float]]:
        """Run the optimization loop.

        Parameters
        ----------
        f : callable
            Objective function f: R^d -> R.
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
        patience : int or None
            If set, stop early when the best function value has not
            improved for ``patience`` consecutive iterations.
            Useful to prevent oscillations after convergence.
        ftol : float or None
            If set, stop when ``|f(x_{k+1}) - f(x_k)| < ftol`` for
            ``patience`` consecutive iterations (requires ``patience``
            to also be set).

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

        # Storage — pre-allocated arrays (faster than dict[tuple])
        all_values_arr: np.ndarray = np.empty(max_iter + 1)
        current_val = f(x)
        all_values_arr[0] = current_val
        x_prev: np.ndarray = x.copy()
        f_prev: float = current_val

        best_value = current_val
        best_values: list[tuple[np.ndarray, float]] = [(x.copy(), best_value)]

        # Early-stopping state (stagnation detection)
        _stall_count: int = 0
        _best_so_far: float = current_val

        if debug:
            logger.info(
                "algorithm=%-6s dim=%-4d init_val=%.6e max_iter=%d%s",
                self.kind,
                dim,
                current_val,
                max_iter,
                f" patience={patience}" if patience else "",
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
                        f_prev,
                        best_value,
                    )

                # 1. Estimate gradient
                x = self._before_gradient(x)
                grad = self.grad_estimator(x, f)

                # Guard against NaN/inf in gradient
                if not np.all(np.isfinite(grad)):
                    logger.warning(
                        "iter=%d: gradient contains NaN/inf — terminating", i
                    )
                    break

                # 2. Update x (with hook for line-search variants)
                x_new, current_val = self._compute_step(x, grad, f, maximize)

                # Guard against NaN/inf in x
                if not np.all(np.isfinite(x_new)):
                    logger.warning("iter=%d: x contains NaN/inf — terminating", i)
                    break

                # Guard against NaN/inf in function value
                if not np.isfinite(current_val):
                    logger.warning("iter=%d: f(x) = %s — terminating", i, current_val)
                    all_values_arr[i] = current_val
                    break

                all_values_arr[i] = current_val

                # 3. Track best
                improved = False
                if (maximize and current_val > best_value) or (
                    not maximize and current_val < best_value
                ):
                    best_value = current_val
                    best_values.append((x_new.copy(), best_value))
                    improved = True

                # 4a. Stagnation detection (early stopping)
                if patience is not None and patience > 0:
                    if improved:
                        _stall_count = 0
                        _best_so_far = best_value
                    elif ftol is not None:
                        if abs(current_val - f_prev) < ftol:
                            _stall_count += 1
                        else:
                            _stall_count = 0
                    else:
                        _stall_count += 1

                    if _stall_count >= patience:
                        logger.info(
                            "Stopped at iteration %d (no improvement for %d iters)",
                            i,
                            patience,
                        )
                        break

                # 4b. Check convergence (step size) — use max norm (faster than L2)
                #     Only check every 5 iterations to reduce overhead
                if i % 5 == 0:
                    max_step = float(np.max(np.abs(x_new - x_prev)))
                    if max_step < self.eps:
                        logger.info("Converged at iteration %d (step < eps)", i)
                        x_prev = x_new
                        f_prev = current_val
                        break

                x_prev = x_new
                f_prev = current_val
                x = x_new

                # 5. Hook: post-iteration (e.g., parameter adaptation)
                self._post_iteration(i, x, grad, f_prev)

            except Exception:
                logger.exception("Error at iteration %d", i)
                break

        if debug:
            last_val = all_values_arr[actual_iter]
            logger.info(
                "final  f(x)=%.6e  iter=%d  best=%.6e  stalled=%d",
                last_val,
                actual_iter,
                best_value,
                _stall_count if patience else 0,
            )

        all_values = all_values_arr[:actual_iter].tolist()
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

    def _before_gradient(self, x: np.ndarray) -> np.ndarray:
        """Hook called before gradient estimation. May return a modified x.

        Override to implement restart-from-best-point or look-ahead.
        """
        return x

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        """Compute the next point and its function value.

        Override to implement line-search or trust-region step selection.
        Default: ``x_new = x ± step_size * grad``.
        """
        step_size = self._get_step_size()
        if maximize:
            x_new = x + step_size * grad
        else:
            x_new = x - step_size * grad
        return x_new, f(x_new)
