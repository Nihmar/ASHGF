"""ASHGF-BT: ASHGF with backtracking safeguard.

Integrates the ASHGF-S safeguard mechanism directly into the main
optimisation loop, eliminating the need for a two-pass ensemble.

When a proposed step would increase the objective (for minimisation),
the smoothing bandwidth ``sigma`` is halved and the gradient is
re-estimated **at the current point** before recomputing the step.
This backtracking is attempted at most once per iteration to avoid
infinite loops on noisy or non-smooth functions.

This is structurally equivalent to ASHGF-S phase 2, but without
running a separate pure-ASHGF pass — the safeguard is always active
and becomes part of the algorithm's core behaviour.

Mathematical details
--------------------
At iteration :math:`k`, let :math:`x_k` be the current point,
:math:`g_k = \\nabla f(x_k)` the estimated gradient, and
:math:`\\alpha_k = \\sigma_k / L_{\\nabla}` the step size.

The candidate point is:

.. math::

    x_{k+1} = x_k - \\alpha_k \\cdot g_k

If :math:`f(x_{k+1}) > f(x_k)` (for minimisation), the safeguard
triggers:

.. math::

    \\sigma_k \\leftarrow \\sigma_k / 2

    g_k \\leftarrow \\nabla f(x_k) \\quad \\text{(re-estimated with new } \\sigma\\text{)}

    \\alpha_k \\leftarrow \\sigma_k / L_{\\nabla}

    x_{k+1} \\leftarrow x_k - \\alpha_k \\cdot g_k

If the retry also fails, the step is rejected (:math:`x_{k+1} = x_k`).

Parameters
----------
All parameters are inherited from :class:`ASHGF`.

Notes
-----
This variant **overrides** ``optimize`` to insert the safeguard logic
inline.  All other methods (``grad_estimator``, ``_post_iteration``,
``_update_alpha``) are inherited unchanged from :class:`ASHGF`.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGFBT"]


class ASHGFBT(ASHGF):
    """ASHGF with backtracking safeguard."""

    kind = "ASHGF-BT"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Optimize — override to add inline safeguard
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
        """Run ASHGF optimisation with inline backtracking safeguard.

        See :meth:`BaseOptimizer.optimize` for parameter documentation.
        """
        # Seed the global NumPy RNG for backward compatibility
        np.random.seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        if x_init is None:
            x = np.random.randn(dim)
        else:
            x = np.copy(x_init)

        # Pre-allocated storage
        all_values_arr: np.ndarray = np.empty(max_iter + 1)
        current_val = f(x)
        all_values_arr[0] = current_val
        x_prev: np.ndarray = x.copy()
        f_prev: float = current_val

        best_value = current_val
        best_values: list[tuple[np.ndarray, float]] = [(x.copy(), best_value)]

        # Early-stopping state
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
                        all_values_arr[i - 1],
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
                    x_new = x + step_size * grad
                else:
                    x_new = x - step_size * grad

                # Guard against NaN/inf in x
                if not np.all(np.isfinite(x_new)):
                    logger.warning("iter=%d: x contains NaN/inf — terminating", i)
                    break

                current_val = f(x_new)

                # Guard against NaN/inf in function value
                if not np.isfinite(current_val):
                    logger.warning("iter=%d: f(x) = %s — terminating", i, current_val)
                    all_values_arr[i] = current_val
                    break

                # ----------------------------------------------------------
                # SAFEGUARD: Armijo-style step-size backtracking
                # ----------------------------------------------------------
                # If the new point is significantly worse than the
                # previous one, we halve the **step size** (not sigma)
                # and retry with the same gradient direction.  This is
                # a classic backtracking line search: it reduces the
                # step until we find a point that is not (much) worse.
                #
                # KEY DESIGN CHOICE: sigma is NOT modified.  The
                # gradient is NOT re-estimated.  This isolates the
                # safeguard from ASHGF's adaptation logic, preventing
                # interference between the two mechanisms.
                #
                # A relative tolerance filters out numerical noise
                # near the optimum on well-conditioned functions.
                # ----------------------------------------------------------
                f_scale = max(abs(f_prev), abs(current_val), 1.0)
                tol = max(1e-15, 1.5e-8 * f_scale)

                is_worse = (not maximize and current_val > f_prev + tol) or (
                    maximize and current_val < f_prev - tol
                )
                # Only trigger safeguard when sigma is still relatively large.
                # If sigma has already been well-adapted (small), the gradient
                # estimate is already local and re-estimating or backtracking
                # is more likely to hurt than help.
                sigma_is_large = self._sigma > 0.1 * self.sigma_zero_ref
                if is_worse and sigma_is_large:
                    if debug:
                        logger.debug(
                            "iter=%d: safeguard triggered "
                            "(f_new=%.6e > f_prev=%.6e + tol=%.2e, "
                            "step=%.4e)",
                            i,
                            current_val,
                            f_prev,
                            tol,
                            step_size,
                        )
                    # Backtracking line search on step size only
                    # (sigma and gradient are NOT modified)
                    max_bt = 4
                    for bt in range(1, max_bt + 1):
                        step_bt = step_size / (2.0**bt)
                        if maximize:
                            x_bt = x + step_bt * grad
                        else:
                            x_bt = x - step_bt * grad

                        if not np.all(np.isfinite(x_bt)):
                            continue

                        val_bt = f(x_bt)
                        if not np.isfinite(val_bt):
                            continue

                        # Accept if not (significantly) worse than f_prev
                        improved = (not maximize and val_bt <= f_prev + tol) or (
                            maximize and val_bt >= f_prev - tol
                        )
                        if improved:
                            x_new = x_bt
                            current_val = val_bt
                            step_size = step_bt
                            if debug:
                                logger.debug(
                                    "iter=%d: safeguard backtrack %d "
                                    "accepted, step=%.4e, f=%.6e",
                                    i,
                                    bt,
                                    step_bt,
                                    val_bt,
                                )
                            break
                    else:
                        # All backtracks failed — reject the step
                        x_new = x.copy()
                        current_val = f_prev
                        if debug:
                            logger.debug(
                                "iter=%d: safeguard all backtracks "
                                "failed, step rejected",
                                i,
                            )

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

                # 4b. Check convergence (step size)
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

                # 5. Hook: post-iteration (parameter adaptation)
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
