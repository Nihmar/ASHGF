"""ASGF-2SR: 2S with restart-on-stall.

When the optimizer stalls (no improvement for *patience* iterations),
instead of stopping it perturbs the best-so-far point by a random
direction scaled by the current *sigma* and continues.  This helps
escape local minima and plateaus without adding any state that could
degrade performance during normal operation.

Up to *max_restarts* restarts are allowed (default 2).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.asgf import ASGF

logger = logging.getLogger(__name__)

__all__ = ["ASGF2SR"]


class ASGF2SR(ASGF):
    """2S step boost with restart-on-stall.

    Parameters
    ----------
    warmup : int
        Streak length for full 2x boost.  Default ``3``.
    max_restarts : int
        Maximum number of restarts allowed.  Default ``2``.
    **kwargs :
        Passed to :class:`ASGF`.
    """

    kind = "ASGF2SR"

    def __init__(
        self,
        warmup: int = 3,
        max_restarts: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._warmup = warmup
        self._max_restarts = max_restarts

        self._improve_streak: int = 0
        self._prev_f_base: float | None = None

    def _setup(self, f, dim, x):
        super()._setup(f, dim, x)
        self._improve_streak = 0
        self._prev_f_base = None

    # ------------------------------------------------------------------
    # Override optimize() to add restart logic
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
        if x_init is None:
            np.random.seed(self.seed)
            self._rng = np.random.default_rng(self.seed)
            x_init = np.random.randn(dim)

        x = x_init.astype(float, copy=True)
        f_init = f(x)

        best_values: list[tuple[np.ndarray, float]] = [(x.copy(), f_init)]
        all_values: list[float] = [f_init]
        total_iter = 0
        restarts_done = 0
        best_global = f_init
        best_global_x = x.copy()

        if debug:
            logger.info(
                "algorithm=%-6s dim=%-4d init_val=%.6e max_iter=%d%s",
                self.kind,
                dim,
                f_init,
                max_iter,
                f" patience={patience}" if patience else "",
            )

        max_iter_per_phase = max_iter // (self._max_restarts + 1)

        while total_iter < max_iter:
            n_this = min(max_iter_per_phase, max_iter - total_iter)
            p = patience if patience else n_this

            bv, av = self._run_phase(
                f, x, dim, maximize, n_this, p, ftol, debug, log_interval,
                best_global, best_global_x,
            )

            best_values.extend(bv[1:])  # skip duplicate first entry
            all_values.extend(av)
            total_iter += len(av)

            for xp, val in bv:
                if (maximize and val > best_global) or (
                    not maximize and val < best_global
                ):
                    best_global = val
                    best_global_x = xp.copy()

            if len(av) < n_this:
                last_step = np.max(np.abs(x - best_global_x))
                if last_step < self.eps:
                    break

            if restarts_done >= self._max_restarts or total_iter >= max_iter:
                break

            restarts_done += 1
            x = best_global_x + self._sigma * np.random.randn(dim)
            if debug:
                logger.info(
                    "ASGF-2SR restart %d/%d at iter=%d: sigma=%.4e best=%.6e",
                    restarts_done,
                    self._max_restarts,
                    total_iter,
                    self._sigma,
                    best_global,
                )

        return best_values, all_values

    def _run_phase(
        self,
        f,
        x,
        dim,
        maximize,
        max_iter,
        patience,
        ftol,
        debug,
        log_interval,
        best_global,
        best_global_x,
    ):
        best_value = f(x)

        best_values: list[tuple[np.ndarray, float]] = [
            (best_global_x.copy(), best_global)
        ]
        all_values_arr = np.full(max_iter + 1, np.nan)
        all_values_arr[0] = best_value

        self._setup(f, dim, x)

        f_prev = best_value
        x_prev = x.copy()
        _stall_count = 0
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

                x = self._before_gradient(x)
                grad = self.grad_estimator(x, f)

                if not np.all(np.isfinite(grad)):
                    logger.warning("gradient NaN/inf — terminating phase")
                    break

                x_new, current_val = self._compute_step(x, grad, f, maximize)

                if not np.all(np.isfinite(x_new)):
                    logger.warning("x NaN/inf — terminating phase")
                    break
                if not np.isfinite(current_val):
                    all_values_arr[i] = current_val
                    break

                all_values_arr[i] = current_val

                improved = False
                if (maximize and current_val > best_value) or (
                    not maximize and current_val < best_value
                ):
                    best_value = current_val
                    best_values.append((x_new.copy(), best_value))
                    improved = True

                if patience is not None and patience > 0:
                    if improved:
                        _stall_count = 0
                    elif ftol is not None:
                        if abs(current_val - f_prev) < ftol:
                            _stall_count += 1
                        else:
                            _stall_count = 0
                    else:
                        _stall_count += 1

                    if _stall_count >= patience:
                        if debug:
                            logger.info(
                                "Phase stalled at iter %d", i
                            )
                        break

                if i % 5 == 0:
                    max_step = float(np.max(np.abs(x_new - x_prev)))
                    if max_step < self.eps:
                        if debug:
                            logger.info(
                                "Phase converged at iteration %d", i
                            )
                        x_prev = x_new
                        f_prev = current_val
                        break

                x_prev = x_new
                f_prev = current_val
                x = x_new

                self._post_iteration(i, x, grad, f_prev)

            except Exception:
                logger.exception("Error at iteration %d", i)
                break

        return best_values, all_values_arr[:actual_iter].tolist()

    # ------------------------------------------------------------------
    # 2S step logic
    # ------------------------------------------------------------------

    def _compute_step(
        self,
        x: np.ndarray,
        grad: np.ndarray,
        f: Callable[[np.ndarray], float],
        maximize: bool,
    ) -> tuple[np.ndarray, float]:
        step_size = self._get_step_size()
        direction = grad if maximize else -grad

        x_base = x + step_size * direction
        f_base = f(x_base)
        if not np.isfinite(f_base):
            return x.copy(), f(x)

        if self._prev_f_base is not None and f_base < self._prev_f_base:
            self._improve_streak += 1
        else:
            self._improve_streak = max(0, self._improve_streak - 1)
        self._prev_f_base = f_base

        confidence = min(self._improve_streak / self._warmup, 1.0)
        k = 1.0 + confidence * 1.0

        if confidence > 0.0 and k > 1.01:
            x_big = x + k * step_size * direction
            f_big = f(x_big)
            f_cur = getattr(self, "_f_at_x", f(x))
            if np.isfinite(f_big) and f_big < f_base and f_big < f_cur:
                return x_big, f_big

        return x_base, f_base
