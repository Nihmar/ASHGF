"""ASHGF-S: ensemble of pure ASHGF and ASHGF with safeguard.

ASHGF-S runs two independent optimisation passes starting from the **same**
random state:

1. **Phase 1** – pure ASHGF (no safeguard).
2. **Phase 2** – ASHGF with a safeguard that detects uphill steps and
   reacts by halving the smoothing bandwidth ``sigma`` and re-estimating
   the gradient.

The better of the two final solutions is returned.  Because both phases
share the same initial random seed, the comparison is fair: any difference
can be attributed solely to the safeguard mechanism.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from ashgf.algorithms.ashgf import ASHGF

logger = logging.getLogger(__name__)

__all__ = ["ASHGFS"]


class ASHGFS(ASHGF):
    """ASHGF with Safeguard — ensemble meta-optimizer.

    Extends :class:`ASHGF` by adding an optional *safeguard* step: whenever
    a proposed move would increase the objective (for minimisation), the
    smoothing bandwidth ``sigma`` is halved and the gradient is re-estimated
    before accepting or rejecting the step.

    The ensemble runs **two** independent passes:

    * **Pure ASHGF** (:attr:`_safeguard_active` = ``False``)
    * **ASHGF + safeguard** (:attr:`_safeguard_active` = ``True``)

    and returns the result with the lower final best value.

    Parameters
    ----------
    **kwargs
        All keyword arguments are forwarded to :class:`ASHGF`.
    """

    kind = "ASHGF-S"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        #: If ``True``, the safeguard logic is active during the
        #: main optimisation loop.
        self._safeguard_active: bool = False

    # ------------------------------------------------------------------
    # Public API
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
        """Run the ASHGF-S ensemble.

        Two passes are executed with the **same** random seed:

        1. Pure ASHGF (no safeguard).
        2. ASHGF with safeguard enabled.

        Returns the result whose final best function value is better
        (lower for minimisation, higher for maximisation).

        See :meth:`BaseOptimizer.optimize` for parameter documentation.
        """
        # ------------------------------------------------------------------
        # Save the global NumPy RNG state so that Phase 2 starts from exactly
        # the same random sequence as Phase 1.  This guarantees a fair
        # comparison between the two passes.
        # ------------------------------------------------------------------
        rng_state = np.random.get_state()

        # ---- Phase 1: pure ASHGF ----
        self._safeguard_active = False
        if debug:
            logger.info("ASHGF-S phase 1/2: pure ASHGF (safeguard OFF)")
        best1, all1 = self._run_single(
            f=f,
            dim=dim,
            max_iter=max_iter,
            x_init=x_init,
            debug=debug,
            log_interval=log_interval,
            maximize=maximize,
            patience=patience,
            ftol=ftol,
        )

        # ---- Restore RNG state ----
        np.random.set_state(rng_state)

        # ---- Phase 2: ASHGF with safeguard ----
        self._safeguard_active = True
        if debug:
            logger.info("ASHGF-S phase 2/2: ASHGF with safeguard ON")
        best2, all2 = self._run_single(
            f=f,
            dim=dim,
            max_iter=max_iter,
            x_init=x_init,
            debug=debug,
            log_interval=log_interval,
            maximize=maximize,
            patience=patience,
            ftol=ftol,
        )

        # ---- Return the better result ----
        val1 = float(all1[-1]) if len(all1) > 0 else float("inf" if not maximize else "-inf")
        val2 = float(all2[-1]) if len(all2) > 0 else float("inf" if not maximize else "-inf")

        if maximize:
            if not np.isfinite(val1) and np.isfinite(val2): winner = 2
            elif not np.isfinite(val2) and np.isfinite(val1): winner = 1
            else: winner = 1 if val1 >= val2 else 2
        else:
            if not np.isfinite(val1) and np.isfinite(val2): winner = 2
            elif not np.isfinite(val2) and np.isfinite(val1): winner = 1
            else: winner = 1 if val1 <= val2 else 2

        if debug:
            logger.info(
                "ASHGF-S: pure=%.6e  safeguarded=%.6e  → returning phase %d",
                val1,
                val2,
                winner,
            )

        if winner == 1:
            return best1, all1
        return best2, all2

    # ------------------------------------------------------------------
    # Internal: single-pass loop (copy of BaseOptimizer.optimize
    # with the safeguard logic inserted)
    # ------------------------------------------------------------------

    def _run_single(
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
        """Core optimisation loop for a single pass.

        This is a copy of :meth:`BaseOptimizer.optimize` with one
        addition: when :attr:`_safeguard_active` is ``True`` and the
        candidate ``x_new`` is *worse* than ``x_prev``, ``sigma`` is
        halved and the gradient is re-estimated before the step is
        recomputed.

        .. note::

           The safeguard is applied **at most once** per iteration to
           avoid infinite loops when the function is noisy or
           non-smooth.
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
                # SAFEGUARD (ASHGF-S specific)
                # ----------------------------------------------------------
                # If the new point is worse than the previous one, halve
                # sigma and re-estimate the gradient.  Applied at most
                # once per iteration to prevent infinite loops.
                #
                # For minimisation: worse means current_val > f_prev.
                # For maximisation: worse means current_val < f_prev.
                # ----------------------------------------------------------
                if self._safeguard_active:
                    is_worse = (not maximize and current_val > f_prev) or (
                        maximize and current_val < f_prev
                    )
                    if is_worse:
                        if debug:
                            logger.debug(
                                "iter=%d: safeguard triggered "
                                "(f_new=%.6e > f_prev=%.6e, sigma=%.4e → %.4e)",
                                i,
                                current_val,
                                f_prev,
                                self._sigma,
                                self._sigma / 2.0,
                            )
                        # Halve sigma
                        self._sigma /= 2.0

                        # Re-estimate gradient at the *current* point x
                        # (not x_new, because we haven't accepted the step)
                        grad = self.grad_estimator(x, f)
                        if not np.all(np.isfinite(grad)):
                            logger.warning(
                                "iter=%d: gradient (after safeguard) "
                                "contains NaN/inf — terminating",
                                i,
                            )
                            break

                        # Recompute step and candidate
                        step_size = self._get_step_size()
                        if maximize:
                            x_new = x + step_size * grad
                        else:
                            x_new = x - step_size * grad

                        if not np.all(np.isfinite(x_new)):
                            logger.warning(
                                "iter=%d: x (after safeguard) "
                                "contains NaN/inf — terminating",
                                i,
                            )
                            break

                        current_val = f(x_new)
                        if not np.isfinite(current_val):
                            logger.warning(
                                "iter=%d: f(x) = %s (after safeguard) — terminating",
                                i,
                                current_val,
                            )
                            all_values_arr[i] = current_val
                            break

                        # If retry also failed, reject the step entirely
                        if not maximize and current_val > f_prev:
                            x_new = x.copy()
                            current_val = f_prev

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
