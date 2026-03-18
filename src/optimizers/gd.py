import numpy as np
import numpy.linalg as la
from typing import Optional, Union, List, Tuple

from functions import Function
from optimizers.base import BaseOptimizer


class GD(BaseOptimizer):
    """
    Evolution Strategy (ES) optimizer using Central Gaussian Smoothing.

    Corresponds to Algorithm 2 in the thesis.
    The gradient is estimated via Monte Carlo central finite differences:
        ∇Fσ(x) ≈ 1/(2σn) Σᵢ (F(x+σξᵢ) - F(x-σξᵢ)) ξᵢ
    with n = dim random directions ξᵢ ~ N(0, I).
    """

    kind = "Vanilla Gradient Descent"

    def __init__(
        self,
        lr: float = 1e-4,
        sigma: float = 1e-4,
        seed: int = 2003,
        eps: float = 1e-8,
    ):
        super().__init__(seed, eps)
        if lr < 0:
            raise ValueError("Error: learning rate < 0")
        if sigma < 0:
            raise ValueError("Error: sigma < 0")

        self.lr = lr
        self.sigma = sigma

    def optimize(
        self,
        function: str,
        dim: int = 100,
        it: int = 1000,
        x_init: Optional[Union[np.ndarray, List[float]]] = None,
        debug: bool = True,
        itprint: int = 25,
    ) -> Tuple[List, List]:
        np.random.seed(self.seed)
        f = Function(function)

        x = self._validate_x_init(x_init, dim)

        current_val = f.evaluate(x)
        best_value = current_val
        best_values = [[x.copy(), best_value]]
        all_values = [current_val]

        if debug:
            print(f"algorithm: es  function: {function}  dimension: {dim}  initial value: {current_val}")

        for i in range(1, it + 1):
            try:
                if debug and i % itprint == 0:
                    print(f"{i}th iteration - value: {current_val}  last best value: {best_value}")

                grad = self._grad_estimator_vectorized(x, f)

                x_new = x - self.lr * grad
                new_val = f.evaluate(x_new)
                all_values.append(new_val)

                if new_val < best_value:
                    best_value = new_val
                    best_values.append([x_new.copy(), best_value])

                if la.norm(x_new - x) < self.eps:
                    break

                x = x_new
                current_val = new_val

            except Exception as e:
                print("Something has gone wrong!")
                print(e)
                break

        if debug:
            print(f"\nlast evaluation: {all_values[-1]}  last_iterate: {len(all_values)-1}  best evaluation: {best_value}\n")

        return best_values, all_values

    def _grad_estimator_vectorized(self, x: np.ndarray, f: Function) -> np.ndarray:
        """
        Vectorized Central Gaussian Smoothing gradient estimator.

        Instead of looping over dim directions one at a time, we:
          1. Generate all directions as a (dim, dim) matrix in one call.
          2. Compute all perturbed points x ± σ·d via broadcasting.
          3. Evaluate all 2·dim perturbed points.
          4. Assemble the gradient with a single matrix-vector product.

        This avoids dim Python-level loop iterations and leverages NumPy's
        optimized BLAS routines for the final dot product.
        """
        dim = len(x)
        directions = np.random.randn(dim, dim)  # (n_dirs, dim)

        # Perturbed points: (dim, dim) each row is x ± sigma * d_i
        points_plus = x + self.sigma * directions   # (dim, dim)
        points_minus = x - self.sigma * directions   # (dim, dim)

        # Evaluate all perturbed points
        evals_plus = np.array([f.evaluate(points_plus[i]) for i in range(dim)])
        evals_minus = np.array([f.evaluate(points_minus[i]) for i in range(dim)])

        # Gradient: (1 / 2σn) Σ (f⁺ - f⁻) * d_i
        diffs = evals_plus - evals_minus  # (dim,)
        grad = diffs @ directions  # (dim,) @ (dim, dim) -> (dim,)
        grad /= (2 * self.sigma * dim)

        return grad
