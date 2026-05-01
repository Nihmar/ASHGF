"""Integration tests for optimization algorithms."""

import numpy as np
import pytest

from ashgf.algorithms import ASEBO, ASGF, ASHGF, GD, SGES
from ashgf.functions import get_function

ALGORITHMS = [GD, SGES, ASHGF]
# ASGF and ASEBO may take longer
ALGORITHMS_SLOW = [ASGF, ASEBO]


class TestOptimizers:
    """Integration tests for all optimizers."""

    @pytest.mark.parametrize("algo_cls", ALGORITHMS)
    def test_convergence_on_sphere(self, algo_cls):
        """All algorithms should converge on the sphere function."""
        f = get_function("sphere")
        dim = 5

        algo = algo_cls(seed=42)
        best_vals, all_vals = algo.optimize(f, dim=dim, max_iter=500, debug=False)

        # Final value should be better than initial
        assert all_vals[-1] < all_vals[0], f"{algo_cls.kind}: no improvement"

        # Should make some progress (threshold > 1.0 means any improvement)
        improvement = all_vals[0] / max(all_vals[-1], 1e-15)
        assert improvement > 1.0, (
            f"{algo_cls.kind}: no improvement ({improvement:.4f}x)"
        )

    @pytest.mark.parametrize("algo_cls", ALGORITHMS)
    def test_deterministic_with_seed(self, algo_cls):
        """Two runs with same seed should give identical results."""
        f = get_function("sphere")
        dim = 5

        algo1 = algo_cls(seed=42)
        _, vals1 = algo1.optimize(f, dim=dim, max_iter=50, debug=False)

        algo2 = algo_cls(seed=42)
        _, vals2 = algo2.optimize(f, dim=dim, max_iter=50, debug=False)

        np.testing.assert_array_equal(vals1, vals2)

    @pytest.mark.parametrize("algo_cls", ALGORITHMS)
    def test_best_value_monotonic(self, algo_cls):
        """Best values should be non-increasing (for minimization)."""
        f = get_function("sphere")
        dim = 5

        algo = algo_cls(seed=42)
        best_vals, _ = algo.optimize(f, dim=dim, max_iter=100, debug=False)

        best_only = [v for _, v in best_vals]
        for i in range(1, len(best_only)):
            assert best_only[i] <= best_only[i - 1], (
                f"{algo_cls.kind}: best value increased at step {i}"
            )

    @pytest.mark.parametrize("algo_cls", ALGORITHMS)
    def test_x_init_respected(self, algo_cls):
        """The algorithm should start from the provided x_init."""
        f = get_function("sphere")
        dim = 5
        x_init = np.ones(dim)

        algo = algo_cls(seed=42)
        best_vals, _ = algo.optimize(
            f, dim=dim, max_iter=10, x_init=x_init, debug=False
        )

        # First best point should be x_init
        np.testing.assert_array_equal(best_vals[0][0], x_init)

    def test_gd_learning_rate(self):
        """GD with lr=0 should raise ValueError."""
        with pytest.raises(ValueError):
            GD(lr=0.0, seed=42)
