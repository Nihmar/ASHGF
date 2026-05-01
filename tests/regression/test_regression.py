"""Regression tests: verify that algorithm performance doesn't degrade."""

import json
import os

import numpy as np
import pytest

from ashgf.algorithms import GD, SGES
from ashgf.functions import get_function

REGRESSION_DIR = os.path.dirname(__file__)
REFERENCE_FILE = os.path.join(REGRESSION_DIR, "reference_values.json")


@pytest.mark.slow
class TestRegression:
    """Regression tests comparing against stored reference values."""

    @pytest.mark.skipif(
        not os.path.exists(REFERENCE_FILE),
        reason="No reference values file found. Run tests once to generate.",
    )
    @pytest.mark.parametrize(
        "algo_name,func_name",
        [
            ("GD", "sphere"),
            ("SGES", "sphere"),
        ],
    )
    def test_against_reference(self, algo_name, func_name):
        """Compare final value against stored reference."""
        with open(REFERENCE_FILE) as f:
            ref = json.load(f)

        key = f"{algo_name}_{func_name}"
        if key not in ref:
            pytest.skip(f"No reference for {key}")

        f = get_function(func_name)
        algo_cls = {"GD": GD, "SGES": SGES}[algo_name]
        algo = algo_cls(seed=2003)
        _, all_vals = algo.optimize(f, dim=10, max_iter=100, debug=False)

        final_val = all_vals[-1]
        ref_val = ref[key]

        # Allow 10% degradation
        assert final_val <= ref_val * 1.1, (
            f"{key}: final={final_val:.6e}, ref={ref_val:.6e} "
            f"(degradation: {final_val / ref_val:.2%})"
        )


def generate_reference():
    """Utility to generate reference values (run once)."""
    ref = {}
    for algo_cls, algo_name in [(GD, "GD"), (SGES, "SGES")]:
        for func_name in ["sphere"]:
            print(f"Running {algo_name} on {func_name}...")
            f = get_function(func_name)
            algo = algo_cls(seed=2003)
            _, all_vals = algo.optimize(f, dim=10, max_iter=100, debug=False)
            ref[f"{algo_name}_{func_name}"] = float(all_vals[-1])

    with open(REFERENCE_FILE, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Reference values written to {REFERENCE_FILE}")


if __name__ == "__main__":
    generate_reference()
