"""Tests for the More & Sugrue profile machinery against hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from ashgf.profiles import (
    ALPHA_GRID_DATA,
    ALPHA_GRID_PERFORMANCE,
    Trajectory,
    convergence_fe,
    data_profile,
    ideal_iterations,
    performance_profile,
    solve_cell,
)


def make_traj(values, fes, solver="s", function="f", dim=10) -> Trajectory:
    return Trajectory(
        solver=solver,
        function=function,
        dim=dim,
        values=np.asarray(values, dtype=np.float64),
        fes=np.asarray(fes, dtype=np.int64),
    )


class TestTrajectory:
    def test_alignment_required(self):
        with pytest.raises(ValueError, match="aligned"):
            make_traj([1.0, 2.0], [1])

    def test_fes_must_start_at_one_and_be_nondecreasing(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            make_traj([1.0, 2.0], [2, 1])
        with pytest.raises(ValueError, match="start at >= 1"):
            make_traj([1.0, 2.0], [0, 5])

    def test_truncate_by_budget_keeps_affordable_prefix(self):
        traj = make_traj([10.0, 8.0, 6.0, 5.0], [1, 21, 41, 61])
        cut = traj.truncate_by_budget(41)
        assert cut.values.tolist() == [10.0, 8.0, 6.0]
        assert cut.fes.tolist() == [1, 21, 41]
        # Budget below the first iterate keeps only x_0.
        assert traj.truncate_by_budget(1).values.tolist() == [10.0]


class TestConvergenceFe:
    TRAJ = make_traj([10.0, 8.0, 6.0, 5.0], [1, 21, 41, 61])

    def test_exact_threshold_hit_counts(self):
        # threshold = f_L + tau*(F(x0) - f_L); with f_L=5, tau=0.1 -> 5.5, hit at i=3.
        assert convergence_fe(self.TRAJ, 5.0, 0.1) == 61.0

    def test_zero_tolerance_needs_reaching_f_l(self):
        assert convergence_fe(self.TRAJ, 5.0, 0.0) == 61.0
        assert convergence_fe(self.TRAJ, 4.9, 0.0) == np.inf

    def test_loose_tolerance_hits_early(self):
        # tau=1 -> threshold = F(x0) = 10 -> satisfied already at x_0.
        assert convergence_fe(self.TRAJ, 5.0, 1.0) == 1.0

    def test_nonfinite_reference_is_unsolved(self):
        assert convergence_fe(self.TRAJ, np.inf, 1e-4) == np.inf

    def test_diverged_values_are_ignored(self):
        traj = make_traj([10.0, np.nan, np.inf, 5.0], [1, 21, 41, 61])
        assert convergence_fe(traj, 5.0, 0.1) == 61.0


class TestSolveCell:
    def test_best_value_and_measures(self):
        a = make_traj([10.0, 7.0, 5.0], [1, 21, 41], solver="a")
        b = make_traj([10.0, 9.0, 8.0, 5.5], [1, 21, 41, 61], solver="b")
        taus = np.array([1e-2, 1e-4])
        f_l, results = solve_cell({"a": a, "b": b}, mu=61, taus=taus)
        assert f_l == 5.0
        # Solver a reaches <= 5.0 + tau*5 at its third iterate (41 evals).
        assert results["a"][1e-2] == 41.0
        assert results["a"][1e-4] == 41.0
        # Solver b never gets within 1e-4 of 5.0 (best 5.5 > 5.05).
        assert results["b"][1e-4] == np.inf
        # Solver b never gets within 1e-4 of 5.0 (best 5.5 > 5.05), nor within 1e-2
        # (threshold 5.05 still below its best value).
        assert results["b"][1e-2] == np.inf

    def test_budget_truncation_changes_f_l(self):
        a = make_traj([10.0, 7.0, 5.0], [1, 21, 41], solver="a")
        b = make_traj([10.0, 6.0, 4.0], [1, 21, 41], solver="b")
        # With mu=21 neither sees its final value; best in budget is 6.0.
        f_l, results = solve_cell({"a": a, "b": b}, mu=21, taus=np.array([0.0]))
        assert f_l == 6.0
        assert results["a"][0.0] == np.inf
        assert results["b"][0.0] == 21.0


class TestProfiles:
    # Two problems, two solvers.
    #   p1: a needs 100, b needs 200          -> ratio a=1, b=2
    #   p2: a unsolved, b needs 50            -> both ratios NaN/unsolved for profile
    T = {
        "a": {"p1": 100.0, "p2": np.inf},
        "b": {"p1": 200.0, "p2": 50.0},
    }

    def test_performance_profile_ratios(self):
        prof = performance_profile(self.T, ["a", "b"], ["p1", "p2"])
        grid = ALPHA_GRID_PERFORMANCE
        # p1: ratios a=1, b=2; p2: only b finite (ratio 1).
        # Solver a: only p1 ever counts (p2 unsolved for a).
        assert np.all(prof["a"] == pytest.approx(0.5))
        # Solver b: p2 counts from alpha=1; p1 joins at alpha=2.
        i_alpha2 = int(np.argmax(grid >= 2.0))
        assert np.all(prof["b"][:i_alpha2] == pytest.approx(0.5))
        assert np.all(prof["b"][i_alpha2:] == pytest.approx(1.0))

    def test_data_profile_thresholds(self):
        prof = data_profile(self.T, ["a", "b"], dim=10)
        # limit(alpha) = alpha * 11. At alpha=0 nothing counts.
        assert prof["a"][0] == 0.0
        # alpha=50 -> limit 550: a solves p1 (100<=550); b solves p1 (200) and p2 (50).
        idx50 = int(np.searchsorted(ALPHA_GRID_DATA, 50))
        assert prof["a"][idx50] == pytest.approx(0.5)
        assert prof["b"][idx50] == pytest.approx(1.0)


class TestIdealIterations:
    def test_notebook_model(self):
        assert ideal_iterations(1e4, 10, "gd") == int(1e4 / 21)
        assert ideal_iterations(1e4, 10, "sges") == int(1e4 / 21)
        assert ideal_iterations(1e4, 10, "asebo") == int(1e4 / 21)
        assert ideal_iterations(1e4, 10, "asgf") == int(1e4 / 41)
        assert ideal_iterations(1e4, 10, "ashgf") == int(1e4 / 41)
