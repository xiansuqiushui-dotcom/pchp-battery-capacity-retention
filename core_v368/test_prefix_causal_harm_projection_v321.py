from __future__ import annotations

import unittest

import numpy as np

from prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    minimum_viable_asymmetric_budget,
    prefix_causal_asymmetric_harm_projection,
    prefix_causal_cellwise_projection,
    prefix_causal_harm_projection,
    prefix_causal_time_varying_asymmetric_harm_projection,
    time_varying_asymmetric_harm_tube_bounds,
    worst_case_asymmetric_absolute_loss_increase,
)
from regret_capped_projection_v312 import worst_case_absolute_loss_regret


class PrefixCausalHarmProjectionTests(unittest.TestCase):
    def test_causal_state_is_nonincreasing(self) -> None:
        raw = np.array([1.0, 0.9, 0.95, 0.7, 0.8])
        state = causal_nonincreasing_state(raw, assimilation=0.5)
        self.assertTrue(np.all(np.diff(state) <= 0.0))
        np.testing.assert_allclose(state, [1.0, 0.95, 0.95, 0.825, 0.8125])

    def test_running_minimum_at_unit_assimilation(self) -> None:
        raw = np.array([0.95, 0.9, 0.92, 0.87])
        state = causal_nonincreasing_state(raw, assimilation=1.0)
        np.testing.assert_allclose(state, np.minimum.accumulate(raw))

    def test_prefix_invariance_of_baseline_state(self) -> None:
        prefix = np.array([1.0, 0.97, 0.99, 0.93])
        extension = np.r_[prefix, 0.2, 1.2]
        short = causal_nonincreasing_state(prefix, assimilation=0.2)
        long = causal_nonincreasing_state(extension, assimilation=0.2)
        np.testing.assert_allclose(short, long[: len(prefix)])

    def test_projection_is_nonincreasing_and_inside_tube(self) -> None:
        baseline = np.array([0.95, 0.93, 0.91, 0.88])
        candidate = np.array([1.1, 0.7, 1.2, 0.86])
        projected = prefix_causal_harm_projection(baseline, candidate, 0.01)
        self.assertTrue(np.all(np.diff(projected) <= 1e-12))
        self.assertTrue(np.all(np.abs(projected - baseline) <= 0.01 + 1e-12))

    def test_projection_is_prefix_invariant(self) -> None:
        baseline = np.array([0.95, 0.93, 0.91, 0.88])
        candidate = np.array([1.1, 0.7, 1.2, 0.86])
        short = prefix_causal_harm_projection(baseline[:3], candidate[:3], 0.02)
        long = prefix_causal_harm_projection(baseline, candidate, 0.02)
        np.testing.assert_allclose(short, long[:3])

    def test_each_step_is_the_closest_current_feasible_value(self) -> None:
        baseline = np.array([0.95, 0.93, 0.91])
        candidate = np.array([0.2, 1.2, 0.905])
        budget = 0.01
        projected = prefix_causal_harm_projection(baseline, candidate, budget)
        previous = 1.3
        for index in range(len(baseline)):
            lower = max(0.0, baseline[index] - budget)
            upper = min(1.3, baseline[index] + budget, previous)
            expected = np.clip(candidate[index], lower, upper)
            self.assertAlmostEqual(projected[index], expected)
            previous = projected[index]

    def test_exact_worst_case_regret_is_bounded(self) -> None:
        baseline = np.array([0.95, 0.93, 0.91, 0.88])
        candidate = np.array([1.1, 0.7, 1.2, 0.86])
        projected = prefix_causal_harm_projection(baseline, candidate, 0.01)
        regret = worst_case_absolute_loss_regret(baseline, projected)
        self.assertTrue(np.all(regret <= 0.01 + 1e-12))

    def test_exact_harm_region_is_necessary_and_sufficient(self) -> None:
        baseline = np.array([0.8, 0.8, 0.8, 0.8])
        budget = 0.05
        predictions = np.array([0.75, 0.85, 0.749, 0.851])
        regret = worst_case_absolute_loss_regret(baseline, predictions)
        np.testing.assert_array_equal(
            regret <= budget + 1e-12,
            np.abs(predictions - baseline) <= budget + 1e-12,
        )
        self.assertTrue(np.all(regret[:2] <= budget + 1e-12))
        self.assertTrue(np.all(regret[2:] > budget))

    def test_zero_budget_forbids_every_nontrivial_update(self) -> None:
        baseline = np.array([0.2, 0.8, -1.5, 3.0])
        changed = baseline + np.array([1e-6, -1e-6, 0.2, -0.4])
        regret = worst_case_absolute_loss_regret(baseline, changed)
        np.testing.assert_allclose(regret, np.abs(changed - baseline))
        self.assertTrue(np.all(regret > 0.0))

    def test_zero_budget_projection_returns_protected_baseline(self) -> None:
        baseline = np.array([1.0, 0.9, 0.82])
        candidate = np.array([0.2, 1.2, 0.1])
        projected = prefix_causal_harm_projection(baseline, candidate, 0.0)
        np.testing.assert_allclose(projected, baseline)

    def test_nonbinding_budget_returns_feasible_candidate(self) -> None:
        baseline = np.array([1.0, 0.9, 0.8])
        candidate = np.array([0.95, 0.85, 0.75])
        projected = prefix_causal_harm_projection(baseline, candidate, 0.1)
        np.testing.assert_allclose(projected, candidate)

    def test_rejects_nonfinite_and_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            prefix_causal_harm_projection(
                np.array([1.0, np.nan]), np.array([1.0, 0.9]), 0.01
            )
        with self.assertRaises(ValueError):
            prefix_causal_harm_projection(
                np.array([1.0, 0.9]), np.array([1.0]), 0.01
            )

    def test_recursive_interval_remains_feasible_at_lower_bound(self) -> None:
        baseline = np.array([0.02, 0.0, 0.0])
        candidate = np.array([-10.0, 10.0, -10.0])
        projected = prefix_causal_harm_projection(baseline, candidate, 0.05)
        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(np.diff(projected) <= 1e-12))

    def test_every_current_viability_point_has_a_future_continuation(self) -> None:
        rng = np.random.default_rng(20260802)
        budget = 0.04
        for _ in range(500):
            baseline = np.minimum.accumulate(rng.uniform(0.0, 1.3, 40))
            lower = np.maximum(0.0, baseline - budget)
            upper = np.minimum(1.3, baseline + budget)
            previous = float(rng.uniform(lower[0], 1.3))
            current_upper = min(float(upper[0]), previous)
            current = float(rng.uniform(lower[0], current_upper))
            witness = np.r_[current, lower[1:]]
            self.assertTrue(np.all(np.diff(witness) <= 1e-12))
            self.assertTrue(np.all((witness >= 0.0) & (witness <= 1.3)))
            self.assertTrue(np.all(np.abs(witness - baseline) <= budget + 1e-12))

    def test_every_point_outside_current_viability_interval_violates_contract(self) -> None:
        baseline = 0.82
        previous = 0.80
        budget = 0.03
        lower = max(0.0, baseline - budget)
        upper = min(1.3, baseline + budget, previous)
        outside = np.array([lower - 1e-6, upper + 1e-6])
        violates = (
            (outside < 0.0)
            | (outside > 1.3)
            | (outside > previous)
            | (np.abs(outside - baseline) > budget)
        )
        self.assertTrue(np.all(violates))

    def test_rejects_increasing_baseline(self) -> None:
        with self.assertRaises(ValueError):
            prefix_causal_harm_projection(
                np.array([0.9, 0.91]), np.array([0.9, 0.9]), 0.01
            )

    def test_rejects_invalid_assimilation(self) -> None:
        with self.assertRaises(ValueError):
            causal_nonincreasing_state(np.array([1.0]), assimilation=0.0)

    def test_cellwise_processing_restores_input_order(self) -> None:
        cells = np.array(["a", "b", "a", "b"])
        cycles = np.array([2.0, 2.0, 1.0, 1.0])
        raw = np.array([0.9, 0.8, 1.0, 0.95])
        candidate = np.array([0.85, 0.75, 1.1, 0.9])
        baseline, projected = prefix_causal_cellwise_projection(
            cells, cycles, raw, candidate, 0.01
        )
        np.testing.assert_allclose(baseline, [0.9, 0.8, 1.0, 0.95])
        self.assertLessEqual(projected[0], projected[2] + 1e-12)
        self.assertLessEqual(projected[1], projected[3] + 1e-12)

    def test_asymmetric_exact_supremum_matches_piecewise_loss(self) -> None:
        rng = np.random.default_rng(20260802)
        baseline = rng.normal(0.8, 0.4, 1000)
        prediction = rng.normal(0.8, 0.4, 1000)
        under = 3.0
        over = 7.0
        analytic = worst_case_asymmetric_absolute_loss_increase(
            baseline,
            prediction,
            underestimation_cost=under,
            overestimation_cost=over,
        )
        outcomes = np.stack(
            [
                np.minimum(baseline, prediction) - 1.0,
                baseline,
                prediction,
                np.maximum(baseline, prediction) + 1.0,
            ],
            axis=1,
        )

        def loss(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
            return under * np.maximum(truth - values, 0.0) + over * np.maximum(
                values - truth, 0.0
            )

        direct = (
            loss(prediction[:, None], outcomes) - loss(baseline[:, None], outcomes)
        ).max(axis=1)
        np.testing.assert_allclose(analytic, direct, atol=1e-12, rtol=0.0)

    def test_asymmetric_harm_region_is_necessary_and_sufficient(self) -> None:
        baseline = np.full(4, 0.8)
        budget = 0.06
        under = 3.0
        over = 6.0
        prediction = np.array(
            [
                0.8 - budget / under,
                0.8 + budget / over,
                0.8 - budget / under - 1e-4,
                0.8 + budget / over + 1e-4,
            ]
        )
        harm = worst_case_asymmetric_absolute_loss_increase(
            baseline,
            prediction,
            underestimation_cost=under,
            overestimation_cost=over,
        )
        self.assertTrue(np.all(harm[:2] <= budget + 1e-12))
        self.assertTrue(np.all(harm[2:] > budget))

    def test_equal_cost_asymmetric_projection_recovers_original_exactly(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(100):
            baseline = np.minimum.accumulate(rng.uniform(0.1, 1.2, 50))
            candidate = rng.uniform(-0.2, 1.5, 50)
            budget = float(rng.uniform(0.0, 0.1))
            original = prefix_causal_harm_projection(baseline, candidate, budget)
            generalized = prefix_causal_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=1.0,
                overestimation_cost=1.0,
            )
            np.testing.assert_array_equal(original, generalized)

    def test_asymmetric_projection_properties_and_prefix_invariance(self) -> None:
        baseline = np.array([1.0, 0.96, 0.93, 0.90, 0.84])
        candidate = np.array([1.2, 0.2, 1.1, 0.5, 0.9])
        budget = 0.04
        projected = prefix_causal_asymmetric_harm_projection(
            baseline,
            candidate,
            budget,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        harm = worst_case_asymmetric_absolute_loss_increase(
            baseline,
            projected,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        self.assertTrue(np.all(np.diff(projected) <= 1e-12))
        self.assertTrue(np.all((projected >= 0.0) & (projected <= 1.3)))
        self.assertTrue(np.all(harm <= budget + 1e-12))
        short = prefix_causal_asymmetric_harm_projection(
            baseline[:3],
            candidate[:3],
            budget,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        np.testing.assert_array_equal(short, projected[:3])

    def test_asymmetric_zero_budget_and_directional_tube(self) -> None:
        baseline = np.array([0.9, 0.85])
        candidate = np.array([1.2, 0.1])
        zero = prefix_causal_asymmetric_harm_projection(
            baseline,
            candidate,
            0.0,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        np.testing.assert_array_equal(zero, baseline)
        projected = prefix_causal_asymmetric_harm_projection(
            baseline[:1],
            np.array([2.0]),
            0.08,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        self.assertAlmostEqual(projected[0], baseline[0] + 0.01)
        projected = prefix_causal_asymmetric_harm_projection(
            baseline[:1],
            np.array([-2.0]),
            0.08,
            underestimation_cost=2.0,
            overestimation_cost=8.0,
        )
        self.assertAlmostEqual(projected[0], baseline[0] - 0.04)

    def test_asymmetric_projection_rejects_invalid_contracts(self) -> None:
        baseline = np.array([0.9, 0.8])
        candidate = np.array([0.9, 0.8])
        for under, over in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, np.inf)):
            with self.assertRaises(ValueError):
                prefix_causal_asymmetric_harm_projection(
                    baseline,
                    candidate,
                    0.01,
                    underestimation_cost=under,
                    overestimation_cost=over,
                )

    def test_time_varying_constant_schedules_recover_v340_exactly(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(100):
            baseline = np.minimum.accumulate(rng.uniform(0.1, 1.2, 80))
            candidate = rng.uniform(-0.2, 1.5, 80)
            budget = float(rng.uniform(0.0, 0.1))
            under = float(rng.uniform(0.2, 8.0))
            over = float(rng.uniform(0.2, 8.0))
            frozen = prefix_causal_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
            )
            scheduled = prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                np.full(80, budget),
                underestimation_cost=np.full(80, under),
                overestimation_cost=np.full(80, over),
            )
            np.testing.assert_array_equal(frozen, scheduled)

    def test_time_varying_nonincreasing_lower_boundary_is_sufficient(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(40):
            baseline = np.minimum.accumulate(rng.uniform(0.2, 1.2, 60))
            lower = np.maximum(
                0.0,
                np.minimum.accumulate(baseline - rng.uniform(0.0, 0.15, 60)),
            )
            under = rng.uniform(0.2, 8.0, 60)
            over = rng.uniform(0.2, 8.0, 60)
            budget = under * (baseline - lower)
            observed_lower, _ = time_varying_asymmetric_harm_tube_bounds(
                baseline,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
            )
            self.assertTrue(np.all(np.diff(observed_lower) <= 1e-12))
            for _ in range(10):
                candidate = rng.uniform(-0.2, 1.5, 60)
                projected = prefix_causal_time_varying_asymmetric_harm_projection(
                    baseline,
                    candidate,
                    budget,
                    underestimation_cost=under,
                    overestimation_cost=over,
                )
                harm = np.where(
                    projected >= baseline,
                    over * (projected - baseline),
                    under * (baseline - projected),
                )
                self.assertTrue(np.all(np.diff(projected) <= 1e-12))
                self.assertTrue(np.all(harm <= budget + 1e-12))
                prefix = prefix_causal_time_varying_asymmetric_harm_projection(
                    baseline[:17],
                    candidate[:17],
                    budget[:17],
                    underestimation_cost=under[:17],
                    overestimation_cost=over[:17],
                )
                np.testing.assert_array_equal(prefix, projected[:17])

    def test_time_varying_upward_lower_boundary_has_constructive_failure(self) -> None:
        baseline = np.array([0.9, 0.88])
        lower = np.array([0.70, 0.75])
        under = np.array([2.0, 2.0])
        over = np.array([4.0, 9.0])
        budget = under * (baseline - lower)
        observed_lower, _ = time_varying_asymmetric_harm_tube_bounds(
            baseline,
            budget,
            underestimation_cost=under,
            overestimation_cost=over,
        )
        np.testing.assert_allclose(observed_lower, lower, atol=1e-12, rtol=0.0)
        with self.assertRaises(RuntimeError):
            prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                np.array([-10.0, 0.8]),
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
            )

    def test_minimum_viable_budget_is_exact_at_realized_prefix(self) -> None:
        baseline = np.array([0.9, 0.85])
        candidate = np.array([-10.0, 1.2])
        under = np.array([2.0, 3.0])
        over = np.array([5.0, 7.0])
        budget = np.array([0.2, 0.15])
        floor = minimum_viable_asymmetric_budget(
            baseline[1:],
            np.array([0.8]),
            underestimation_cost=under[1:],
        )
        np.testing.assert_allclose(floor, [0.15], atol=1e-12, rtol=0.0)
        feasible = prefix_causal_time_varying_asymmetric_harm_projection(
            baseline,
            candidate,
            budget,
            underestimation_cost=under,
            overestimation_cost=over,
        )
        np.testing.assert_allclose(feasible, [0.8, 0.8], atol=1e-12, rtol=0.0)
        with self.assertRaises(RuntimeError):
            prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                np.array([0.2, 0.15 - 1e-6]),
                underestimation_cost=under,
                overestimation_cost=over,
            )

    def test_time_varying_upper_cost_does_not_determine_feasibility(self) -> None:
        baseline = np.array([1.0, 0.95, 0.9, 0.85])
        under = np.array([2.0, 3.0, 4.0, 5.0])
        lower = np.array([0.90, 0.85, 0.80, 0.75])
        budget = under * (baseline - lower)
        candidate = np.array([2.0, 2.0, 2.0, 2.0])
        for over in (
            np.array([0.1, 100.0, 0.2, 200.0]),
            np.array([500.0, 0.3, 700.0, 0.4]),
        ):
            projected = prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
            )
            harm = np.where(
                projected >= baseline,
                over * (projected - baseline),
                under * (baseline - projected),
            )
            self.assertTrue(np.all(np.diff(projected) <= 1e-12))
            self.assertTrue(np.all(harm <= budget + 1e-12))

    def test_time_varying_contracts_fail_closed(self) -> None:
        baseline = np.array([0.9, 0.8])
        candidate = np.array([0.9, 0.8])
        invalid = (
            (np.array([0.01]), np.ones(2), np.ones(2)),
            (np.array([0.01, -0.01]), np.ones(2), np.ones(2)),
            (np.array([0.01, np.nan]), np.ones(2), np.ones(2)),
            (np.ones(2) * 0.01, np.array([1.0, 0.0]), np.ones(2)),
            (np.ones(2) * 0.01, np.ones(2), np.array([1.0, np.inf])),
        )
        for budget, under, over in invalid:
            with self.assertRaises(ValueError):
                prefix_causal_time_varying_asymmetric_harm_projection(
                    baseline,
                    candidate,
                    budget,
                    underestimation_cost=under,
                    overestimation_cost=over,
                )


if __name__ == "__main__":
    unittest.main()
