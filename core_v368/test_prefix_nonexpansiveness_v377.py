from __future__ import annotations

import unittest

import numpy as np

from prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    prefix_causal_harm_projection,
)


class PrefixNonexpansivenessV377Tests(unittest.TestCase):
    def test_protected_state_update_has_exact_local_proximal_form(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(5000):
            previous = float(rng.uniform(0.0, 1.3))
            raw = float(rng.uniform(-0.5, 1.8))
            alpha = float(rng.uniform(1e-6, 1.0))
            update = causal_nonincreasing_state(
                np.array([previous, raw]), assimilation=alpha
            )[1]
            unconstrained = (1.0 - alpha) * previous + alpha * raw
            proximal = np.clip(unconstrained, 0.0, previous)
            self.assertAlmostEqual(update, proximal, places=14)

    def test_protected_state_prefix_map_is_sup_norm_nonexpansive(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(1000):
            length = int(rng.integers(1, 80))
            alpha = float(rng.uniform(1e-5, 1.0))
            raw = rng.uniform(-0.5, 1.8, length)
            perturbation = rng.uniform(-0.05, 0.05, length)
            state = causal_nonincreasing_state(raw, assimilation=alpha)
            perturbed = causal_nonincreasing_state(
                raw + perturbation, assimilation=alpha
            )
            for prefix in range(1, length + 1):
                input_radius = np.max(np.abs(perturbation[:prefix]))
                output_radius = np.max(np.abs(state[:prefix] - perturbed[:prefix]))
                self.assertLessEqual(output_radius, input_radius + 2e-12)

    def test_full_pchp_prefix_map_is_sup_norm_nonexpansive(self) -> None:
        rng = np.random.default_rng(20260802)
        for _ in range(1000):
            length = int(rng.integers(1, 80))
            alpha = float(rng.uniform(1e-5, 1.0))
            budget = float(rng.uniform(0.0, 0.2))
            raw = rng.uniform(-0.5, 1.8, length)
            candidate = rng.uniform(-0.5, 1.8, length)
            raw_shift = rng.uniform(-0.05, 0.05, length)
            candidate_shift = rng.uniform(-0.05, 0.05, length)
            baseline = causal_nonincreasing_state(raw, assimilation=alpha)
            changed_baseline = causal_nonincreasing_state(
                raw + raw_shift, assimilation=alpha
            )
            projected = prefix_causal_harm_projection(
                baseline, candidate, budget
            )
            changed_projected = prefix_causal_harm_projection(
                changed_baseline, candidate + candidate_shift, budget
            )
            for prefix in range(1, length + 1):
                input_radius = max(
                    np.max(np.abs(raw_shift[:prefix])),
                    np.max(np.abs(candidate_shift[:prefix])),
                )
                output_radius = np.max(
                    np.abs(projected[:prefix] - changed_projected[:prefix])
                )
                self.assertLessEqual(output_radius, input_radius + 2e-12)

    def test_full_pchp_nonexpansive_constant_is_tight(self) -> None:
        epsilon = 0.03
        raw = np.array([0.90, 0.85, 0.80])
        candidate = raw.copy()
        baseline = causal_nonincreasing_state(raw, assimilation=0.5)
        shifted_baseline = causal_nonincreasing_state(
            raw + epsilon, assimilation=0.5
        )
        projected = prefix_causal_harm_projection(baseline, candidate, 0.1)
        shifted = prefix_causal_harm_projection(
            shifted_baseline, candidate + epsilon, 0.1
        )
        np.testing.assert_allclose(
            shifted - projected,
            np.full(raw.size, epsilon),
            atol=1e-14,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
