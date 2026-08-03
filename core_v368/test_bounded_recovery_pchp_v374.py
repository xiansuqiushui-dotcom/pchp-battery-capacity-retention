"""Property and regression tests for bounded-recovery PCHP V374."""

from __future__ import annotations

import unittest

import numpy as np

from bounded_recovery_pchp_v374 import (
    causal_bounded_recovery_state,
    prefix_causal_bounded_recovery_cellwise_projection,
    prefix_causal_bounded_recovery_projection,
)
from paper_q1.rccp_reproducibility_v368.prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    prefix_causal_harm_projection,
)


class BoundedRecoveryPCHPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20_260_802)

    def test_zero_recovery_exactly_recovers_strict_pchp(self) -> None:
        for _ in range(100):
            raw = self.rng.uniform(-0.2, 1.5, 128)
            candidate = self.rng.uniform(-0.2, 1.5, 128)
            alpha = float(self.rng.choice([1.0, 0.5, 0.2, 0.05, 0.01]))
            strict_state = causal_nonincreasing_state(
                raw,
                assimilation=alpha,
            )
            strict_output = prefix_causal_harm_projection(
                strict_state,
                candidate,
                0.01,
            )
            state, output = prefix_causal_bounded_recovery_projection(
                raw,
                candidate,
                0.01,
                assimilation=alpha,
                recovery_allowance=0.0,
            )
            np.testing.assert_array_equal(state, strict_state)
            np.testing.assert_array_equal(output, strict_output)

    def test_state_and_output_obey_recovery_envelope(self) -> None:
        raw = self.rng.uniform(-0.2, 1.5, 2000)
        candidate = self.rng.uniform(-0.2, 1.5, 2000)
        recovery = self.rng.uniform(0.0, 0.005, raw.size)
        state, output = prefix_causal_bounded_recovery_projection(
            raw,
            candidate,
            0.01,
            assimilation=0.03,
            recovery_allowance=recovery,
        )
        self.assertLessEqual(float(np.max(np.diff(state) - recovery[1:])), 1e-15)
        self.assertLessEqual(float(np.max(np.diff(output) - recovery[1:])), 1e-15)
        self.assertLessEqual(float(np.max(np.abs(output - state))), 0.01 + 1e-15)
        self.assertTrue(bool(((0.0 <= output) & (output <= 1.3)).all()))

    def test_prefix_invariance(self) -> None:
        raw = self.rng.uniform(0.4, 1.2, 257)
        candidate = self.rng.uniform(0.4, 1.2, 257)
        recovery = self.rng.uniform(0.0, 0.002, raw.size)
        full_state, full_output = prefix_causal_bounded_recovery_projection(
            raw,
            candidate,
            0.01,
            assimilation=0.02,
            recovery_allowance=recovery,
        )
        for length in (1, 2, 17, 64, 128, 256):
            prefix_state, prefix_output = prefix_causal_bounded_recovery_projection(
                raw[:length],
                candidate[:length],
                0.01,
                assimilation=0.02,
                recovery_allowance=recovery[:length],
            )
            np.testing.assert_array_equal(prefix_state, full_state[:length])
            np.testing.assert_array_equal(prefix_output, full_output[:length])

    def test_every_current_interval_point_has_future_continuation(self) -> None:
        delta = 0.01
        for _ in range(1000):
            b_current = float(self.rng.uniform(0.0, 1.3))
            previous = float(self.rng.uniform(max(0.0, b_current - delta), 1.3))
            rho = float(self.rng.uniform(0.0, 0.005))
            lower = max(0.0, b_current - delta)
            upper = min(1.3, b_current + delta, previous + rho)
            q = float(self.rng.uniform(lower, upper))
            future_rho = self.rng.uniform(0.0, 0.005, 32)
            future_b = np.empty(32)
            last = b_current
            for index, allowance in enumerate(future_rho):
                lower_b = max(0.0, last - 0.05)
                upper_b = min(1.3, last + allowance)
                last = float(self.rng.uniform(lower_b, upper_b))
                future_b[index] = last
            continuation = np.maximum(0.0, future_b - delta)
            self.assertLessEqual(continuation[0] - q, future_rho[0] + 1e-15)
            self.assertTrue(
                bool(np.diff(continuation).__le__(future_rho[1:] + 1e-15).all())
            )
            self.assertTrue(
                bool((np.abs(continuation - future_b) <= delta + 1e-15).all())
            )

    def test_cellwise_processing_restores_input_order(self) -> None:
        identifiers = np.array(["b", "a", "b", "a", "b", "a"])
        cycles = np.array([2.0, 3.0, 1.0, 1.0, 3.0, 2.0])
        raw = np.array([0.91, 0.89, 0.92, 0.91, 0.90, 0.90])
        candidate = raw + np.array([0.02, -0.03, 0.01, 0.02, -0.02, 0.03])
        state, output = prefix_causal_bounded_recovery_cellwise_projection(
            identifiers,
            cycles,
            raw,
            candidate,
            0.01,
            assimilation=0.05,
            recovery_allowance=0.001,
        )
        for identifier in ("a", "b"):
            positions = np.flatnonzero(identifiers == identifier)
            order = positions[np.argsort(cycles[positions], kind="mergesort")]
            expected_state, expected_output = prefix_causal_bounded_recovery_projection(
                raw[order],
                candidate[order],
                0.01,
                assimilation=0.05,
                recovery_allowance=0.001,
            )
            np.testing.assert_array_equal(state[order], expected_state)
            np.testing.assert_array_equal(output[order], expected_output)

    def test_invalid_contracts_fail_closed(self) -> None:
        raw = np.array([0.9, 0.8])
        candidate = np.array([0.9, 0.8])
        invalid = (
            {"assimilation": 0.0, "recovery_allowance": 0.0},
            {"assimilation": 1.1, "recovery_allowance": 0.0},
            {"assimilation": 0.5, "recovery_allowance": -0.1},
            {"assimilation": 0.5, "recovery_allowance": [0.0]},
        )
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                prefix_causal_bounded_recovery_projection(
                    raw,
                    candidate,
                    0.01,
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            causal_bounded_recovery_state(
                np.array([0.9, np.nan]),
                assimilation=0.5,
                recovery_allowance=0.0,
            )


if __name__ == "__main__":
    unittest.main()
