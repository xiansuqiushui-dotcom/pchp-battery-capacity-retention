"""Regression tests for the validated V390 public API."""

from __future__ import annotations

import unittest

import numpy as np

from pchp_release_api_v390 import prefix_causal_cellwise_projection
from updates_v389.prefix_causal_harm_projection_v321 import (
    prefix_causal_cellwise_projection as frozen_projection,
)


class ReleaseApiV390Tests(unittest.TestCase):
    def test_valid_inputs_exactly_match_frozen_operator(self) -> None:
        identifiers = np.asarray(["A", "A", "B", "B"], dtype=object)
        cycles = np.asarray([2.0, 1.0, 1.0, 2.0])
        raw = np.asarray([0.98, 1.00, 0.95, 0.94])
        candidate = np.asarray([0.96, 0.99, 0.97, 0.92])
        expected = frozen_projection(
            identifiers, cycles, raw, candidate, 0.01, assimilation=0.2
        )
        observed = prefix_causal_cellwise_projection(
            identifiers, cycles, raw, candidate, 0.01, assimilation=0.2
        )
        np.testing.assert_array_equal(observed[0], expected[0])
        np.testing.assert_array_equal(observed[1], expected[1])

    def test_missing_and_empty_identifiers_fail_closed(self) -> None:
        cycle = np.asarray([1.0, 2.0])
        raw = np.asarray([1.0, 0.9])
        candidate = np.asarray([0.99, 0.91])
        for identifiers in (
            np.asarray(["A", None], dtype=object),
            np.asarray(["A", np.nan], dtype=object),
            np.asarray(["A", "  "], dtype=object),
        ):
            with self.subTest(identifiers=identifiers.tolist()):
                with self.assertRaises(ValueError):
                    prefix_causal_cellwise_projection(
                        identifiers, cycle, raw, candidate, 0.01
                    )


if __name__ == "__main__":
    unittest.main()

