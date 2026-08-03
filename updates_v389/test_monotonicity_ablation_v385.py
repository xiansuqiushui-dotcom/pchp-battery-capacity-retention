"""Verification tests for the V385 monotonicity-contract ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "monotonicity_ablation_v385"
REPORT = OUT / "monotonicity_ablation_v385_report.json"


class MonotonicityAblationV385Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.predictions = pd.read_parquet(
            OUT / "monotonicity_ablation_predictions_v385.parquet"
        )
        cls.summary = pd.read_csv(OUT / "monotonicity_ablation_summary_v385.csv")
        cls.transitions = pd.read_csv(
            OUT / "monotonicity_ablation_transitions_v385.csv"
        ).set_index("method")

    def test_roster(self) -> None:
        boundary = self.report["information_boundary"]
        self.assertEqual(boundary["domains"], 12)
        self.assertEqual(boundary["cells"], 586)
        self.assertEqual(boundary["records"], 601932)
        self.assertEqual(len(self.predictions), 601932)

    def test_strict_exactly_reproduces_v327(self) -> None:
        self.assertEqual(self.report["strict_v327_maximum_reproduction_error"], 0.0)

    def test_exact_sign_flip_enumeration(self) -> None:
        for comparison in self.report["comparisons"].values():
            self.assertEqual(comparison["exact_assignments"], 4096)
            self.assertEqual(sum(comparison["domain_wins_ties_losses_for_method"]), 12)

    def test_strict_output_is_nonincreasing(self) -> None:
        self.assertEqual(
            int(self.transitions.loc["strict_monotone", "positive_output_transitions"]),
            0,
        )
        self.assertLessEqual(
            float(self.transitions.loc["strict_monotone", "maximum_output_increase"]),
            1e-12,
        )

    def test_pointwise_arm_exposes_order_violations(self) -> None:
        self.assertGreater(
            int(self.transitions.loc["pointwise_tube", "positive_output_transitions"]),
            0,
        )
        self.assertGreater(
            int(self.transitions.loc["pointwise_tube", "affected_cells"]),
            0,
        )

    def test_bounded_recovery_envelope(self) -> None:
        self.assertLessEqual(
            float(
                self.transitions.loc[
                    "bounded_recovery", "maximum_recovery_envelope_excess"
                ]
            ),
            1e-12,
        )

    def test_all_arms_pass_harm_and_range_certificates(self) -> None:
        for certificate in self.report["deterministic_certificates"].values():
            self.assertTrue(certificate["harm_budget_passed"])
            self.assertTrue(certificate["range_passed"])

    def test_summary_matches_report(self) -> None:
        report = pd.DataFrame(self.report["summary"]).sort_values("method").reset_index(drop=True)
        saved = self.summary.sort_values("method").reset_index(drop=True)
        self.assertEqual(report["method"].tolist(), saved["method"].tolist())
        np.testing.assert_allclose(
            report["domain_equal_cell_macro_mae"],
            saved["domain_equal_cell_macro_mae"],
            rtol=0.0,
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
