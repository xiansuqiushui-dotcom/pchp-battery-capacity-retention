"""Verification tests for the V384 frozen external robustness audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "external_robustness_v384"
REPORT = OUT / "external_robustness_v384_report.json"


class ExternalRobustnessV384Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.domains = pd.read_csv(OUT / "external_robustness_domain_metrics_v384.csv")
        cls.cells = pd.read_csv(OUT / "external_robustness_cell_metrics_v384.csv")
        cls.loo = pd.read_csv(OUT / "external_robustness_leave_one_out_v384.csv")

    def test_frozen_roster(self) -> None:
        boundary = self.report["information_boundary"]
        self.assertEqual(boundary["datasets"], 6)
        self.assertEqual(boundary["cells"], 659)
        self.assertEqual(boundary["records"], 9712)
        self.assertEqual(len(self.cells), 659)

    def test_no_exact_duplicate_rows(self) -> None:
        predictions = pd.read_parquet(
            ROOT / "external_mechanism_decision_v380" / "external_mechanism_predictions_v380.parquet"
        )
        self.assertFalse(predictions.duplicated().any())

    def test_primary_reproduces_v380(self) -> None:
        observed = self.report["comparisons"]["fixed_shift"]["estimands"][
            "dataset_equal_cell_macro_mean_difference"
        ]
        self.assertAlmostEqual(observed, -0.00689716916991253, places=14)

    def test_exact_enumeration_and_comparators(self) -> None:
        self.assertEqual(set(self.report["comparisons"]), {"fixed_shift", "protected_state"})
        for comparison in self.report["comparisons"].values():
            self.assertEqual(comparison["exact_dataset_sign_flip"]["assignments"], 64)
            self.assertEqual(sum(comparison["dataset_wins_ties_losses"]), 6)

    def test_leave_one_out_is_complete(self) -> None:
        self.assertEqual(len(self.loo), 12)
        for comparator, group in self.loo.groupby("comparator"):
            self.assertEqual(len(group), 6, comparator)
            self.assertTrue(np.isfinite(group["dataset_equal_mean_difference"]).all())

    def test_largest_dataset_identity(self) -> None:
        for comparison in self.report["comparisons"].values():
            largest = comparison["largest_dataset_sensitivity"]
            self.assertEqual(largest["largest_by_cells"], "ISU_ILCC_NMC")
            self.assertEqual(largest["largest_by_records"], "ISU_ILCC_NMC")

    def test_domain_reaggregation(self) -> None:
        for comparator, group in self.domains.groupby("comparator"):
            observed = float(group["difference"].mean())
            reported = self.report["comparisons"][comparator]["estimands"][
                "dataset_equal_cell_macro_mean_difference"
            ]
            self.assertAlmostEqual(observed, reported, places=14)


if __name__ == "__main__":
    unittest.main()
