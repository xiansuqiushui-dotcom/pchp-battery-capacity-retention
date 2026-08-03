"""Verification tests for the frozen V386 output-range audit."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

import audit_output_range_sensitivity_v386 as audit


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output_range_sensitivity_v386"


@lru_cache(maxsize=1)
def load() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    audit.main()
    report = json.loads((OUT / "output_range_sensitivity_v386_report.json").read_text(encoding="utf-8"))
    caps = pd.read_csv(OUT / "cap_summary_v386.csv")
    distributions = pd.read_csv(OUT / "saved_value_distributions_v386.csv")
    return report, caps, distributions


def test_rosters_and_reference_reproduction() -> None:
    report, _, _ = load()
    expected = {"development": (12, 586, 601932, 601932), "external": (6, 659, 9712, 9712), "basytec": (1, 45, 2969, 2969), "nasa": (1, 34, 2598, 2556)}
    for name, counts in expected.items():
        item = report["surface_reports"][name]
        assert tuple(item["counts"].values()) == counts
        assert item["cap_1.3_reproduction_maximum_error"]["state"] <= audit.TOL
        assert item["cap_1.3_reproduction_maximum_error"]["method"] <= audit.TOL


def test_grid_and_classification_are_frozen() -> None:
    report, caps, _ = load()
    assert report["classification"] == "NONBINDING"
    assert set(caps["upper_bound"]) == set(audit.CAPS)
    assert caps.groupby("surface").size().eq(len(audit.CAPS)).all()


def test_compatible_outputs_and_mae_are_invariant() -> None:
    _, caps, _ = load()
    compatible = caps[caps["compatible_target"]]
    assert (compatible["changed_outputs_vs_1.3"] == 0).all()
    assert (compatible["maximum_output_change_vs_1.3"] <= audit.TOL).all()
    spread = compatible.groupby("surface")["domain_equal_cell_macro_mae"].agg(lambda x: x.max() - x.min())
    assert (spread <= audit.TOL).all()


def test_every_replay_preserves_certificates() -> None:
    _, caps, _ = load()
    for column in ("harm_certificate_passed", "range_passed", "state_nonincreasing", "output_nonincreasing"):
        assert caps[column].all()


def test_compatible_saved_values_are_below_smallest_guard() -> None:
    _, _, distributions = load()
    compatible = distributions[distributions["surface"].isin(["development", "external", "basytec"])]
    assert (compatible["maximum"] < min(audit.CAPS)).all()


def test_nasa_is_target_scale_boundary_not_guard_selection() -> None:
    report, caps, distributions = load()
    nasa_truth = distributions[(distributions["surface"] == "nasa") & (distributions["variable"] == "truth")].iloc[0]
    assert nasa_truth["maximum"] > max(audit.CAPS)
    assert caps.loc[caps["surface"] == "nasa", "compatible_target"].eq(False).all()
    assert "do not select" in report["nasa_boundary_note"]


def test_no_saved_raw_candidate_hits_any_guard() -> None:
    _, caps, _ = load()
    assert (caps["raw_candidate_above_cap_records"] == 0).all()


def test_protocol_does_not_tune_guard() -> None:
    protocol = json.loads(audit.PROTOCOL.read_text(encoding="utf-8"))
    assert "does not tune" in protocol["selection_rule"]
    assert "NASA cannot select" in protocol["selection_rule"]
