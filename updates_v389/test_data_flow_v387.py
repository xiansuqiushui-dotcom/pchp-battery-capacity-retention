"""Verification tests for V387 evidence and sample flow."""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

import audit_data_flow_v387 as audit


@lru_cache(maxsize=1)
def load() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    audit.main()
    report = json.loads((audit.OUT / "data_flow_v387_report.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(audit.OUT / "surface_sample_flow_v387.csv")
    summary = pd.read_csv(audit.OUT / "aggregate_sample_flow_v387.csv")
    return report, flow, summary


def test_twenty_unique_surfaces_and_pass() -> None:
    report, flow, _ = load()
    assert report["status"] == "PASS"
    assert len(flow) == flow["surface"].nunique() == 20


def test_development_identity_and_final_roster() -> None:
    _, flow, summary = load()
    dev = flow[flow["evidence_group"] == "development"]
    assert len(dev) == 12
    assert dev["feature_records"].sum() == 604862
    assert dev["post_reference_prediction_records"].sum() == 601932
    assert (dev["feature_records"] - dev["post_reference_prediction_records"]).sum() == 5 * 586
    item = summary[summary["evidence_group"] == "development"].iloc[0]
    assert (item["source_cells"], item["final_scored_cells"], item["final_scored_records"]) == (586, 586, 601932)


def test_six_external_cell_flow_and_final_roster() -> None:
    _, flow, summary = load()
    ext = flow[flow["evidence_group"] == "six_external"]
    assert len(ext) == 6
    assert ext["reported_cell_exclusions"].sum() == 286
    item = summary[summary["evidence_group"] == "six_external"].iloc[0]
    assert (item["source_cells"], item["final_scored_cells"], item["final_scored_records"]) == (945, 659, 9712)


def test_each_final_surface_is_nonempty_and_nested() -> None:
    _, flow, _ = load()
    assert (flow["final_scored_cells"] > 0).all()
    assert (flow["final_scored_records"] >= flow["final_scored_cells"]).all()


def test_basytec_flow() -> None:
    _, flow, _ = load()
    item = flow[flow["surface"] == "BASYTEC"].iloc[0]
    assert (item["source_cells"], item["feature_cells"], item["target_reference_cells"], item["final_scored_records"]) == (48, 47, 45, 2969)
    assert item["reported_cell_exclusions"] == 3


def test_nasa_projection_precedes_label_join() -> None:
    _, flow, _ = load()
    item = flow[flow["surface"] == "NASA"].iloc[0]
    assert (item["source_cells"], item["feature_records"], item["post_reference_prediction_records"]) == (34, 2768, 2598)
    assert (item["target_reference_cells"], item["final_scored_cells"], item["final_scored_records"]) == (33, 33, 2556)


def test_missing_raw_record_totals_are_explicit() -> None:
    report, flow, _ = load()
    missing = set(report["missing_raw_record_surfaces"])
    assert missing == {"ISU_ILCC_NMC", "STANFORD_CALENDAR", "BASYTEC"}
    rows = flow[flow["surface"].isin(missing)]
    assert rows["source_record_candidates"].isna().all()
    assert rows["raw_record_total_status"].eq("NOT_RETAINED").all()


def test_exclusion_reasons_are_never_blank() -> None:
    _, flow, _ = load()
    assert flow["exclusion_reasons"].notna().all()
    assert flow["exclusion_reasons"].str.len().gt(0).all()
