"""Reconstruct the manuscript evidence and sample flow from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "DATA_FLOW_PROTOCOL_V387.json"
OUT = ROOT / "data_flow_v387"

INPUT_HASHES = {
    "batterylife_early_charge_soh_v109_results.json": "ff7814e7bbd9f68c5c317dff0eea829a94743c7889d6d85f5474cc0cb6f145ca",
    "batterylife_external_hust_rwth_v124_results.json": "caad74f6c9bb9fa34dbcaf828706ba6ba972ccb3ee5189274abef0d681b5f5b2",
    "batterylife_sdu_early_charge_soh_v145_results.json": "dc3f53164a94f7a69eb0365a49696abeb9c1bc9b1e5ac7f34e50818ada9a36e0",
    "batterylife_matr_early_charge_soh_v151_results.json": "fbda77d752a5a50a4c320a038aa806adc37a3dfe89340cc1899dd662266430dd",
    "batterylife_early_charge_soh_v109.parquet": "89ce9711f91b0ba7d4a14db7766055a76cce1223a553baff270abac08fed6c8e",
    "batterylife_external_hust_rwth_v124.parquet": "2765e6ee85b7ed4000140b55396e40d271cfc9ecf464663d391a607964cdd9d8",
    "batterylife_sdu_early_charge_soh_v145.parquet": "8db4f121dad4afb151327e85facfd6da1346f2b56409a9b487e7aaa345e33613",
    "batterylife_matr_early_charge_soh_v151.parquet": "00395c408410efa6d72c7e5936964993388739ec52ced6693fc4f3ff4b544542",
    "external_confirmation_v237/ampere_a123_lfp_early_charge_soh_v243_report.json": "e50b822a7932be3e4628c587f02c6b4e2cb2c6ac43f3710194d024dbb9023729",
    "external_confirmation_v237/imperial_m50t_early_charge_soh_v249_report.json": "f7a9c538a0420160e44096d784095c19bb26d046533814f935af36a3a42e0c4f",
    "external_luh_v255/luh_nmc_sio_early_charge_soh_v256_report.json": "3a59b0d3fafff6e4129a095664d2e9d382e765de4cc33c5b1fa1feb88b688958",
    "external_isu_ilcc_v270/isu_ilcc_early_charge_soh_v271_report.json": "8e7bc55a558b5379d43c7c49ef49d96d05d63e08eb1a0023c7b52a0571804a1e",
    "external_stanford_calendar_v280/stanford_calendar_early_charge_soh_v281_report.json": "628221cce131c56194ceb7afd5e46f68b5edf8791785d851c99730b53549fd58",
    "external_multistage_50e_v290/multistage_50e_early_charge_soh_v291_report.json": "6ee13a9f3903392cd909dceb8fa0e03ce314a1caf51299dd3567073ddb23f2b8",
    "external_basytec_v352/label_blind_v353/basytec_label_blind_prediction_report_v353.json": "2e27780968f909b0d1d1a05b2b90370e2735e99f3a8e8926b781150751b6f8f6",
    "external_basytec_v352/scored_v354/basytec_frozen_confirmation_v354_report.json": "eb6bf718810577fe6e91228301e7352236a991d9899b3002fc2ee2d5753ec2b0",
    "external_nasa_v329/label_blind_v330/nasa_label_blind_prediction_report_v330.json": "3d961418c8aab4e4e71afa2e723b3aff0832aa62ac9bf517829db8f64b3999c8",
    "external_nasa_v329/scored_v331/nasa_frozen_confirmation_v331_report.json": "d92be40ab5c34b6324667af549cfde34b15857d02e8c0c654579111854de48b1",
    "bounded_recovery_pchp_v374/bounded_recovery_outer_predictions_v374.parquet": "c7c0045abfa8fdaa862604a273fe1da8f4409e0741148a74c6220f2212c11bdb",
    "external_mechanism_decision_v380/external_mechanism_predictions_v380.parquet": "474d01b7473ed30aaa79f117ab1a9237e5028201ecab74dbce37932d3ac9f5e3"
}

DEV_INPUTS = [
    ("batterylife_early_charge_soh_v109_results.json", "batterylife_early_charge_soh_v109.parquet"),
    ("batterylife_external_hust_rwth_v124_results.json", "batterylife_external_hust_rwth_v124.parquet"),
    ("batterylife_sdu_early_charge_soh_v145_results.json", "batterylife_sdu_early_charge_soh_v145.parquet"),
    ("batterylife_matr_early_charge_soh_v151_results.json", "batterylife_matr_early_charge_soh_v151.parquet"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate_inputs() -> None:
    for relative, expected in INPUT_HASHES.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen data-flow input mismatch: {relative}")


def row(**values: object) -> dict[str, object]:
    template = {
        "evidence_group": None, "surface": None,
        "source_cells": None, "source_record_candidates": None,
        "feature_cells": None, "feature_records": None,
        "feature_reference_cells": None, "post_reference_prediction_records": None,
        "target_reference_cells": None, "final_scored_cells": None, "final_scored_records": None,
        "reported_cell_exclusions": 0, "reported_record_exclusions": 0,
        "raw_record_total_status": "REPORTED", "exclusion_reasons": "none",
    }
    template.update(values)
    return template


def development_rows(final: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for report_name, parquet_name in DEV_INPUTS:
        report = load_json(report_name)
        frame = pd.read_parquet(ROOT / parquet_name)
        per_report = {item["domain"]: item for item in report["per_domain"]}
        for domain, block in frame.groupby("domain", sort=True):
            item = per_report[str(domain)]
            cells = int(block["cell_id"].nunique())
            parsed = int(len(block))
            invalid = int(item["invalid_rows"])
            reference = block.loc[block["aligned_cycle_rank"] <= 5].groupby("cell_id")["aligned_cycle_rank"].nunique()
            if len(reference) != cells or not (reference == 5).all():
                raise RuntimeError(f"development reference roster mismatch: {domain}")
            post = int(block["after_initial5_reference_window"].astype(bool).sum())
            frozen = final.loc[final["domain"] == domain]
            reasons = "five commissioning-reference records per cell withheld from scoring"
            if invalid:
                reasons = f"{invalid} structurally invalid cycle records; " + reasons
            records.append(row(
                evidence_group="development", surface=str(domain),
                source_cells=cells, source_record_candidates=parsed + invalid,
                feature_cells=cells, feature_records=parsed,
                feature_reference_cells=cells, post_reference_prediction_records=post,
                target_reference_cells=cells,
                final_scored_cells=int(frozen["cell_id"].nunique()), final_scored_records=int(len(frozen)),
                reported_record_exclusions=invalid, exclusion_reasons=reasons,
            ))
    return records


def external_rows(final: pd.DataFrame) -> list[dict[str, object]]:
    a = load_json("external_confirmation_v237/ampere_a123_lfp_early_charge_soh_v243_report.json")
    a_counts = {"candidates": 0, "valid": 0, "excluded": 0}
    for archive in a["archives"]:
        counts = archive["counts"]
        a_counts["candidates"] += int(counts["characterization_members"])
        a_counts["valid"] += int(counts["valid_rows"])
        a_counts["excluded"] += int(counts.get("member_exclusion:characterization has fewer than three Step 13 rows", 0))

    i = load_json("external_confirmation_v237/imperial_m50t_early_charge_soh_v249_report.json")
    i_candidates = sum(int(x["counts"]["summary_rows"]) for x in i["archives"])
    i_missing = sum(int(x["counts"].get("missing_charge_member", 0)) for x in i["archives"])
    i_valid = sum(int(x["counts"]["valid_rows"]) for x in i["archives"])

    l = load_json("external_luh_v255/luh_nmc_sio_early_charge_soh_v256_report.json")
    l_invalid = int(l["eoc_counts"]["checkup_without_charge_eoc"]) + int(l["eoc_counts"]["checkup_without_valid_discharge_capacity"])
    l_valid = int(l["eoc_counts"]["valid_paired_checkups"])
    isu = load_json("external_isu_ilcc_v270/isu_ilcc_early_charge_soh_v271_report.json")
    st = load_json("external_stanford_calendar_v280/stanford_calendar_early_charge_soh_v281_report.json")
    ms = load_json("external_multistage_50e_v290/multistage_50e_early_charge_soh_v291_report.json")

    specs = [
        dict(surface="AMPERE_A123_LFP", source_cells=20, source_record_candidates=a_counts["candidates"], feature_cells=20, feature_records=a_counts["valid"], feature_reference_cells=20, post_reference_prediction_records=743, target_reference_cells=20, reported_record_exclusions=a_counts["excluded"], exclusion_reasons="4 characterization members had fewer than three Step-13 rows; first five reference records per cell withheld"),
        dict(surface="IMPERIAL_M50T", source_cells=40, source_record_candidates=i_candidates, feature_cells=40, feature_records=i_valid, feature_reference_cells=40, post_reference_prediction_records=269, target_reference_cells=40, reported_record_exclusions=i_missing, exclusion_reasons="42 summary rows lacked a matched charge member; first five reference records per cell withheld"),
        dict(surface="LUH_NMC_SIO", source_cells=int(l["cfg_cells"]), source_record_candidates=l_valid + l_invalid, feature_cells=int(l["cfg_cells"]), feature_records=l_valid, feature_reference_cells=int(l["retained_cells"]), post_reference_prediction_records=int(l["retained_rows"]), target_reference_cells=int(l["retained_cells"]), reported_cell_exclusions=int(l["cfg_cells"])-int(l["retained_cells"]), reported_record_exclusions=l_invalid, exclusion_reasons="48 cells had fewer than six valid capacity checkups; 1 checkup lacked charge EOC and 5 lacked valid discharge capacity"),
        dict(surface="ISU_ILCC_NMC", source_cells=int(isu["published_roster_cells"]), source_record_candidates=None, feature_cells=int(isu["published_roster_cells"]), feature_records=None, feature_reference_cells=int(isu["retained_cells"]), post_reference_prediction_records=int(isu["counts"]["valid_post_reference_rows"]), target_reference_cells=int(isu["retained_cells"]), reported_cell_exclusions=int(isu["counts"]["cell_fewer_than_six_valid_capacities"]), raw_record_total_status="NOT_RETAINED", exclusion_reasons="7 cells had fewer than six valid capacities; upstream report did not retain a comparable raw record total"),
        dict(surface="STANFORD_CALENDAR", source_cells=int(st["published_roster_cells"]), source_record_candidates=None, feature_cells=int(st["archive_raw_members"] if "archive_raw_members" in st else st["counts"]["archive_raw_members"]), feature_records=None, feature_reference_cells=int(st["retained_cells"]), post_reference_prediction_records=int(st["counts"]["valid_post_reference_rows"]), target_reference_cells=int(st["retained_cells"]), reported_cell_exclusions=int(st["counts"]["cells_without_six_valid_diagnostics"]), reported_record_exclusions=int(st["counts"]["diagnostic_exclusion:charge_window_gap_exceeds_120s"])+int(st["counts"]["diagnostic_exclusion:charge_window_not_covered"])+int(st["counts"]["diagnostic_exclusion:diagnostic_fewer_than_seven_cycles"]), raw_record_total_status="NOT_RETAINED", exclusion_reasons="229 cells lacked six valid diagnostics; 9,060 diagnostic exclusions from window gaps, incomplete coverage, or too few cycles; raw diagnostic total not retained"),
        dict(surface="MULTISTAGE_50E", source_cells=int(ms["published_cycle_roster_cells"]), source_record_candidates=int(ms["counts"]["eligible_rpt_members"]), feature_cells=int(ms["published_cycle_roster_cells"]), feature_records=int(ms["counts"]["eligible_rpt_members"])-int(ms["counts"]["rpt_exclusion:missing_capacity_charge_or_discharge"]), feature_reference_cells=int(ms["retained_cells"]), post_reference_prediction_records=int(ms["counts"]["valid_post_reference_rows"]), target_reference_cells=int(ms["retained_cells"]), reported_cell_exclusions=int(ms["counts"]["cells_without_six_valid_rpts"]), reported_record_exclusions=int(ms["counts"]["rpt_exclusion:missing_capacity_charge_or_discharge"]), exclusion_reasons="2 cells lacked six valid RPTs; 3 RPTs missed capacity, charge, or discharge fields"),
    ]
    output=[]
    for spec in specs:
        frozen=final.loc[final["domain"]==spec["surface"]]
        spec.update(evidence_group="six_external", final_scored_cells=int(frozen["cell_id"].nunique()), final_scored_records=int(len(frozen)))
        output.append(row(**spec))
    return output


def boundary_rows() -> list[dict[str, object]]:
    bp = load_json("external_basytec_v352/label_blind_v353/basytec_label_blind_prediction_report_v353.json")
    bs = load_json("external_basytec_v352/scored_v354/basytec_frozen_confirmation_v354_report.json")
    npred = load_json("external_nasa_v329/label_blind_v330/nasa_label_blind_prediction_report_v330.json")
    nscore = load_json("external_nasa_v329/scored_v331/nasa_frozen_confirmation_v331_report.json")
    nasa_cycles = sum(int(x["cycle_records"]) for x in npred["structural_audit"]["cell_structural_audit"])
    return [
        row(evidence_group="boundary", surface="BASYTEC", source_cells=48, source_record_candidates=None,
            feature_cells=int(bp["cells_with_structural_features"]), feature_records=int(bp["feature_cycles"]),
            feature_reference_cells=int(bp["prediction_cells"]), post_reference_prediction_records=int(bp["prediction_cycles"]),
            target_reference_cells=int(bs["design"]["eligible_physical_cells"]), final_scored_cells=int(bs["design"]["eligible_physical_cells"]), final_scored_records=int(bs["design"]["scored_cycle_records"]),
            reported_cell_exclusions=3, reported_record_exclusions=int(bp["prediction_cycles"])-int(bs["design"]["scored_cycle_records"]), raw_record_total_status="NOT_RETAINED",
            exclusion_reasons="F0001 reserved for parser development; F0009 and F0048 outside the frozen reference-capacity range; upstream raw row total not retained"),
        row(evidence_group="boundary", surface="NASA", source_cells=int(npred["discovered_physical_cells"]), source_record_candidates=nasa_cycles,
            feature_cells=int(npred["structurally_valid_feature_cells"]), feature_records=int(npred["structurally_valid_feature_rows"]),
            feature_reference_cells=int(npred["prediction_cells"]), post_reference_prediction_records=int(npred["prediction_rows"]),
            target_reference_cells=int(nscore["label_audit"]["cells_with_valid_initial5_reference"]), final_scored_cells=int(nscore["independent_cells"]), final_scored_records=int(nscore["scored_prediction_rows"]),
            reported_cell_exclusions=1, reported_record_exclusions=nasa_cycles-int(npred["structurally_valid_feature_rows"])+int(npred["prediction_rows"])-int(nscore["scored_prediction_rows"]),
            exclusion_reasons="5,117 cycle records lacked a valid fixed charge window; 170 commissioning-reference records withheld; one predicted cell lacked a valid discharge-based initial-five reference, leaving 42 unmatched prediction records"),
    ]


def main() -> None:
    validate_inputs()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dev_final = pd.read_parquet(ROOT / "bounded_recovery_pchp_v374" / "bounded_recovery_outer_predictions_v374.parquet")
    ext_final = pd.read_parquet(ROOT / "external_mechanism_decision_v380" / "external_mechanism_predictions_v380.parquet")
    flow = pd.DataFrame(development_rows(dev_final) + external_rows(ext_final) + boundary_rows())
    if len(flow) != 20 or flow["surface"].nunique() != 20:
        raise RuntimeError("unexpected data-flow surface roster")
    summary = flow.groupby("evidence_group", as_index=False).agg(
        surfaces=("surface", "nunique"), source_cells=("source_cells", "sum"),
        feature_reference_cells=("feature_reference_cells", "sum"),
        target_reference_cells=("target_reference_cells", "sum"),
        final_scored_cells=("final_scored_cells", "sum"), final_scored_records=("final_scored_records", "sum"),
    )
    dev = summary.loc[summary["evidence_group"] == "development"].iloc[0]
    ext = summary.loc[summary["evidence_group"] == "six_external"].iloc[0]
    expected = protocol["expected_final_rosters"]
    if (int(dev["surfaces"]), int(dev["source_cells"]), int(dev["final_scored_cells"]), int(dev["final_scored_records"])) != (12, 586, 586, 601932):
        raise RuntimeError("development aggregate flow mismatch")
    if (int(ext["surfaces"]), int(ext["source_cells"]), int(ext["final_scored_cells"]), int(ext["final_scored_records"])) != (6, 945, 659, 9712):
        raise RuntimeError("external aggregate flow mismatch")
    report = {
        "protocol_version": protocol["protocol_version"], "status": "PASS",
        "surface_rows": int(len(flow)), "missing_raw_record_surfaces": flow.loc[flow["raw_record_total_status"] != "REPORTED", "surface"].tolist(),
        "development_identity": "604,862 contract-admissible records minus five commissioning records for each of 586 cells equals 601,932 scored records.",
        "external_identity": "945 published/source cells minus 286 structurally or reference-ineligible cells equals 659 scored cells.",
        "interpretation": protocol["interpretation"],
        "artifacts": {"protocol_sha256": sha256_file(PROTOCOL)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    flow.to_csv(OUT / "surface_sample_flow_v387.csv", index=False)
    summary.to_csv(OUT / "aggregate_sample_flow_v387.csv", index=False)
    (OUT / "data_flow_v387_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
