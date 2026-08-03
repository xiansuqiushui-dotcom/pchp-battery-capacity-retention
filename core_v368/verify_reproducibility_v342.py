"""Failure-closed verifier for the PCHP V342 review-lite package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import verify_reproducibility_v341 as core
from test_basytec_header_family_v355 import test_all_inventory_signatures


ROOT = Path(__file__).resolve().parent
TOL = 1e-12
EXCLUDED_MANIFEST_NAMES = {"manifest_v342.json", "verification_receipt_v342.json"}

EXPECTED_NEW_CODE_SHA256 = {
    "basytec_external_v343_common.py": "01ec40172587e8a246a8a19c5a268319cdd584a1ecdac442a756dabc696d7be9",
    "basytec_external_v347_common.py": "df305ea6294660f9ccd43a93fcc50f9f355b67bc96fd501f6e4294d5401b9e8b",
    "basytec_external_v352_common.py": "93276a7a7ed418cded94fc0130c9b79c163c016039e93b6f2ae437d4ca6f28e7",
    "inventory_basytec_headers_v351.py": "5b3f8cbd10fa735d6ff7692dd5dbb6a7ca11fa7c212a8dbfbca85a4fae831ea0",
    "prepare_basytec_confirmation_roster_v347.py": "010864b019c83940ec380e563a6d40db8aef64868aa1d98286bfdcfc5d18ae90",
    "build_basytec_label_blind_predictions_v344.py": "8376e54519b89cb4cc492c9f5669f65a89a32d50cbdff0f86fc26c00b8059f9a",
    "build_basytec_label_blind_predictions_v353.py": "13bd7747cf611a4b2834c0bb425b6aee98f6ffae88a600046018e0d3938aed07",
    "score_basytec_frozen_confirmation_v345.py": "e6096f69b0cd533af7c7206e4b913217a8809c874ad98244d12bd65584fb41d2",
    "score_basytec_frozen_confirmation_v354.py": "2bec2cd58d09c163fa95a34b03267d24e13b5325fbfe07fea1dfd2a968d02d84",
    "test_basytec_header_family_v355.py": "c79860f82ba924bbda021a5505e4ad204fb29a88b504fea51bf623cfd9265eb1",
    "audit_basytec_external_statistics_v357.py": "24a957bff270add6fb2b7cf1fdefd4e77b8e720992b3c52cd55a87a6e7653974",
    "analyze_basytec_excluded_condition_v358.py": "fcc40f933f33efc0c3511302bcd437cbc11973a7d253e0d962b03750ac667716",
    "analyze_source_tuned_causal_candidate_v342.py": "d436d336767d8ea831b9c7c8636dbf2955ce7d30a2165d11e49d71fe571e0d84",
    "audit_source_tuned_causal_candidate_statistics_v342.py": "7ff051b2a74ea1cd154e5a0f89c9257d92fa503d7f13fbac677a5722e5968d0a",
    "paper_q1/rccp_causal_manuscript_v342/make_basytec_figure_v359.py": "0def250de922deb3ae3293964da06b57a3c669157ccfa4028b60f5176b2e70d1",
}

EXPECTED_ARTIFACT_SHA256 = {
    "paper_q1/rccp_basytec_external_prefreeze_v352.json": "2e7c34cc3e56cbd5901d18b3989d18754d08e66e7448205197e5b6aa9658643a",
    "paper_q1/RCCP_BASYTEC_EXTERNAL_PREFREEZE_V352_ZH.md": "6f40826557240d5ef613d2cca2dfe47803c4681f0807d0aecb6a35974b0cc34e",
    "paper_q1/RCCP_BASYTEC_SCHEMA_PILOT_DECLARATION_V347_ZH.md": "27019169843c5b090325d986840c57a3b85b3dea3ae8b0bd3411a79e8552a7e4",
    "external_basytec_v343/download_receipt_v343.json": "89afa7a67a2287309a7d7c3f02c84117a642d38fbe49daee66bc848be1b693a3",
    "external_basytec_v343/label_blind_failure_receipt_v344.json": "13c031f2996af6a5ebafb9770897494e1e130eb61809acbbfc75a7e77722f881",
    "external_basytec_v347/label_blind_failure_receipt_v348.json": "3f2e06284d907ab8769bec223aea57740410a6d78e7034ac57fab4f5b55e3bca",
    "external_basytec_v347/header_inventory_v351.json": "35c4fa46ad72f00f09a17282e0ed4a2c04ec03c33c852ea621fafc746ab6a656",
    "external_basytec_v347/confirmation_roster_v347.json": "c84cb083e8bb927a4cfc955a433e46d03db8373b8ef8cf7c07616c88ed93907b",
    "external_basytec_v352/condition_map_v356.json": "b4bb5c715e3c737e3474b865d2bcdb464c9591e6d636d5896d0ca8a7aa08cb7c",
    "external_basytec_v352/label_blind_v353/basytec_label_blind_prediction_report_v353.json": "2e27780968f909b0d1d1a05b2b90370e2735e99f3a8e8926b781150751b6f8f6",
    "external_basytec_v352/scored_v354/basytec_cell_metrics_v354.csv": "463c633344c53d9cad5041091df5ac8159d6c8a5b7d29b76afeedf4ca787ae0a",
    "external_basytec_v352/scored_v354/basytec_frozen_confirmation_v354_report.json": "eb6bf718810577fe6e91228301e7352236a991d9899b3002fc2ee2d5753ec2b0",
    "external_basytec_v352/scored_v354/basytec_external_statistics_audit_v357.json": "7726061409e3a897fa2677c519a7a1af8e3ca1771fac384d8f0feed2c0682fc7",
    "external_basytec_v352/scored_v354/basytec_excluded_condition_sensitivity_v358.json": "b580e2851a94cf1bf61296edf8f3aab3db45d4c38d09b9a05b300e63067dd2e2",
    "source_tuned_causal_candidate_v342/source_tuned_causal_candidate_v342_report.json": "ed11518ea3e1eaa42148bf4421f0570d775074ae697260a8edf9fbac0798c252",
    "source_tuned_causal_candidate_v342/source_tuned_causal_candidate_statistics_audit_v342.json": "ff5eee483b1e239612913e5df78b9251a51714343d82a7a7c2a71a3d8c7d8b50",
}

EXPECTED_NEW_FIGURE_SHA256 = {
    "fig6_basytec_confirmation.png": "4523ab5ab53245990095ca123f19b1cf6eb384098e579197dc199b364e6389ea",
    "fig6_basytec_confirmation_zh.png": "4e30c03a6a2067319ea9a4378c95fa1469ad9f4e90643c4f70770efcef3cf690",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.name in EXCLUDED_MANIFEST_NAMES:
        return False
    if "__pycache__" in relative.parts or "figures" in relative.parts:
        return False
    return True


def verify_manifest(audit: core.Audit) -> None:
    manifest = load_json("manifest_v342.json")
    audit.check(manifest["version"] == "v342", "V342 manifest version")
    audit.check(manifest["raw_third_party_archives_included"] is False, "V342 raw-data boundary")
    expected_paths = {item["path"] for item in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if tracked(path)
    }
    audit.check(actual_paths == expected_paths, "V342 manifest has no untracked scientific files")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.stat().st_size == item["bytes"], f"V342 manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"V342 manifest hash: {item['path']}")


def verify_pinned_new_artifacts(audit: core.Audit) -> None:
    for relative, expected in {**EXPECTED_NEW_CODE_SHA256, **EXPECTED_ARTIFACT_SHA256}.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"V342 pinned artifact exists: {relative}")
        audit.check(sha256_file(path) == expected, f"V342 pinned artifact hash: {relative}")


def percentile_bootstrap(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(100_000, dtype=float)
    for start in range(0, len(means), 10_000):
        stop = min(start + 10_000, len(means))
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def verify_source_tuned_comparator(audit: core.Audit) -> None:
    report = load_json("source_tuned_causal_candidate_v342/source_tuned_causal_candidate_v342_report.json")
    domains = pd.read_csv(ROOT / "source_tuned_causal_candidate_v342/source_tuned_causal_candidate_domain_metrics_v342.csv")
    audit.check(report["status"] == "SOURCE_TUNED_CAUSAL_CANDIDATE_CONTROL_COMPLETED", "strong comparator status")
    audit.check(report["decision"] == "NARROW", "strong comparator narrows rather than rescues claim")
    audit.check(len(domains) == 12, "strong comparator domain count")
    accuracy = report["accuracy"]
    comparisons = accuracy["comparisons"]
    for column, key in [
        ("pchp_minus_baseline", "pchp_minus_baseline"),
        ("source_tuned_candidate_minus_baseline", "source_tuned_candidate_minus_baseline"),
        ("pchp_minus_source_tuned_candidate", "pchp_minus_source_tuned_candidate"),
    ]:
        values = domains[column].to_numpy(float)
        summary = comparisons[key]
        audit.close(float(values.mean()), summary["domain_equal_mean"], f"strong comparator mean: {key}")
    audit.close(accuracy["pchp_utility_retention_fraction"], 0.2366725296103409, "strong comparator utility retention")
    contract = report["contract_activity"]
    audit.check(contract["pchp_violating_records"] == 0, "PCHP has no strong-comparator budget violations")
    audit.check(contract["source_tuned_candidate_violating_domains"] == 12, "unprotected comparator violates every domain")
    audit.check(contract["source_tuned_candidate_violating_cells"] == 586, "unprotected comparator violates every cell")
    audit.check(contract["source_tuned_candidate_violating_records"] == 524_165, "unprotected comparator violating-record count")
    audit.check(report["gates"]["utility"] is False, "strong comparator prevents an accuracy-dominance claim")


def verify_basytec_chronology(audit: core.Audit) -> None:
    failure_1 = load_json("external_basytec_v343/label_blind_failure_receipt_v344.json")
    failure_2 = load_json("external_basytec_v347/label_blind_failure_receipt_v348.json")
    inventory = load_json("external_basytec_v347/header_inventory_v351.json")
    roster = load_json("external_basytec_v347/confirmation_roster_v347.json")
    audit.check(failure_1["status"] == "INCONCLUSIVE_SCHEMA_FAILURE_BEFORE_OUTCOME_ACCESS", "first schema failure retained")
    audit.check(failure_1["outcome_values_read"] is False, "first schema failure predates outcome access")
    audit.check(failure_2["status"] == "INCONCLUSIVE_HETEROGENEOUS_HEADER_FAMILY_BEFORE_OUTCOME_ACCESS", "second schema failure retained")
    audit.check(failure_2["confirmation_capacity_values_accessed"] is False, "second schema failure predates capacity access")
    audit.check(inventory["status"] == "47_CELL_HEADER_ONLY_INVENTORY_COMPLETE", "header-only inventory status")
    audit.check(inventory["data_rows_read"] == 0, "header-only inventory reads no data rows")
    audit.check(inventory["capacity_values_accessed"] is False, "header-only inventory excludes capacity values")
    audit.check(len(inventory["records"]) == 47, "header-only inventory cell count")
    audit.check(roster["status"] == "BASYTEC_V347_CONFIRMATION_ROSTER_FROZEN", "confirmation roster frozen")
    audit.check(roster["development_cell_excluded"] == "F0001.zip", "schema-pilot cell excluded")
    test_all_inventory_signatures()
    audit.check(True, "four BaSyTec header signatures exclude outcome sentinel")


def verify_basytec_confirmation(audit: core.Audit) -> None:
    blind = load_json("external_basytec_v352/label_blind_v353/basytec_label_blind_prediction_report_v353.json")
    audit.check(blind["status"] == "BASYTEC_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_CAPACITY_ACCESS", "BaSyTec prediction-freeze status")
    audit.check(blind["capacity_column_values_accessed"] is False, "BaSyTec predictions exclude capacity column values")
    audit.check(blind["external_outcome_values_accessed"] == [], "BaSyTec predictions exclude external outcomes")
    audit.check(blind["prediction_cells"] == 47 and blind["prediction_cycles"] == 3101, "BaSyTec frozen prediction dimensions")
    audit.check(blind["certificate"]["label_blind_structural_certificate_passed"] is True, "BaSyTec label-blind structural certificate")

    report = load_json("external_basytec_v352/scored_v354/basytec_frozen_confirmation_v354_report.json")
    cells = pd.read_csv(ROOT / "external_basytec_v352/scored_v354/basytec_cell_metrics_v354.csv")
    differences = cells["pchp_minus_baseline"].to_numpy(float)
    primary = report["accuracy"]["comparisons"]["pchp_minus_baseline"]
    audit.check(report["status"] == "BASYTEC_FROZEN_EXTERNAL_CONFIRMATION_COMPLETED", "BaSyTec scoring status")
    audit.check(report["decision"] == "RETAIN", "BaSyTec frozen gate decision")
    audit.check(len(cells) == report["design"]["eligible_physical_cells"] == 45, "BaSyTec eligible cell count")
    audit.check(int(cells["records"].sum()) == report["design"]["scored_cycle_records"] == 2969, "BaSyTec scored-cycle count")
    audit.close(float(differences.mean()), primary["cell_equal_mean"], "BaSyTec cell-equal mean effect")
    wins = int((differences < -TOL).sum())
    ties = int((np.abs(differences) <= TOL).sum())
    losses = int((differences > TOL).sum())
    audit.check([wins, ties, losses] == primary["wins_ties_losses"] == [45, 0, 0], "BaSyTec cell wins ties losses")
    interval = percentile_bootstrap(differences, 20260808)
    audit.check(np.allclose(interval, primary["cell_bootstrap_ci95"], rtol=0.0, atol=TOL), "BaSyTec paired-cell bootstrap")
    audit.check(all(report["gates"].values()), "all frozen BaSyTec gates")
    contract = report["contract"]
    audit.check(contract["maximum_pchp_displacement"] <= 0.01 + TOL, "BaSyTec PCHP displacement budget")
    audit.check(contract["maximum_observed_pchp_loss_regret"] <= 0.01 + TOL, "BaSyTec observed-loss budget")
    audit.check(contract["source_tuned_candidate_violating_cells"] == 45, "BaSyTec unprotected comparator violates every eligible cell")
    audit.close(report["accuracy"]["pchp_utility_retention_fraction"], 0.08703838437344463, "BaSyTec utility retention")

    statistical = load_json("external_basytec_v352/scored_v354/basytec_external_statistics_audit_v357.json")
    audit.check(statistical["status"] == "BASYTEC_V354_INDEPENDENT_STATISTICAL_AUDIT_COMPLETE", "BaSyTec independent audit status")
    audit.check(statistical["reconstruction"]["all_reconstructed_within_1e-12"] is True, "BaSyTec aggregate reconstruction")
    condition = statistical["posthoc_condition_cluster_sensitivity"]
    audit.check(condition["conditions"] == 23, "BaSyTec represented-condition count")
    audit.check(condition["comparisons"]["pchp_minus_baseline"]["wins_ties_losses"] == [23, 0, 0], "BaSyTec condition sensitivity direction")
    diagnostic = statistical["reviewer_risk_diagnostics"]
    audit.close(diagnostic["fraction_baseline_predictions_below_truth"], 1.0, "BaSyTec baseline underprediction diagnostic")
    audit.check(diagnostic["cell_equal_pchp_minus_constant_safe_up"] > 0.0, "BaSyTec fixed-shift adaptivity boundary")

    sensitivity = load_json("external_basytec_v352/scored_v354/basytec_excluded_condition_sensitivity_v358.json")
    audit.check(sensitivity["status"] == "POSTHOC_FULL_47_CELL_SENSITIVITY_COMPLETE", "BaSyTec full-roster sensitivity status")
    audit.check(sensitivity["full_47_cell_analysis"]["pchp_minus_baseline"]["wins_ties_losses"] == [47, 0, 0], "BaSyTec full-roster sensitivity direction")
    audit.check("does not replace" in sensitivity["warning"], "BaSyTec sensitivity remains explicitly post-hoc")


def regenerate_and_verify_figures(audit: core.Audit) -> None:
    core.regenerate_and_verify_figures(audit)
    script = ROOT / "paper_q1/rccp_causal_manuscript_v342/make_basytec_figure_v359.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    figure_root = script.parent / "figures"
    for name, expected in EXPECTED_NEW_FIGURE_SHA256.items():
        path = figure_root / name
        audit.check(path.is_file(), f"generated BaSyTec figure exists: {name}")
        audit.check(sha256_file(path) == expected, f"generated BaSyTec figure hash: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = core.Audit()
    verify_manifest(audit)
    core.verify_code_and_boundary(audit)
    core.verify_operator(audit)
    core.verify_asymmetric_extension(audit)
    core.verify_time_varying_viability(audit)
    core.verify_development(audit)
    core.verify_candidate_control(audit)
    core.verify_source_group_sensitivity(audit)
    core.verify_nasa(audit)
    core.verify_falsification_and_freeze(audit)
    verify_pinned_new_artifacts(audit)
    verify_source_tuned_comparator(audit)
    verify_basytec_chronology(audit)
    verify_basytec_confirmation(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_V342_REVIEW_LITE_VERIFICATION_PASSED",
        "named_checks_passed": len(audit.passed),
        "checks": audit.passed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "python_prefix": sys.prefix,
        },
        "boundary": (
            "The package verifies aggregate development, NASA, strong-comparator, and "
            "outcome-isolated BaSyTec evidence. It does not claim a prospective laboratory "
            "trial, team-wide nonexposure, adaptive identification in every external domain, "
            "or empirical deployment-cost calibration."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v342.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
