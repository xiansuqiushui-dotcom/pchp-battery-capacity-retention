"""Failure-closed verifier for the PCHP V365 novelty-audited review-lite package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import verify_reproducibility_v342 as prior


ROOT = Path(__file__).resolve().parent
TOL = 1e-12
EXCLUDED_MANIFEST_NAMES = {"manifest_v365.json", "verification_receipt_v365.json"}

EXPECTED_V365_SHA256 = {
    "analyze_budget_path_candidate_control_v361.py": "bdd0a1aeaf715d25fe574feb52b1679551a608999ab49848a3ff8c6dfd701f79",
    "audit_budget_path_anchor_v362.py": "ca3e2be80c1e995daeaa90741ab6d6e07af94424a294d6b80e09f050b32640b5",
    "paper_q1/BUDGET_PATH_CANDIDATE_CONTROL_PREFREEZE_V361.json": "7bc32a42d2c20e61471357126f6281e3a11ac976979d56056b68b243f6a6c4d6",
    "paper_q1/BUDGET_PATH_V361_EXECUTION_INTERRUPTION_RECEIPT.json": "bce3a394b82b99d5beeea493cd8020b6816aa9dc90d045dca78c99d6758fb44b",
    "paper_q1/BUDGET_PATH_ANCHOR_AUDIT_PREFREEZE_V362.json": "49b986a77346f28c80c3d024bae6caff58de469124c422db726945c72400ea56",
    "budget_path_candidate_control_v361/budget_path_candidate_control_v361_report.json": "6e8d09554f73f1df7cdf6cf8affdaaf1c13eec80b6778b5e58616038ec02781b",
    "budget_path_candidate_control_v361/budget_path_summary_v361.csv": "340adf8ed60a68c748b5b03aa0fa20ede6ed068a93561bd2a8ca164d747e4240",
    "budget_path_candidate_control_v361/budget_path_domain_metrics_v361.csv": "9e9798814776c22cd0542247ceb86b3abb0e1421ee02ad9985e86c88728fe191",
    "budget_path_candidate_control_v361/budget_path_domain_auc_v361.csv": "f3c1275be42e9de2d38a4d516c3518a3777bad5a0341f4d07a2852d28693c416",
    "budget_path_candidate_control_v361/budget_path_pchp_selections_v361.csv": "338748019aad99d9e8901dc54a1f8415ee80dc1538486a5e950a873886e6f1fe",
    "budget_path_candidate_control_v361/budget_path_exact_shift_selections_v361.csv": "0d893da67bb51d71966f824adb8a15c60f9a263b3e8ad5dd156addc5697b47bc",
    "budget_path_anchor_audit_v362/budget_path_anchor_audit_v362_report.json": "8e02eca68f7016d0ec08100195689e0129cccc90fc8add7b6ecc4a275b323608",
    "budget_path_anchor_audit_v362/v361_v327_domain_anchor_comparison_v362.csv": "8a5ee5c58e1039718db8cd82ec98c2eda17047fcf12c616492f70b301f3b1f66",
    "paper_q1/rccp_causal_manuscript_v365/main_en.tex": "da84577cd0b7a2e50623ab4f95dac01330aa63334af192e02f852a4548c720c3",
    "paper_q1/rccp_causal_manuscript_v365/main_zh.tex": "2c454c39fe55650282b97d552862aa02b357304c998b848cca5f9ed6cc84672c",
    "paper_q1/rccp_causal_manuscript_v365/supplement_en.tex": "fa2cf666aebb0476bb6053f0b58a100ac8ced7147132d003d80d0adcb5f90180",
    "paper_q1/rccp_causal_manuscript_v365/supplement_zh.tex": "a66333dba9c29dcf466a72b892cc8d6e55b45ec2633ffcec94f1e3b073f805d6",
    "paper_q1/rccp_causal_manuscript_v365/make_budget_path_figure_v363.py": "104f84c494d837ce5f0846df0cf35c87f0b4b5664939e5c836e7f5531d52d6fd",
    "paper_q1/rccp_causal_manuscript_v365/figure_budget_path_v363_report.json": "1814194adb0df3e15cc92dcc3986c5d53243627bcac5c48d8ca8b02e256827e6",
    "paper_q1/CLAIM_SOURCE_AND_NOVELTY_AUDIT_V365_ZH.md": "f0c16b6549801ce2c46fc2e30f0431eebcc1465ddf311bb1796217eb98807d4b",
}

EXPECTED_BUDGET_FIGURE_PNG_SHA256 = {
    "fig4_candidate_control.png": "a22bf8c29698829ef14ee5b3e73f72bf1eae7c1373e80c12d87950cbeb377bf8",
    "fig4_candidate_control_zh.png": "a1bb2024b1fc2ab4cb97cae981f9997d1d27a5a7be78afd2711275c6c40bf9c8",
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


def verify_manifest(audit: prior.core.Audit) -> None:
    manifest = load_json("manifest_v365.json")
    audit.check(manifest["version"] == "v365", "V365 manifest version")
    audit.check(manifest["raw_third_party_archives_included"] is False, "V365 raw-data boundary")
    audit.check(manifest["contains_protocol_locked_retrospective_budget_path"] is True, "V365 budget path declared")
    audit.check(manifest["retains_original_v361_narrow_record"] is True, "V365 retains original V361 record")
    audit.check(manifest["contains_claim_source_and_novelty_audit"] is True, "V365 novelty audit declared")
    expected_paths = {item["path"] for item in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if tracked(path)
    }
    audit.check(actual_paths == expected_paths, "V365 manifest has no untracked scientific files")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.stat().st_size == item["bytes"], f"V365 manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"V365 manifest hash: {item['path']}")


def verify_pinned_v365_artifacts(audit: prior.core.Audit) -> None:
    for relative, expected in EXPECTED_V365_SHA256.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"V365 pinned artifact exists: {relative}")
        audit.check(sha256_file(path) == expected, f"V365 pinned artifact hash: {relative}")


def verify_budget_path(audit: prior.core.Audit) -> None:
    report = load_json("budget_path_candidate_control_v361/budget_path_candidate_control_v361_report.json")
    summary = pd.read_csv(ROOT / "budget_path_candidate_control_v361/budget_path_summary_v361.csv")
    auc = pd.read_csv(ROOT / "budget_path_candidate_control_v361/budget_path_domain_auc_v361.csv")
    audit.check(report["status"] == "PROTOCOL_LOCKED_RETROSPECTIVE_BUDGET_PATH_GATE_NOT_PASSED", "V361 original status retained")
    audit.check(report["decision"] == "NARROW", "V361 original NARROW retained")
    audit.check(report["anchor_checks"]["pchp_mae_matches_v327"] is False, "V361 false anchor retained")
    audit.check(report["anchor_checks"]["selected_alpha_roster_matches_v326"] is True, "V361 alpha anchor passed")
    audit.check(report["anchor_checks"]["pchp_minus_baseline_matches_v327"] is True, "V361 effect anchor passed")
    for name, value in report["gates"].items():
        if name != "all_anchor_checks":
            audit.check(value is True, f"V361 scientific gate passed: {name}")
    audit.check(all(report["structural_checks"].values()), "V361 structural checks passed")
    audit.close(report["primary"]["domain_equal_mean"], -0.0069422304526310374, "V361 primary AUC mean")
    audit.check(
        np.allclose(
            report["primary"]["domain_bootstrap_ci95"],
            [-0.011082909073005586, -0.0029569289946119034],
            rtol=0.0,
            atol=TOL,
        ),
        "V361 primary AUC bootstrap interval",
    )
    audit.check(report["primary"]["wins_ties_losses"] == [11, 0, 1], "V361 AUC domain directions")
    expected_domains = {
        "CALB", "CALCE", "HNEI", "HUST", "MATR", "MICH",
        "MICH_EXP", "RWTH", "SDU", "SNL", "UL_PUR", "XJTU",
    }
    audit.check(
        len(auc) == 12 and set(auc["domain"]) == expected_domains,
        "V361 AUC domain roster",
    )

    expected_budgets = np.array([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03])
    audit.check(np.allclose(summary["budget"], expected_budgets, rtol=0.0, atol=TOL), "V361 frozen budget grid")
    positive = summary.loc[summary["budget"] > 0.0]
    audit.check((positive["domain_equal_pchp_minus_control"] < 0.0).all(), "V361 all positive-budget means favor PCHP")
    zero = summary.loc[np.isclose(summary["budget"], 0.0, rtol=0.0, atol=TOL)].iloc[0]
    audit.close(zero["domain_equal_pchp_minus_control"], 0.0, "V361 zero-budget identity")
    anchor = summary.loc[np.isclose(summary["budget"], 0.01, rtol=0.0, atol=TOL)].iloc[0]
    audit.close(anchor["domain_equal_pchp_minus_control"], -0.004773128528170719, "V361 delta=0.01 control effect")


def verify_v362_audit(audit: prior.core.Audit) -> None:
    result = load_json("budget_path_anchor_audit_v362/budget_path_anchor_audit_v362_report.json")
    comparison = pd.read_csv(ROOT / "budget_path_anchor_audit_v362/v361_v327_domain_anchor_comparison_v362.csv")
    audit.check(result["audit_pass"] is True, "V362 independent audit passed")
    audit.check(result["scientific_outputs_recomputed"] is False, "V362 did not recompute scientific outputs")
    audit.check(result["v361_files_modified"] is False, "V362 did not modify V361 files")
    audit.check(result["status"] == "V361_ANCHOR_FAILURE_CONFIRMED_AS_JSON_PRECISION_FALSE_NEGATIVE", "V362 false-negative classification")
    audit.check(result["operational_diagnostic"]["v361_recorded_status_preserved"] is True, "V362 preserves V361 status")
    audit.check(result["operational_diagnostic"]["v361_recorded_decision_preserved"] is True, "V362 preserves V361 decision")
    audit.check(all(result["corrected_anchor_checks"].values()), "V362 authoritative anchors all pass")
    audit.check(all(result["scientific_gates_preserved"].values()), "V362 scientific gates unchanged")
    audit.check(len(comparison) == 12, "V362 domain comparison count")
    audit.check(comparison["pchp_matches"].all(), "V362 all domain PCHP metrics match")
    audit.check(comparison["baseline_matches"].all(), "V362 all domain baseline metrics match")
    audit.check(float(comparison["absolute_pchp_difference"].max()) <= 1e-15, "V362 domain PCHP numerical tolerance")


def regenerate_and_verify_figures(audit: prior.core.Audit) -> None:
    prior.regenerate_and_verify_figures(audit)
    script = ROOT / "paper_q1/rccp_causal_manuscript_v365/make_budget_path_figure_v363.py"
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
    figure_root = script.parent / "figures"
    for name, expected in EXPECTED_BUDGET_FIGURE_PNG_SHA256.items():
        path = figure_root / name
        audit.check(path.is_file(), f"generated budget-path figure exists: {name}")
        audit.check(sha256_file(path) == expected, f"generated budget-path figure hash: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = prior.core.Audit()
    verify_manifest(audit)
    prior.core.verify_code_and_boundary(audit)
    prior.core.verify_operator(audit)
    prior.core.verify_asymmetric_extension(audit)
    prior.core.verify_time_varying_viability(audit)
    prior.core.verify_development(audit)
    prior.core.verify_candidate_control(audit)
    prior.core.verify_source_group_sensitivity(audit)
    prior.core.verify_nasa(audit)
    prior.core.verify_falsification_and_freeze(audit)
    prior.verify_pinned_new_artifacts(audit)
    prior.verify_source_tuned_comparator(audit)
    prior.verify_basytec_chronology(audit)
    prior.verify_basytec_confirmation(audit)
    verify_pinned_v365_artifacts(audit)
    verify_budget_path(audit)
    verify_v362_audit(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_V365_NOVELTY_AUDITED_REVIEW_LITE_VERIFICATION_PASSED",
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
            "The package verifies the V342 evidence chain plus the protocol-locked retrospective "
            "budget path and its independent numerical-provenance audit. It retains the original "
            "V361 NARROW record, pins the V365 claim-source and novelty audit, and does not convert "
            "opened development evidence into confirmation, "
            "claim a prospective laboratory trial, or calibrate deployment costs."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v365.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
