"""Failure-closed verifier for the PCHP V366 loss-geometry review-lite package."""

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
import sympy as sp

import verify_reproducibility_v365 as prior


ROOT = Path(__file__).resolve().parent
TOL = 1e-12
EXCLUDED_MANIFEST_NAMES = {"manifest_v366.json", "verification_receipt_v366.json"}

EXPECTED_V366_SHA256 = {
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
    "paper_q1/CLAIM_SOURCE_AND_NOVELTY_AUDIT_V365_ZH.md": "f0c16b6549801ce2c46fc2e30f0431eebcc1465ddf311bb1796217eb98807d4b",
    "paper_q1/rccp_causal_manuscript_v366/main_en.tex": "5c237d9d1b317e8880e4502a07c42e8e1968787237f5e95d19664b3ccb70f97b",
    "paper_q1/rccp_causal_manuscript_v366/main_zh.tex": "445d9eefeee2b994be3165acbe65c8a2e50491aedfa1906130c802949388fc27",
    "paper_q1/rccp_causal_manuscript_v366/supplement_en.tex": "4e79a42a5a1041758dae326a0b29b151543960b18ff553b64a642933a18e8c69",
    "paper_q1/rccp_causal_manuscript_v366/supplement_zh.tex": "32bf8876cd9b888bbe2a522b53a1ec44f0e6dd18c55901c0b278d6b00dffc43e",
    "paper_q1/rccp_causal_manuscript_v366/references.bib": "606af7f9792b54c5fa086edc5ad1b077c88699da973441f5679a73def06b92d6",
    "paper_q1/rccp_causal_manuscript_v366/make_budget_path_figure_v363.py": "104f84c494d837ce5f0846df0cf35c87f0b4b5664939e5c836e7f5531d52d6fd",
    "paper_q1/rccp_causal_manuscript_v366/figure_budget_path_v363_report.json": "1814194adb0df3e15cc92dcc3986c5d53243627bcac5c48d8ca8b02e256827e6",
    "paper_q1/LOSS_GEOMETRY_EXTENSION_PREFREEZE_V366_ZH.md": "3e9b0494c23d10500889a27811fe8b9c29de74e60712271467e4d0c28ad0d1d0",
    "paper_q1/LOSS_GEOMETRY_CLAIM_SOURCE_AUDIT_V366_ZH.md": "4c430aca07747dc4651173c4458a69fc664d4fae4d684071fe0f97fe5dc2308a",
    "paper_q1/LOSS_GEOMETRY_EXTENSION_DECISION_V366_ZH.md": "04051a0e6a3e4f7a519c48cf08881adbbc91788beebe3b479ab7562374bf9185",
    "paper_q1/THEORY_IMPLEMENTATION_CONTRACT_V366_ZH.md": "334e221bb636088150ef009d94b524532dd154535e7052fe216f8c296a18b8fe",
    "validate_loss_geometry_v366.py": "6961dcf3d141397b07e6890d952d52e60689209a7386395d3f51e54e2557a344",
    "loss_geometry_extension_v366/loss_geometry_extension_v366_report.json": "72cbb80861c9fc456a2aa7ba857757a33f3eb3585b7b9121366f70894476c8b0",
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


def verify_manifest(audit: prior.prior.core.Audit) -> None:
    manifest = load_json("manifest_v366.json")
    audit.check(manifest["version"] == "v366", "V366 manifest version")
    audit.check(manifest["raw_third_party_archives_included"] is False, "V366 raw-data boundary")
    audit.check(manifest["contains_protocol_locked_retrospective_budget_path"] is True, "V366 budget path declared")
    audit.check(manifest["retains_original_v361_narrow_record"] is True, "V366 retains original V361 record")
    audit.check(manifest["contains_claim_source_and_novelty_audit"] is True, "V366 novelty audit declared")
    audit.check(manifest["contains_loss_geometry_extension"] is True, "V366 loss geometry declared")
    audit.check(manifest["contains_theory_implementation_contract"] is True, "V366 theory contract declared")
    audit.check(manifest["squared_loss_full_real_line_scope_only"] is True, "V366 squared-loss scope declared")
    expected_paths = {item["path"] for item in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if tracked(path)
    }
    audit.check(actual_paths == expected_paths, "V366 manifest has no untracked scientific files")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.is_file(), f"V366 manifest file exists: {item['path']}")
        audit.check(path.stat().st_size == item["bytes"], f"V366 manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"V366 manifest hash: {item['path']}")
    audit.check(manifest["tracked_files"] == len(manifest["files"]), "V366 manifest file count")


def verify_pinned_v366_artifacts(audit: prior.prior.core.Audit) -> None:
    for relative, expected in EXPECTED_V366_SHA256.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"V366 pinned artifact exists: {relative}")
        audit.check(sha256_file(path) == expected, f"V366 pinned artifact hash: {relative}")


def verify_loss_geometry(audit: prior.prior.core.Audit) -> None:
    script = ROOT / "validate_loss_geometry_v366.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit.check(completed.returncode == 0, "V366 loss-geometry validator executed")
    report_path = ROOT / "loss_geometry_extension_v366/loss_geometry_extension_v366_report.json"
    audit.check(
        sha256_file(report_path)
        == EXPECTED_V366_SHA256["loss_geometry_extension_v366/loss_geometry_extension_v366_report.json"],
        "V366 regenerated loss-geometry report hash",
    )
    report = load_json("loss_geometry_extension_v366/loss_geometry_extension_v366_report.json")
    audit.check(report["status"] == "RETAIN_FOR_V366", "V366 loss-geometry decision")
    audit.check(report["seed"] == 20260802, "V366 loss-geometry seed")
    audit.close(report["tolerance"], TOL, "V366 loss-geometry tolerance")
    audit.check(report["named_checks_passed"] == 27, "V366 loss-geometry named checks")
    audit.check(len(report["checks"]) == 27 and len(set(report["checks"])) == 27, "V366 loss-geometry check identities")
    audit.check(
        report["prefreeze_sha256"]
        == EXPECTED_V366_SHA256["paper_q1/LOSS_GEOMETRY_EXTENSION_PREFREEZE_V366_ZH.md"],
        "V366 pre-freeze identity in report",
    )
    audit.close(
        report["metric_identity"]["scalar_max_absolute_error"],
        8.881784197001252e-16,
        "V366 scalar metric identity error",
    )
    audit.close(report["metric_identity"]["euclidean_max_absolute_error"], 0.0, "V366 Euclidean metric identity error")
    bounded = report["bounded_squared_loss"]
    audit.close(
        bounded["supremum_formula_max_absolute_error"],
        1.3322676295501878e-15,
        "V366 bounded squared supremum error",
    )
    audit.close(
        bounded["active_lower_boundary_max_absolute_error"],
        8.881784197001252e-16,
        "V366 lower boundary error",
    )
    audit.close(
        bounded["active_upper_boundary_max_absolute_error"],
        1.5543122344752192e-15,
        "V366 upper boundary error",
    )
    recursive = report["recursive_squared_projection"]
    audit.check(recursive["trajectories"] == 2000, "V366 recursive trajectory count")
    audit.check(recursive["records_per_trajectory"] == 128, "V366 recursive trajectory length")
    audit.close(recursive["minimum_interval_width"], 0.0, "V366 minimum interval width")
    audit.close(recursive["maximum_output_increase"], 0.0, "V366 maximum output increase")
    audit.close(
        recursive["maximum_budget_excess"],
        7.771561172376096e-16,
        "V366 maximum squared-loss budget excess",
    )
    audit.check(
        report["unbounded_squared_loss"]["upward_update_harm"][-1] > 10000.0,
        "V366 full-real-line upward divergence",
    )
    audit.check(
        report["unbounded_squared_loss"]["downward_update_harm"][-1] > 10000.0,
        "V366 full-real-line downward divergence",
    )
    audit.check("do not prove literature priority" in report["interpretation_boundary"], "V366 interpretation firewall")

    main_en = (ROOT / "paper_q1/rccp_causal_manuscript_v366/main_en.tex").read_text(encoding="utf-8")
    main_zh = (ROOT / "paper_q1/rccp_causal_manuscript_v366/main_zh.tex").read_text(encoding="utf-8")
    contract = (ROOT / "paper_q1/THEORY_IMPLEMENTATION_CONTRACT_V366_ZH.md").read_text(encoding="utf-8")
    audit.check("full real line" in main_en and "\\R" in main_en, "V366 English full-real-line scope present")
    audit.check("整个实数轴" in main_zh and "\\R" in main_zh, "V366 Chinese full-real-line scope present")
    audit.check("任意无界结果集上的平方损失不可能性" in contract, "V366 contract forbids unbounded-set overclaim")
    audit.check("unbounded outcome space" not in main_en, "V366 English unbounded-space overclaim absent")
    audit.check("无界结果空间" not in main_zh, "V366 Chinese unbounded-space overclaim absent")


def regenerate_and_verify_figures(audit: prior.prior.core.Audit) -> None:
    prior.prior.regenerate_and_verify_figures(audit)
    script = ROOT / "paper_q1/rccp_causal_manuscript_v366/make_budget_path_figure_v363.py"
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
    figure_root = script.parent / "figures"
    for name, expected in EXPECTED_BUDGET_FIGURE_PNG_SHA256.items():
        path = figure_root / name
        audit.check(path.is_file(), f"generated V366 budget-path figure exists: {name}")
        audit.check(sha256_file(path) == expected, f"generated V366 budget-path figure hash: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = prior.prior.core.Audit()
    verify_manifest(audit)
    prior.prior.core.verify_code_and_boundary(audit)
    prior.prior.core.verify_operator(audit)
    prior.prior.core.verify_asymmetric_extension(audit)
    prior.prior.core.verify_time_varying_viability(audit)
    prior.prior.core.verify_development(audit)
    prior.prior.core.verify_candidate_control(audit)
    prior.prior.core.verify_source_group_sensitivity(audit)
    prior.prior.core.verify_nasa(audit)
    prior.prior.core.verify_falsification_and_freeze(audit)
    prior.prior.verify_pinned_new_artifacts(audit)
    prior.prior.verify_source_tuned_comparator(audit)
    prior.prior.verify_basytec_chronology(audit)
    prior.prior.verify_basytec_confirmation(audit)
    prior.verify_budget_path(audit)
    prior.verify_v362_audit(audit)
    verify_pinned_v366_artifacts(audit)
    verify_loss_geometry(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_V366_LOSS_GEOMETRY_REVIEW_LITE_VERIFICATION_PASSED",
        "named_checks_passed": len(audit.passed),
        "checks": audit.passed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sympy": sp.__version__,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "python_prefix": sys.prefix,
        },
        "boundary": (
            "The package verifies the inherited causal, comparator, BaSyTec, NASA, and retrospective "
            "budget-path evidence, plus the frozen V366 loss geometry and theory-implementation contract. "
            "The squared-loss impossibility is restricted to outcomes ranging over the full real line. "
            "The package does not establish universal priority, squared-loss empirical utility, a prospective "
            "laboratory trial, electrochemical safety, or deployment-cost calibration."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v366.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
