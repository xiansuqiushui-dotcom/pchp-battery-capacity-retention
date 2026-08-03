"""Explain and preflight every published V380--V387 execution entry point.

The release intentionally separates direct replay from provider-dependent full
recomputation.  This launcher never reports an omitted workflow as runnable and
never downloads or substitutes third-party data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UPDATES = ROOT / "updates_v389"
CORE = ROOT / "core_v368"

WORKFLOWS: dict[str, dict[str, object]] = {
    "audit_external_robustness_v384.py": {
        "mode": "direct_replay",
        "required": ["external_mechanism_decision_v380/external_mechanism_predictions_v380.parquet"],
    },
    "audit_external_threshold_decisions_v383.py": {
        "mode": "direct_replay",
        "required": ["external_mechanism_decision_v380/external_mechanism_predictions_v380.parquet"],
    },
    "audit_monotonicity_ablation_v385.py": {
        "mode": "provider_dependent_full_recomputation",
        "required": [
            "bounded_recovery_pchp_v374/bounded_recovery_outer_predictions_v374.parquet",
            "nested_prefix_causal_outer_v327/nested_outer_predictions_v327.parquet",
        ],
    },
    "audit_output_range_sensitivity_v386.py": {
        "mode": "provider_dependent_full_recomputation",
        "required": [
            "bounded_recovery_pchp_v374/bounded_recovery_outer_predictions_v374.parquet",
            "external_basytec_v352/scored_v354/basytec_scored_records_v354.parquet",
            "external_nasa_v329/label_blind_v330/nasa_frozen_predictions_v330.parquet",
            "external_nasa_v329/scored_v331/nasa_scored_predictions_v331.parquet",
        ],
    },
    "audit_data_flow_v387.py": {
        "mode": "provider_dependent_full_recomputation",
        "required": [
            "batterylife_early_charge_soh_v109_results.json",
            "batterylife_early_charge_soh_v109.parquet",
            "bounded_recovery_pchp_v374/bounded_recovery_outer_predictions_v374.parquet",
        ],
    },
    "audit_feature_weighting_pchp_factorial_v383.py": {
        "mode": "provider_dependent_refit",
        "required": [
            "batterylife_early_charge_soh_v109.parquet",
            "method_ablation_v315/ablation_predictions_v315.parquet",
            "prefix_causal_rccp_v322/prefix_causal_predictions_v322.parquet",
            "nested_prefix_causal_outer_v327/nested_outer_predictions_v327.parquet",
        ],
    },
    "audit_pchp_sequence_transformer_v383.py": {
        "mode": "historical_archival_refit",
        "required": [
            "evaluate_postconfirmatory_sequence_baselines_v161.py",
            "evaluate_postconfirmatory_transformer_baselines_v181.py",
            "batterylife_early_charge_soh_v109.parquet",
            "nested_prefix_causal_outer_v327/nested_outer_predictions_v327.parquet",
        ],
        "packages": ["torch"],
    },
    "run_pchp_external_mechanism_decision_v380.py": {
        "mode": "provider_dependent_refit",
        "required": [
            "batterylife_early_charge_soh_v109.parquet",
            "prefix_causal_rccp_v322/prefix_causal_predictions_v322.parquet",
            "external_confirmation_v237/ampere_a123_lfp_early_charge_soh_v243.parquet",
            "external_confirmation_v237/imperial_m50t_early_charge_soh_v249.parquet",
            "external_luh_v255/luh_nmc_sio_early_charge_soh_v256.parquet",
            "external_isu_ilcc_v270/isu_ilcc_early_charge_soh_v271.parquet",
            "external_stanford_calendar_v280/stanford_calendar_early_charge_soh_v281.parquet",
            "external_multistage_50e_v290/multistage_50e_early_charge_soh_v291.parquet",
        ],
    },
}


def readiness(script: str) -> dict[str, object]:
    specification = WORKFLOWS[script]
    missing = [
        relative
        for relative in specification.get("required", [])
        if not (UPDATES / str(relative)).is_file()
    ]
    missing_packages = [
        package
        for package in specification.get("packages", [])
        if importlib.util.find_spec(str(package)) is None
    ]
    return {
        "script": script,
        "mode": specification["mode"],
        "available_in_release": not missing and not missing_packages,
        "missing_release_inputs": missing,
        "missing_optional_packages": missing_packages,
    }


def check_package() -> None:
    actual = {
        path.name
        for path in UPDATES.glob("*.py")
        if path.name.startswith(("audit_", "run_"))
    }
    expected = set(WORKFLOWS)
    if actual != expected:
        raise RuntimeError(
            f"unclassified execution entry points: missing={sorted(actual - expected)}, "
            f"stale={sorted(expected - actual)}"
        )
    matrix = (ROOT / "REPLAY_READINESS_V390.md").read_text(encoding="utf-8")
    undocumented = [name for name in sorted(expected) if name not in matrix]
    if undocumented:
        raise RuntimeError(f"execution entries missing from replay matrix: {undocumented}")
    print("PASS: all V380--V387 execution entry points are explicitly classified")


def run_workflow(script: str) -> int:
    report = readiness(script)
    if not report["available_in_release"]:
        print(json.dumps(report, indent=2))
        print(
            "This workflow is not directly runnable from the review-lite release. "
            "Acquire the provider-authorized inputs and reconstruct their frozen "
            "paths and hashes as described in core_v368/FULL_REPRODUCTION.md.",
            file=sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(CORE), str(UPDATES), environment.get("PYTHONPATH", "")]
    )
    return subprocess.call(
        [sys.executable, script], cwd=UPDATES, env=environment
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-package", action="store_true")
    parser.add_argument("--check", choices=sorted(WORKFLOWS))
    parser.add_argument("--run", choices=sorted(WORKFLOWS))
    args = parser.parse_args()
    if args.check_package:
        check_package()
        return 0
    if args.check:
        print(json.dumps(readiness(args.check), indent=2))
        return 0
    if args.run:
        return run_workflow(args.run)
    print(json.dumps([readiness(name) for name in sorted(WORKFLOWS)], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

