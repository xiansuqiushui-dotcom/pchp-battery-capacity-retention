"""Failure-closed verifier for the PCHP review-lite reproducibility package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    minimum_viable_asymmetric_budget,
    prefix_causal_asymmetric_harm_projection,
    prefix_causal_harm_projection,
    prefix_causal_time_varying_asymmetric_harm_projection,
    time_varying_asymmetric_harm_tube_bounds,
    worst_case_asymmetric_absolute_loss_increase,
)


ROOT = Path(__file__).resolve().parent
TOL = 1e-12
EXPECTED_CODE_SHA256 = {
    "analyze_method_ablation_v315.py": "60bef7d6449c4b668e63e8dbac1b4f8fe828e3c5b6161138a784eafda46f3f5d",
    "analyze_source_group_sensitivity_v339.py": "0317361f88eda976540bceab3dfd01e80a76dc0c9d310efc381f53aa555a23e9",
    "build_batterylife_early_charge_soh_v109.py": "d32278311ac4938d06bcf156d7591384e4ef7152aa9bb23eb279062aa7bc3be3",
    "build_matr_early_charge_soh_v151.py": "fc1c3540d5f8851a34a6490553b9464b94534014a52979c27cc61977148d5511",
    "build_nasa_label_blind_predictions_v330.py": "305d623f711758ad1b4be0b518f302d468b6d4bae17f5642ea09f4ce00d11b5c",
    "build_nested_source_only_alpha_selections_v326.py": "21a95fb2be153da29c861b988eda709d172275adecb04533ec88aadafc6491ca",
    "build_sdu_early_charge_soh_v145.py": "19e8004bf45ec3a68afb72d23c3b81d782097de89a1d54d20bfb380ce02a817b",
    "develop_anchor_invariant_soh_v306.py": "e84fe83f4dc71389bf3d6081aaa903a5ca22c0fdb1b4bbf5c7c6e6b6acab5109",
    "develop_context_change_soh_v310.py": "45a3c45ef7bf3c157de565a59f099bc09aa34c06185cc6d618f16cdf7313f88a",
    "evaluate_candidate_information_control_v333.py": "0f6523700082cff38aa26cd05385d3cf3252d436ccdbc22821355e7029861873",
    "evaluate_monotone_regret_capped_soh_v313.py": "a31eddd3f07c21add848fed10a4896da2747f03bcf8ccb953e0e65606715ac36",
    "evaluate_regret_capped_context_change_v312.py": "28b5056eb1aec834da29eee48cea721fcadd23bb27af58c7c0e0d5e61f7360de",
    "freeze_external_causal_source_spec_v328.py": "06a9fd58beb23f3a4481962b65a09ac57a0bda0d7d9ebe529e29805ffee49806",
    "freeze_rccp_source_models_v316.py": "62715f8492e0f8cdfed9a5aa3f5f8822d1b955c57024ab523ba9dc8aa6ca541b",
    "pilot_batterylife_early_charge_lodo_v110.py": "0ded169bc4c4db43f5c040d2570396504d9e82bfbc0fa47a7c004de53e7a3f72",
    "prefix_causal_harm_projection_v321.py": "ce7288a129c17114e1ca57432c6417beba7938d58db2b1fd0a87171c479eb54c",
    "test_prefix_causal_harm_projection_v321.py": "7d21846c6f23670cfeb4ddba9cc4035b3f7b85e13914f148f239794cb75df1ea",
    "validate_asymmetric_harm_extension_v340.py": "d1d944449d167cbf04b0d22111e1f413d2f5dd67424ea449905c92994dd3f397",
    "validate_time_varying_viability_v341.py": "9d40e71875cff1933623e149bb0a1ba8de4b454143de91030a6c01c83811a0d6",
    "regret_capped_projection_v312.py": "815e5397412870bc6f93246bd4c0814377708d36b924467877fdc18e23712b6d",
    "score_nasa_frozen_confirmation_v331.py": "9894c1cbecae80e8a0712b812257ab474974ee3a3ea8df418e6306f94bdf40ba",
    "score_nested_prefix_causal_outer_v327.py": "ec5c0a94b4a50d05af2e0fabdf01a715cfb3735ce77e80876d8c280e89d7f135",
    "download_batterylife_external_v123.py": "ffd013a07325f0ab4c6c9616cafe750ce2b5e493bfe8a7db90ec59179720a329",
    "download_batterylife_zenodo_v109.py": "fe45a81c6a23a6f64c8bbf011f348e60c6445229e33a4fdbc326e663b63e1ae7",
    "paper_q1/rccp_causal_manuscript_v335/make_figures_v335.py": "414fa73bb8eb01b3d49bc6fd7bb9c66185264287bd958d0313e1a25484baef17",
}
FORBIDDEN_SUFFIXES = {".zip", ".mat", ".h5", ".hdf5", ".xlsx", ".xls", ".pkl", ".pickle", ".joblib"}
ALLOWED_PARQUET = {
    "external_nasa_v329/scored_v331/nasa_released_labels_v331.parquet"
}
EXPECTED_FIGURE_PNG_SHA256 = {
    "fig1_method_workflow.png": "39264011e8cce0986cf827c71542cf2c71f4f0c4aee4c7a4adef74ad43d1478b",
    "fig1_method_workflow_zh.png": "ded7bc91df862da51aa0349dfb0b1a8749d3e0dfe6f90bd6f7378793be613be3",
    "fig2_domain_effects.png": "93b1e9051cc92ddf1ce586b3679f4f6ff2173b936a17401960b12c2a38a183a2",
    "fig2_domain_effects_zh.png": "1034a88ecd4d6c47e52a09c9bb6f98f8525642a8eb422c487f56c2a11dffa73e",
    "fig3_prefix_causality.png": "152655a24792b06c0ff23373935d0c2a446a6cad579c15111e7e2d12f5ed4a1c",
    "fig3_prefix_causality_zh.png": "641c720722df8639c9500d1b0e1db8c2b49cac955b37c5a6b19f2dba1e857882",
    "fig4_candidate_control.png": "0a8d531196b6855b8d39f5fc4e76c8a2c61c2075cd2b6fcfa8a4905ac0dc251d",
    "fig4_candidate_control_zh.png": "741ac9bcfe7cd258f3bdc3925b4a18cc4f49fcfaa41738c1063acf268b306290",
    "fig5_nasa_stress.png": "12b49c708b719c2f4c5dc68176789338d4e8190ae1db57aa8f6d68b0cd3adca2",
    "fig5_nasa_stress_zh.png": "fb52651682d2f31f56c98ec4b622e7f21e5f5f3ce7c7d77480210286a3fdd388",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class Audit:
    def __init__(self) -> None:
        self.passed: list[str] = []

    def check(self, condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        self.passed.append(name)

    def close(self, observed: float, expected: float, name: str, atol: float = TOL) -> None:
        self.check(bool(np.isclose(observed, expected, rtol=0.0, atol=atol)), name)


def verify_manifest(audit: Audit) -> None:
    manifest = load_json("manifest_v341.json")
    audit.check(manifest["version"] == "v341", "manifest version")
    audit.check(manifest["raw_third_party_archives_included"] is False, "manifest raw-data boundary")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.is_file(), f"manifest file exists: {item['path']}")
        audit.check(path.stat().st_size == item["bytes"], f"manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"manifest hash: {item['path']}")
    audit.check(manifest["tracked_files"] == len(manifest["files"]), "manifest file count")


def verify_code_and_boundary(audit: Audit) -> None:
    for relative, expected in EXPECTED_CODE_SHA256.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"authoritative code exists: {relative}")
        audit.check(sha256_file(path) == expected, f"authoritative code hash: {relative}")

    all_files = [path for path in ROOT.rglob("*") if path.is_file()]
    forbidden = [path for path in all_files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    audit.check(not forbidden, "no forbidden raw/archive/model files")
    parquet = {path.relative_to(ROOT).as_posix() for path in all_files if path.suffix.lower() == ".parquet"}
    audit.check(parquet == ALLOWED_PARQUET, "only whitelisted derived NASA parquet")

    local_modules = {path.stem for path in ROOT.glob("*.py")}
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                if module in local_modules:
                    audit.check((ROOT / f"{module}.py").is_file(), f"local import closure: {path.name} -> {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in local_modules:
                        audit.check((ROOT / f"{module}.py").is_file(), f"local import closure: {path.name} -> {module}")

    for relative in EXPECTED_CODE_SHA256:
        if "/" not in relative and not relative.startswith("download_"):
            importlib.import_module(Path(relative).stem)
    audit.check(True, "all frozen local modules import successfully")


def verify_operator(audit: Audit) -> None:
    rng = np.random.default_rng(20260802)
    raw = rng.normal(0.9, 0.2, 200)
    candidate = rng.normal(0.8, 0.25, 200)
    baseline = causal_nonincreasing_state(raw, assimilation=0.02)
    projected = prefix_causal_harm_projection(baseline, candidate, 0.01)
    audit.check(bool((np.diff(baseline) <= TOL).all()), "protected state is non-increasing")
    audit.check(bool((np.diff(projected) <= TOL).all()), "projected output is non-increasing")
    audit.check(bool((np.abs(projected - baseline) <= 0.01 + TOL).all()), "pointwise displacement budget")
    audit.check(bool(((projected >= -TOL) & (projected <= 1.3 + TOL)).all()), "physical output range")
    for length in (1, 2, 5, 31, 109, 200):
        prefix_baseline = causal_nonincreasing_state(raw[:length], assimilation=0.02)
        prefix = prefix_causal_harm_projection(prefix_baseline, candidate[:length], 0.01)
        audit.check(np.array_equal(prefix, projected[:length]), f"prefix invariance at length {length}")
    zero = prefix_causal_harm_projection(baseline, candidate, 0.0)
    audit.check(np.array_equal(zero, baseline), "zero-budget identity")

    for base, value in zip(baseline[::17], projected[::17]):
        outcomes = np.array([-10.0, 10.0, base, value])
        regret = np.abs(value - outcomes) - np.abs(base - outcomes)
        audit.close(float(regret.max()), abs(value - base), "exact absolute-loss supremum")

    try:
        prefix_causal_harm_projection(np.array([0.8, 0.9]), np.array([0.8, 0.8]), 0.01)
    except ValueError:
        audit.check(True, "increasing protected state fails closed")
    else:
        audit.check(False, "increasing protected state fails closed")
    try:
        prefix_causal_harm_projection(np.array([0.8]), np.array([0.8]), -0.01)
    except ValueError:
        audit.check(True, "negative budget fails closed")
    else:
        audit.check(False, "negative budget fails closed")


def verify_asymmetric_extension(audit: Audit) -> None:
    prefreeze = load_json("paper_q1/ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json")
    audit.check(
        prefreeze["status"] == "PREFROZEN_BEFORE_IMPLEMENTATION",
        "asymmetric extension pre-freeze status",
    )
    audit.check(
        prefreeze["falsification"]["failure_rule"].startswith(
            "Any failed gate yields REJECT"
        ),
        "asymmetric extension failure rule",
    )

    unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "test_prefix_causal_harm_projection_v321.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    unit_output = unit.stdout + unit.stderr
    audit.check(unit.returncode == 0, "operator unit-test process passed")
    audit.check("Ran 30 tests" in unit_output, "all 30 operator unit tests executed")

    subprocess.run(
        [sys.executable, str(ROOT / "validate_asymmetric_harm_extension_v340.py")],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    report = load_json(
        "paper_q1/asymmetric_harm_extension_v340/"
        "asymmetric_harm_extension_v340_report.json"
    )
    audit.check(
        report["status"] == "ASYMMETRIC_HARM_EXTENSION_RETAINED",
        "asymmetric extension retained status",
    )
    audit.check(report["decision"] == "RETAIN", "asymmetric extension decision")
    audit.check(
        report["random_scalar_tuples"] == 10000,
        "asymmetric scalar replay count",
    )
    audit.check(
        report["random_trajectory_trials"] == 100,
        "asymmetric trajectory replay count",
    )
    audit.check(
        report["maximum_direct_supremum_error"] <= TOL,
        "asymmetric direct supremum tolerance",
    )
    for name, passed in report["checks"].items():
        audit.check(passed is True, f"asymmetric gate: {name}")

    artifact_hashes = {
        item["path"]: item["sha256"].lower() for item in report["artifacts"]
    }
    for relative in (
        "paper_q1/ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json",
        "prefix_causal_harm_projection_v321.py",
        "test_prefix_causal_harm_projection_v321.py",
        "validate_asymmetric_harm_extension_v340.py",
    ):
        audit.check(
            artifact_hashes[relative] == sha256_file(ROOT / relative),
            f"asymmetric artifact identity: {relative}",
        )

    baseline = np.array([0.9])
    eta = 0.08
    projected_up = prefix_causal_asymmetric_harm_projection(
        baseline,
        np.array([2.0]),
        eta,
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    projected_down = prefix_causal_asymmetric_harm_projection(
        baseline,
        np.array([-2.0]),
        eta,
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    audit.close(
        float(projected_up[0] - baseline[0]),
        eta / 8.0,
        "asymmetric upward radius",
    )
    audit.close(
        float(baseline[0] - projected_down[0]),
        eta / 2.0,
        "asymmetric downward radius",
    )
    harm = worst_case_asymmetric_absolute_loss_increase(
        np.array([0.9, 0.9]),
        np.array([projected_down[0], projected_up[0]]),
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    audit.check(
        bool((harm <= eta + TOL).all()),
        "asymmetric directional harm budget",
    )


def verify_time_varying_viability(audit: Audit) -> None:
    prefreeze = load_json("paper_q1/TIME_VARYING_VIABILITY_PREFREEZE_V341.json")
    audit.check(
        prefreeze["status"] == "PREFROZEN_BEFORE_IMPLEMENTATION",
        "time-varying viability pre-freeze status",
    )
    audit.check(
        prefreeze["falsification"]["failure_rule"].startswith(
            "Any failed gate yields REJECT"
        ),
        "time-varying viability failure rule",
    )

    subprocess.run(
        [sys.executable, str(ROOT / "validate_time_varying_viability_v341.py")],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    report = load_json(
        "paper_q1/time_varying_viability_v341/"
        "time_varying_viability_v341_report.json"
    )
    audit.check(report["version"] == "v341", "time-varying report version")
    audit.check(report["decision"] == "RETAIN", "time-varying viability decision")
    audit.check(report["status"] == "PASS", "time-varying viability status")
    gates = {item["gate"]: item for item in report["gates"]}
    expected = {
        "exhaustive_small_grid": {
            "schedules": 2160,
            "candidate_trajectories": 58320,
            "viable_schedules": 1029,
            "nonviable_schedules": 1131,
        },
        "randomized_viable_schedules": {
            "schedules": 300,
            "candidate_trajectories": 12000,
            "records": 504880,
            "prefix_replays": 12000,
        },
        "constructive_upward_boundary_failures": {
            "counterexamples": 1000,
            "matched_failure_indices": 1000,
        },
        "realized_prefix_equivalence": {
            "trajectories": 3000,
            "feasible": 305,
            "infeasible": 2695,
        },
        "minimum_budget_exactness": {
            "active_trials": 1000,
            "rejected_immediately_below": 1000,
            "inactive_zero_floor_trials": 500,
        },
        "upper_cost_feasibility_invariance": {
            "trajectories": 2000,
            "feasible": 163,
            "infeasible": 1837,
        },
        "constant_schedule_exact_recovery": {
            "random_trajectories": 1000,
            "exact_array_equalities": 2000,
        },
        "invalid_contracts_fail_closed": {"invalid_cases": 11, "rejected": 11},
        "existing_unit_suite": {"tests": 30, "return_code": 0},
    }
    audit.check(set(gates) == set(expected), "all nine time-varying gates present")
    for name, evidence in expected.items():
        audit.check(gates[name]["status"] == "PASS", f"time-varying gate passed: {name}")
        audit.check(gates[name]["evidence"] == evidence, f"time-varying evidence: {name}")

    artifact_hashes = {
        "paper_q1/TIME_VARYING_VIABILITY_PREFREEZE_V341.json": report["prefreeze_sha256"],
        "prefix_causal_harm_projection_v321.py": report["implementation_sha256"],
        "test_prefix_causal_harm_projection_v321.py": report["unit_tests_sha256"],
    }
    for relative, expected_hash in artifact_hashes.items():
        audit.check(
            expected_hash.lower() == sha256_file(ROOT / relative),
            f"time-varying artifact identity: {relative}",
        )

    baseline = np.array([0.90, 0.89, 0.88])
    budget = np.array([0.02, 0.04, 0.06])
    under = np.array([2.0, 4.0, 6.0])
    over = np.array([8.0, 5.0, 3.0])
    lower, upper = time_varying_asymmetric_harm_tube_bounds(
        baseline,
        budget,
        underestimation_cost=under,
        overestimation_cost=over,
    )
    audit.check(bool((np.diff(lower) <= TOL).all()), "example lower schedule is universally viable")
    projected = prefix_causal_time_varying_asymmetric_harm_projection(
        baseline,
        np.array([0.0, 2.0, 0.0]),
        budget,
        underestimation_cost=under,
        overestimation_cost=over,
    )
    audit.check(bool((projected >= lower - TOL).all()), "time-varying projection respects lower tube")
    audit.check(bool((projected <= upper + TOL).all()), "time-varying projection respects upper tube")
    floors = minimum_viable_asymmetric_budget(
        baseline[1:], projected[:-1], underestimation_cost=under[1:]
    )
    expected_floors = under[1:] * np.maximum(baseline[1:] - projected[:-1], 0.0)
    audit.check(np.array_equal(floors, expected_floors), "exact minimum viable budget API")


def verify_development(audit: Audit) -> None:
    report = load_json("nested_prefix_causal_outer_v327/nested_prefix_causal_outer_v327_report.json")
    audit.check(report["status"] == "NESTED_PREFIX_CAUSAL_OUTER_GATE_PASSED", "development gate status")
    audit.check(report["outer_target_labels_used_for_alpha_selection"] is False, "outer labels excluded from selection")
    audit.check((report["domains"], report["physical_cells"], report["post_reference_rows"]) == (12, 586, 601932), "development roster")
    audit.check(report["deterministic_certificate"]["passed"] is True, "development deterministic certificate")

    domains = pd.read_csv(ROOT / "nested_prefix_causal_outer_v327/nested_outer_domain_metrics_v327.csv")
    method = domains.loc[domains["method"] == "selected_causal_method", ["domain", "cell_macro_mae"]]
    baseline = domains.loc[domains["method"] == "selected_causal_baseline", ["domain", "cell_macro_mae"]]
    paired = method.merge(baseline, on="domain", suffixes=("_method", "_baseline"), validate="one_to_one")
    differences = (paired["cell_macro_mae_method"] - paired["cell_macro_mae_baseline"]).to_numpy(float)
    primary = report["primary_comparison"]
    audit.close(float(differences.mean()), primary["domain_equal_mean_difference"], "development mean effect")
    audit.check([
        int((differences < -TOL).sum()),
        int((np.abs(differences) <= TOL).sum()),
        int((differences > TOL).sum()),
    ] == [primary["domain_wins"], primary["domain_ties"], primary["domain_losses"]], "development wins ties losses")
    rng = np.random.default_rng(20260802)
    indices = rng.integers(0, len(differences), size=(100000, len(differences)))
    ci = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
    audit.check(np.allclose(ci, [primary["ci95_lower"], primary["ci95_upper"]], rtol=0.0, atol=TOL), "development domain bootstrap")


def verify_candidate_control(audit: Audit) -> None:
    report = load_json("candidate_information_control_v333/candidate_information_control_v333_report.json")
    audit.check(report["status"] == "CANDIDATE_INFORMATION_CONTROL_GATE_PASSED", "candidate control status")
    audit.check(report["control"]["outer_target_labels_used_for_selection"] is False, "candidate control selection blindness")
    audit.check(report["nasa_artifacts_accessed"] is False, "candidate control excludes NASA")
    domain = pd.read_csv(ROOT / "candidate_information_control_v333/candidate_information_domain_metrics_v333.csv")
    differences = domain["pchp_minus_control"].to_numpy(float)
    comparison = report["comparison"]
    audit.close(float(differences.mean()), comparison["domain_equal_pchp_minus_control"], "candidate-information mean effect")
    rng = np.random.default_rng(20260801)
    indices = rng.integers(0, len(differences), size=(100000, len(differences)))
    ci = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
    audit.check(np.allclose(ci, comparison["domain_cluster_percentile_bootstrap_ci95"], rtol=0.0, atol=TOL), "candidate-information domain bootstrap")
    audit.check(comparison["candidate_information_gate_passed"] is True, "candidate-information gate")


def verify_source_group_sensitivity(audit: Audit) -> None:
    script = ROOT / "analyze_source_group_sensitivity_v339.py"
    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    report = load_json(
        "source_group_sensitivity_v339/source_group_sensitivity_v339_report.json"
    )
    audit.check(
        report["status"] == "POSTHOC_SOURCE_GROUP_SENSITIVITY_COMPLETED",
        "source-group sensitivity status",
    )
    audit.check(report["primary_estimator_unchanged"] is True, "primary estimator remains frozen")
    audit.check(
        [item["sha256"] for item in report["inputs"]]
        == [
            "724C65D212BEA87B4C35E99D6FDF1A72708C9991D51DC51EC1777592AB29A7C3",
            "D9C48CF22F5602029A7CD06D6B0E884EB667B2A10C6EA9C6AEDFFA54655FEF70",
        ],
        "source-group sensitivity input identities",
    )
    audit.check(
        [item["path"] for item in report["inputs"]]
        == [
            "nested_prefix_causal_outer_v327/nested_outer_domain_metrics_v327.csv",
            "candidate_information_control_v333/candidate_information_domain_metrics_v333.csv",
        ],
        "source-group sensitivity portable input paths",
    )
    groups_path = ROOT / "source_group_sensitivity_v339/source_group_effects_v339.csv"
    audit.check(
        report["output"]["path"]
        == "source_group_sensitivity_v339/source_group_effects_v339.csv",
        "source-group sensitivity portable output path",
    )
    audit.check(
        report["output"]["sha256"].lower() == sha256_file(groups_path),
        "source-group sensitivity output identity",
    )
    groups = pd.read_csv(groups_path)
    audit.check(set(groups["source_group"]) == {
        "CALB", "CALCE", "HNEI", "HUST", "MATR", "MICH_PROGRAM",
        "RWTH", "SDU", "SNL", "UL_PUR", "XJTU"
    }, "source-group roster and MICH merge")
    expected = {
        "primary_development": {
            "mean": -0.004685852821470012,
            "ci": [-0.006667048938040285, -0.002738406164937248],
            "wins": [10, 0, 1],
            "loo": [-0.005263669898055313, -0.004167638973188053],
            "seed": 20260803,
        },
        "candidate_control": {
            "mean": -0.004666637021702945,
            "ci": [-0.00824926401148149, -0.0012516269734047557],
            "wins": [8, 0, 3],
            "loo": [-0.00555707689891555, -0.003466528652346539],
            "seed": 20260804,
        },
    }
    summaries = {item["analysis"]: item for item in report["analyses"]}
    audit.check(set(summaries) == set(expected), "source-group sensitivity analyses")
    for analysis, target in expected.items():
        summary = summaries[analysis]
        rows = groups.loc[groups["analysis"] == analysis].sort_values("source_group")
        differences = rows["pchp_minus_baseline"].to_numpy(float)
        audit.check(len(differences) == summary["source_groups"] == 11, f"{analysis} source-group count")
        audit.close(float(differences.mean()), target["mean"], f"{analysis} source-group mean")
        audit.check(summary["source_group_wins_ties_losses"] == target["wins"], f"{analysis} source-group wins ties losses")
        rng = np.random.default_rng(target["seed"])
        indices = rng.integers(0, len(differences), size=(100000, len(differences)))
        ci = np.percentile(differences[indices].mean(axis=1), [2.5, 97.5])
        audit.check(np.allclose(ci, target["ci"], rtol=0.0, atol=TOL), f"{analysis} source-group bootstrap")
        loo = np.asarray(
            [np.delete(differences, index).mean() for index in range(len(differences))],
            dtype=float,
        )
        audit.check(
            np.allclose([loo.min(), loo.max()], target["loo"], rtol=0.0, atol=TOL),
            f"{analysis} leave-one-source-group-out range",
        )
        audit.check(bool((loo < 0.0).all()), f"{analysis} all leave-one-source-group-out means negative")


def verify_nasa(audit: Audit) -> None:
    blind = load_json("external_nasa_v329/label_blind_v330/nasa_label_blind_prediction_report_v330.json")
    audit.check(blind["status"] == "NASA_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS", "NASA prediction freeze status")
    audit.check(blind["external_outcome_fields_accessed"] == [], "NASA outcome fields excluded before freeze")
    audit.check(blind["archive"]["sha256"] == "82302A7DB4FC1B34E0B6676326610438D43B816BDF11A69D1D012A464EF2F92E", "NASA archive identity")
    audit.check(blind["certificate"]["label_free_structural_certificate_passed"] is True, "NASA label-free certificate")

    report = load_json("external_nasa_v329/scored_v331/nasa_frozen_confirmation_v331_report.json")
    cells = pd.read_csv(ROOT / "external_nasa_v329/scored_v331/nasa_cell_metrics_v331.csv")
    differences = cells["method_minus_baseline"].to_numpy(float)
    audit.check(report["status"] == "NASA_ONE_SHOT_EXTERNAL_GATE_PASSED", "NASA one-shot gate status")
    audit.check(report["no_post_outcome_rescue_permitted"] is True, "NASA no post-outcome rescue")
    audit.check(len(differences) == report["independent_cells"] == 33, "NASA independent battery count")
    audit.close(float(differences.mean()), report["cell_equal_method_minus_baseline"], "NASA cell-equal mean effect")
    audit.close(float(differences.max()), report["maximum_cell_macro_harm"], "NASA maximum cell harm")
    wins = int((differences < -TOL).sum())
    ties = int((np.abs(differences) <= TOL).sum())
    losses = int((differences > TOL).sum())
    audit.check([wins, ties, losses] == report["cell_wins_ties_losses"], "NASA wins ties losses")
    rng = np.random.default_rng(20260801)
    means = np.empty(100000, dtype=float)
    for start in range(0, 100000, 10000):
        stop = min(start + 10000, 100000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    ci = np.percentile(means, [2.5, 97.5])
    audit.check(np.allclose(ci, report["cell_cluster_percentile_bootstrap"]["ci95"], rtol=0.0, atol=TOL), "NASA cell bootstrap")
    trials = wins + losses
    tail = min(wins, losses)
    sign_p = min(1.0, 2.0 * sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials))
    audit.close(sign_p, report["exact_two_sided_sign_test"]["p_value"], "NASA exact sign test")
    audit.check(all(report["primary_gates"].values()), "all NASA primary gates")


def verify_falsification_and_freeze(audit: Audit) -> None:
    future = pd.read_csv(ROOT / "prefix_causal_falsification_v325/future_revision_domain_v325.csv")
    shock = pd.read_csv(ROOT / "prefix_causal_falsification_v325/directional_shock_domain_v325.csv")
    audit.check(bool((future["maximum_causal_difference"].abs() <= TOL).all()), "causal prefix outputs never revised")
    audit.check(int((future["fraction_cells_revised"] >= 0.5).sum()) >= 9, "retrospective revision falsification gate")
    audit.check(int((shock["fraction_reduced"] >= 0.95).sum()) >= 9, "directional shock attenuation gate")
    audit.check(bool((shock["median_cumulative_disturbance_alpha_0p02"] < shock["median_cumulative_disturbance_alpha_1"]).all()), "slow assimilation reduces disturbance")

    freeze = load_json("external_causal_source_freeze_v328/external_source_alpha_freeze_v328_report.json")
    audit.check(freeze["external_data_accessed"] is False, "external alpha frozen without external data")
    audit.close(float(freeze["selected_alpha"]), 0.01, "frozen external assimilation")
    audit.check(freeze["all_grid_certificates_verified"] is True, "external alpha grid certificates")


def regenerate_and_verify_figures(audit: Audit) -> None:
    script = ROOT / "paper_q1/rccp_causal_manuscript_v335/make_figures_v335.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    figure_root = script.parent / "figures"
    for name, expected in EXPECTED_FIGURE_PNG_SHA256.items():
        path = figure_root / name
        audit.check(path.is_file(), f"generated figure exists: {name}")
        audit.check(sha256_file(path) == expected, f"generated figure content hash: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    verify_manifest(audit)
    verify_code_and_boundary(audit)
    verify_operator(audit)
    verify_asymmetric_extension(audit)
    verify_time_varying_viability(audit)
    verify_development(audit)
    verify_candidate_control(audit)
    verify_source_group_sensitivity(audit)
    verify_nasa(audit)
    verify_falsification_and_freeze(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_REVIEW_LITE_VERIFICATION_PASSED",
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
            "No source-model refit, third-party development raw-data redistribution, "
            "team-wide prospective external confirmation, or empirical deployment-cost "
            "calibration is claimed."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v341.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
