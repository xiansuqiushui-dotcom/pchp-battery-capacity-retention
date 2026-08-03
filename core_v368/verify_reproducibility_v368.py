"""Failure-closed verifier for the PCHP V368 external-validation review-lite package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

import verify_reproducibility_v366 as prior


ROOT = Path(__file__).resolve().parent
EXCLUDED_MANIFEST_NAMES = {"manifest_v368.json", "verification_receipt_v368.json"}

EXPECTED_V368_SHA256 = {
    "paper_q1/rccp_causal_manuscript_v368/main_en.tex": "17ee0cea641872f098c5190846dcd68c2046f2fb889aba735b4005ab82370a50",
    "paper_q1/rccp_causal_manuscript_v368/main_zh.tex": "cb9a35a36655a4d70472408fc95b5a85b8838dbd67682e83822e87d0e8504e4c",
    "paper_q1/rccp_causal_manuscript_v368/supplement_en.tex": "a14251cc6a389564a3fe6028d04b3bc96586238019163332bc1a696f6d5991cf",
    "paper_q1/rccp_causal_manuscript_v368/supplement_zh.tex": "67573728f54410889b0341da5da663db1d393b8a9873b489dffab18f9669480b",
    "paper_q1/rccp_causal_manuscript_v368/references.bib": "ebeb9be496fe5858ee38839aa74d7d81e02bc1719709bfca1a8f7987492aa1bb",
    "paper_q1/rccp_causal_manuscript_v368/highlights.txt": "0f8aaa040a410a21e196ef775e93972eabd471bf76a49efefb19447a473c4a9c",
    "paper_q1/rccp_causal_manuscript_v368/make_figures_v335.py": "8cabdb7ce51f768fa3c6dd7e7f86f6e4ac95dbc37744e58db2f1de4b4415136e",
    "paper_q1/rccp_causal_manuscript_v368/make_budget_path_figure_v363.py": "967c9d279e9584072b474f019a225e2d7def9565cee13e46766181020088dcba",
    "paper_q1/rccp_causal_manuscript_v368/make_basytec_figure_v359.py": "ed4469ced96a3c0e85578c21044b121f5929ac960b8393d44d14a51a4688eba1",
    "paper_q1/rccp_causal_manuscript_v368/make_graphical_abstract_v368.py": "e883dcd0872dce5fda79059f2058627b6b93f99730c97892527e7349e91f1319",
    "paper_q1/rccp_causal_manuscript_v368/graphical_abstract_applied_energy_v368.png": "dc603b611e0772c48cc0a9eb35a480418adfd94683586ac1739b95341f84a235",
    "paper_q1/VENUE_ROUTE_AND_CONCURRENT_WORK_AUDIT_V368_ZH.md": "b6dfef00bb701fb62bc62539993286da015c12ca0134d771383013b113869b19",
    "paper_q1/RCCP_METHOD_AND_EVIDENCE_AUDIT_V368_ZH.md": "04e43c0d0a9bb99bef5cde6242e3714cb8307fcfe760c989a59f092250c164d6",
    "paper_q1/APPLIED_ENERGY_SUBMISSION_COMPONENTS_V368.md": "e8ed4a169c77945de22bcfd4e0b0c22f1a786446f3d2e236334deb0b2a5e3948",
    "paper_q1/APPLIED_ENERGY_COVER_LETTER_V368.md": "60c29e7dc66994285489039e9fbe7e0deec6171da4be8cd74c5b5d4dd2aaac4c",
    "paper_q1/APPLIED_ENERGY_SUBMISSION_CHECKLIST_V368_ZH.md": "3129a18477c9e785285b3287074095b6e32250a59bf00bbacd876d521c7c515f",
    "paper_q1/RCCP_CAUSAL_PROJECT_HANDOFF_V368_ZH.md": "06b681ef7b1d0f5d387341d114d2265c2c71b0cda5c292621298935187c6182f",
    "paper_q1/TOP_Q1_EDITOR_METHOD_STAT_NOVELTY_AUDIT_V369_ZH.md": "042d143b72f629b8b6d1bfafa4e87ac8eaaa00b8c3dabe71d4fe6218da718de1",
    "paper_q1/TOP_Q1_ADVERSARIAL_EDITOR_AUDIT_V370_ZH.md": "8dee72db2b12e1db50dc3d087fdf1ed346d56289e42e2487fb79a7d5e6fc122f",
    "paper_q1/EXTERNAL_CONFIRMATION_AND_ENERGY_RELEVANCE_AUDIT_V371_ZH.md": "43aff7ed17745b4456bb78d17f2319511d947a05902a27fe3b311779861b3c5b",
    "paper_q1/APPLIED_ENERGY_EDITOR_FIRST_IMPRESSION_AUDIT_V372_ZH.md": "9b22ca3402517fe1296c51e389003a00342e4364b92c43983ea07b17abb58a85",
    "paper_q1/TOP_Q1_EXACT_VIABILITY_AND_CLOSEST_NEIGHBOR_AUDIT_V373_ZH.md": "1c64931f25bbb75687c772b8425c2e95f216b4eda8ea90b6c4121bcbb3b83c14",
    "test_prefix_causal_harm_projection_v321.py": "7d21846c6f23670cfeb4ddba9cc4035b3f7b85e13914f148f239794cb75df1ea",
    "BOUNDED_RECOVERY_PCHP_PREFREEZE_V374_ZH.md": "394152b0ce4b13f150007c0b3ee0e8e7c92a20a7043fda803e51887abbf50489",
    "BOUNDED_RECOVERY_PCHP_PREFREEZE_V374.json": "4f92382d67297e5a062a7bd96c7b9584d2966cae35094803ded10332e5a9328b",
    "BOUNDED_RECOVERY_PCHP_DECISION_V374_ZH.md": "b35f19c80bee0e58e69078033d029aa4a350d9662d2156a1192b9b1848b247e4",
    "bounded_recovery_pchp_v374.py": "284a1b5766cf5cf46b8f23b90b2a7ab1eb089b32ba2d01baaeb125e851dc2158",
    "test_bounded_recovery_pchp_v374.py": "dab2aeef97852596ad95ced130525e4cf35e9289a8703528e2a635bc4a8c5063",
    "evaluate_bounded_recovery_pchp_v374.py": "aa749ce15205fb21baf44c8dee76cbe17cde648085edd687f3f4947252d993f3",
    "bounded_recovery_pchp_v374/bounded_recovery_pchp_v374_report.json": "1e29936f4f3532e4dfe9f9795efcfcf4f336b9b7316c70ad4892bb348275f6b1",
    "PCHP_BACKBONE_PORTABILITY_PREFREEZE_V375_ZH.md": "f433559114bca1002d90e3ca1e3f5328011c74fddc8ab9477d1072399006a218",
    "PCHP_BACKBONE_PORTABILITY_PREFREEZE_V375.json": "ed6c7b6d119d4a3bb9111e27cae198c73f9c8590b3bb09756548ef06a95993e9",
    "PCHP_BACKBONE_PORTABILITY_DECISION_V375_ZH.md": "b8c2815154207dde96961796ac4db6daba4e9ae4823752de9d706ee709e5d419",
    "paper_q1/TOP_Q1_POST_V375_NOVELTY_AND_REVIEWER_RISK_AUDIT_V376_ZH.md": "bf7e98e628ca59d5f7143b88927399333597f82f22bdbc3170bddf248363ba67",
    "paper_q1/TOP_Q1_METHOD_ELEVATION_AND_STABILITY_AUDIT_V377_ZH.md": "a9e5912e9a214038a9a2cc1ba108b0b984cc5f2b67d6dcbfa7800eff4f704097",
    "validate_prefix_nonexpansiveness_v377.py": "9abe330253305b02250bba00da337ec8003fe5c9aa99b535a4af2cb842e38b35",
    "prefix_nonexpansiveness_v377_report.json": "2421dd474feb23a349b3be01354de240baf7b945ce5bbcdc95d5c7edbbd683f8",
    "test_prefix_nonexpansiveness_v377.py": "66f2fc349826e6e9c1fa22529a99546e5a30405dbbc7ee11b4dc8f2c1c76f14e",
    "paper_q1/TOP_Q1_SMALL_N_STATISTICAL_STRESS_AUDIT_V378_ZH.md": "437dc61bc3d9ddc076d46555e0525fbff39134ef5f864d5e7a0a37075654dd02",
    "audit_small_n_domain_inference_v378.py": "50426e90264f15e1632d0fa8403be7b16c7ef09c9076a34be05897f1a0b0b578",
    "small_n_domain_inference_v378_report.json": "411007452ed8c095258f3949af959e2518dc1c63e648583e5ffe9dde3b265ae9",
    "audit_pchp_backbone_portability_v375.py": "d1924aa47c516e4f3474942e87044df7d5bec746531b8f3816244f610e92d5eb",
    "verify_pchp_backbone_portability_v375.py": "90b5206ba7f596b81300f4a1095ba79b277f614919446e4c1fc408f39fc566ac",
    "pchp_backbone_portability_v375/pchp_backbone_portability_v375_report.json": "862482438df32b59a19cfe8c4d15f413a11ee0baa2fd8e4451c9066e4d76562d",
    "pchp_backbone_portability_v375/pchp_backbone_portability_v375_verification_receipt.json": "26e81d0a73374a954e518dcda7e37bdd61ba04e88b407f8de55c82476a889376",
}

EXPECTED_V368_FIGURE_PNG_SHA256 = {
    "fig1_method_workflow.png": "0c81cb79af9e0a763da7558c0642b0e6e515b7ad2cf9e99fa0a4a663096c3f1f",
    "fig1_method_workflow_zh.png": "91ab0756df65c6dad5a5fed39e74319d5b97ff48948e93c732a707757d444ccc",
    "fig2_domain_effects.png": "93b1e9051cc92ddf1ce586b3679f4f6ff2173b936a17401960b12c2a38a183a2",
    "fig2_domain_effects_zh.png": "1034a88ecd4d6c47e52a09c9bb6f98f8525642a8eb422c487f56c2a11dffa73e",
    "fig3_prefix_causality.png": "152655a24792b06c0ff23373935d0c2a446a6cad579c15111e7e2d12f5ed4a1c",
    "fig3_prefix_causality_zh.png": "641c720722df8639c9500d1b0e1db8c2b49cac955b37c5a6b19f2dba1e857882",
    "fig4_candidate_control.png": "d3e388de4147914796890e84eb535c6a10de82800ec5e7a9a89ac6981cb8f029",
    "fig4_candidate_control_zh.png": "2d044fb54c33bb0275a6b0ea5e9829fae618572a120cfe1d43cce1a6cead7c37",
    "fig5_nasa_stress.png": "12b49c708b719c2f4c5dc68176789338d4e8190ae1db57aa8f6d68b0cd3adca2",
    "fig5_nasa_stress_zh.png": "fb52651682d2f31f56c98ec4b622e7f21e5f5f3ce7c7d77480210286a3fdd388",
    "fig6_basytec_confirmation.png": "bf6bd386d29dd393c79ae09201a8cb1b11dd71b807c2658353ee84dca74d8e7b",
    "fig6_basytec_confirmation_zh.png": "e1a563967a596e70b3f309418471a951f6898879cf37bd1cfeb0d3c9e75c24ff",
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


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise AssertionError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", data[16:24])


def abstract_word_count(tex: str) -> int:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if match is None:
        raise AssertionError("English abstract not found")
    plain = re.sub(r"\$([^$]+)\$", r"\1", match.group(1))
    plain = plain.replace(r"\%", "%")
    plain = re.sub(r"\\[A-Za-z]+\*?", " ", plain)
    plain = re.sub(r"[{}]", " ", plain)
    return len([token for token in re.split(r"\s+", plain.strip()) if token])


def verify_manifest(audit: prior.prior.prior.core.Audit) -> None:
    manifest = load_json("manifest_v368.json")
    audit.check(manifest["version"] == "v368", "V368 manifest version")
    audit.check(manifest["scientific_evidence_inherited_from_v366"] is True, "V368 inherits frozen V366 science")
    audit.check(manifest["contains_applied_energy_submission_assets"] is True, "V368 submission assets declared")
    audit.check(manifest["contains_v368_venue_and_concurrent_work_audit"] is True, "V368 venue audit declared")
    audit.check(manifest["contains_v368_method_and_evidence_audit"] is True, "V368 method and evidence audit declared")
    audit.check(manifest["contains_v369_editor_method_statistics_novelty_audit"] is True, "V369 editor method-statistics-novelty audit declared")
    audit.check(manifest["contains_v370_adversarial_editor_audit"] is True, "V370 adversarial editor audit declared")
    audit.check(manifest["contains_v371_external_confirmation_energy_relevance_audit"] is True, "V371 external-confirmation and energy-relevance audit declared")
    audit.check(manifest["contains_v372_editor_first_impression_audit"] is True, "V372 editor first-impression audit declared")
    audit.check(manifest["contains_v373_exact_viability_and_closest_neighbor_audit"] is True, "V373 exact-viability and closest-neighbor audit declared")
    audit.check(manifest["contains_v374_rejected_bounded_recovery_audit"] is True, "V374 rejected bounded-recovery audit declared")
    audit.check(manifest["contains_v375_untuned_backbone_portability_audit"] is True, "V375 untuned-backbone portability audit declared")
    audit.check(manifest["contains_v376_novelty_and_reviewer_risk_audit"] is True, "V376 novelty and reviewer-risk audit declared")
    audit.check(manifest["contains_v377_method_elevation_and_stability_audit"] is True, "V377 method-elevation and stability audit declared")
    audit.check(manifest["contains_prefix_nonexpansiveness_theorem_and_falsification"] is True, "V377 prefix-nonexpansiveness evidence declared")
    audit.check(manifest["v377_stability_is_algorithmic_not_accuracy_or_physical_safety"] is True, "V377 stability claim boundary declared")
    audit.check(manifest["contains_v378_small_n_domain_inference_audit"] is True, "V378 finite-sample domain audit declared")
    audit.check(manifest["v378_development_sensitivity_is_posthoc_not_confirmation"] is True, "V378 retrospective evidence boundary declared")
    audit.check(manifest["v378_candidate_control_supports_mean_not_universal_or_majority"] is True, "V378 candidate-control estimand boundary declared")
    audit.check(manifest["v375_backbone_portability_is_retrospective_not_external_confirmation"] is True, "V375 retrospective evidence boundary declared")
    audit.check(manifest["contains_v368_bilingual_manuscript"] is True, "V368 bilingual manuscript declared")
    audit.check(manifest["contains_v368_graphical_abstract"] is True, "V368 graphical abstract declared")
    audit.check(manifest["contains_v368_ai_use_declaration"] is True, "V368 AI declaration declared")
    audit.check(manifest["contains_outcome_blind_prospectively_locked_external_validation"] is True, "V368 external-validation evidence declared")
    audit.check(manifest["does_not_claim_prospective_data_collection_or_independent_lab_replication"] is True, "V368 higher-tier evidence boundary declared")
    audit.check(manifest["external_confirmation_uses_design_independence_not_absolute_lineage"] is True, "V376 design-independence confirmation standard declared")
    audit.check(manifest["raw_third_party_archives_included"] is False, "V368 raw-data boundary")
    expected_paths = {item["path"] for item in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if tracked(path)
    }
    audit.check(actual_paths == expected_paths, "V368 manifest has no untracked scientific files")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.is_file(), f"V368 manifest file exists: {item['path']}")
        audit.check(path.stat().st_size == item["bytes"], f"V368 manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"V368 manifest hash: {item['path']}")
    audit.check(manifest["tracked_files"] == len(manifest["files"]), "V368 manifest file count")


def verify_inherited_science(audit: prior.prior.prior.core.Audit) -> None:
    core = prior.prior.prior.core
    core.ALLOWED_PARQUET = set(core.ALLOWED_PARQUET) | {
        "pchp_backbone_portability_v375/pchp_backbone_portability_predictions_v375.parquet"
    }
    core.verify_code_and_boundary(audit)
    core.verify_operator(audit)
    core.verify_asymmetric_extension(audit)
    core.verify_time_varying_viability(audit)
    core.verify_development(audit)
    core.verify_candidate_control(audit)
    core.verify_source_group_sensitivity(audit)
    core.verify_nasa(audit)
    core.verify_falsification_and_freeze(audit)
    prior.prior.prior.verify_pinned_new_artifacts(audit)
    prior.prior.prior.verify_source_tuned_comparator(audit)
    prior.prior.prior.verify_basytec_chronology(audit)
    prior.prior.prior.verify_basytec_confirmation(audit)
    prior.prior.verify_budget_path(audit)
    prior.prior.verify_v362_audit(audit)
    prior.verify_pinned_v366_artifacts(audit)
    prior.verify_loss_geometry(audit)


def verify_v368_pinned_artifacts(audit: prior.prior.prior.core.Audit) -> None:
    for relative, expected in EXPECTED_V368_SHA256.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"V368 pinned artifact exists: {relative}")
        audit.check(sha256_file(path) == expected, f"V368 pinned artifact hash: {relative}")


def verify_v374_rejected_extension(audit: prior.prior.prior.core.Audit) -> None:
    report = load_json("bounded_recovery_pchp_v374/bounded_recovery_pchp_v374_report.json")
    audit.check(report["version"] == "v374", "V374 report version")
    audit.check(report["decision"] == "REJECT", "V374 rejected decision retained")
    audit.check(report["status"] == "BOUNDED_RECOVERY_PCHP_REJECTED", "V374 rejected status retained")
    comparison = report["primary_comparison"]
    audit.check(comparison["domain_equal_mean_difference"] < 0.0, "V374 negative point estimate retained")
    audit.check(comparison["ci95_domain_cluster_percentile"][1] > 0.0, "V374 failed interval gate retained")
    audit.check(comparison["domain_wins_ties_losses"] == [7, 0, 5], "V374 failed domain-win gate retained")
    gates = report["gates"]
    audit.check(gates["ci95_upper_below_zero"] is False, "V374 interval gate is not rescued")
    audit.check(gates["minimum_improved_outer_domains"] is False, "V374 win-count gate is not rescued")
    deterministic_gate_names = (
        "absolute_displacement_budget",
        "observed_absolute_loss_regret",
        "state_recovery_envelope",
        "output_recovery_envelope",
        "physical_range",
        "prefix_replay_exact",
        "zero_recovery_exactly_recovers_v321",
        "unit_tests",
    )
    audit.check(all(gates[name] for name in deterministic_gate_names), "V374 deterministic gates passed despite empirical rejection")
    audit.check(not (ROOT / "bounded_recovery_pchp_v374/bounded_recovery_outer_predictions_v374.parquet").exists(), "V374 large rejected outer-prediction artifact excluded from review-lite package")
    audit.check(report["unit_tests"] == {"return_code": 0, "tests": 6, "passed": True}, "V374 frozen property-test result retained")


def verify_v375_backbone_portability(audit: prior.prior.prior.core.Audit) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "verify_pchp_backbone_portability_v375.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    audit.check(payload["status"] == "PCHP_BACKBONE_PORTABILITY_V375_VERIFICATION_PASSED", "V375 independent verifier status")
    audit.check(payload["named_checks_passed"] == 196, "V375 independent verifier check count")
    audit.check(payload["report_sha256"] == EXPECTED_V368_SHA256["pchp_backbone_portability_v375/pchp_backbone_portability_v375_report.json"], "V375 independently verified report identity")
    receipt = ROOT / "pchp_backbone_portability_v375/pchp_backbone_portability_v375_verification_receipt.json"
    audit.check(sha256_file(receipt) == EXPECTED_V368_SHA256["pchp_backbone_portability_v375/pchp_backbone_portability_v375_verification_receipt.json"], "V375 deterministic verification receipt identity")


def verify_v377_prefix_nonexpansiveness(audit: prior.prior.prior.core.Audit) -> None:
    unit_completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_prefix_nonexpansiveness_v377.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    audit.check(unit_completed.returncode == 0, "V377 four-test extension passed")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "validate_prefix_nonexpansiveness_v377.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    audit.check(report["schema"] == "pchp_prefix_nonexpansiveness_v377", "V377 report schema")
    audit.check(report["status"] == "PCHP_PREFIX_NONEXPANSIVENESS_V377_PASSED", "V377 validator status")
    audit.check(report["seed"] == 20260802, "V377 frozen random seed")
    audit.check(report["absolute_tolerance"] == 2e-12, "V377 frozen absolute tolerance")
    audit.check(report["exhaustive_local_protected_checks"] == 3888, "V377 protected-state local check count")
    audit.check(report["exhaustive_local_projection_checks"] == 21168, "V377 projection local check count")
    audit.check(report["proximal_equivalence_checks"] == 108, "V377 proximal-equivalence check count")
    audit.check(report["random_trajectory_pairs"] == 30000, "V377 random trajectory-pair count")
    audit.check(report["maximum_baseline_excess"] <= report["absolute_tolerance"], "V377 protected-state prefix bound")
    audit.check(report["maximum_output_excess"] <= report["absolute_tolerance"], "V377 complete-output prefix bound")
    audit.check(abs(report["tightness_attained"] - 0.03) <= report["absolute_tolerance"], "V377 tight unit-Lipschitz witness")
    report_path = ROOT / "prefix_nonexpansiveness_v377_report.json"
    audit.check(
        sha256_file(report_path) == EXPECTED_V368_SHA256["prefix_nonexpansiveness_v377_report.json"],
        "V377 deterministic validation report identity",
    )


def verify_v378_small_n_domain_inference(audit: prior.prior.prior.core.Audit) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "audit_small_n_domain_inference_v378.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    audit.check(report["version"] == "v378", "V378 report version")
    audit.check(
        report["status"] == "SMALL_N_DOMAIN_INFERENCE_SENSITIVITY_PASSED",
        "V378 finite-sample sensitivity status",
    )
    audit.check(
        report["role"] == "posthoc_small_sample_sensitivity_not_new_confirmation",
        "V378 evidence tier",
    )
    audit.check(report["design"]["independent_unit"] == "complete_dataset_domain", "V378 independent unit")
    audit.check(report["design"]["n"] == 12, "V378 complete-domain count")
    audit.check(report["design"]["primary_analysis_unchanged"] is True, "V378 frozen primary analysis retained")
    comparisons = report["comparisons"]
    primary = comparisons["primary_pchp_minus_protected_baseline"]
    control = comparisons["pchp_minus_candidate_free_shift"]
    budget = comparisons["budget_path_auc_pchp_minus_shift"]
    for name, item in comparisons.items():
        audit.check(item["n_complete_domains"] == 12, f"V378 twelve-domain comparison: {name}")
        audit.check(item["sign_flip_assignments"] == 4096, f"V378 exhaustive sign assignments: {name}")
        audit.check(item["mean"] < 0, f"V378 negative domain-equal mean: {name}")
        audit.check(item["student_t_interval_95"][1] < 0, f"V378 Student-t interval upper bound: {name}")
    audit.check(primary["wins_ties_losses_for_pchp"] == [11, 0, 1], "V378 primary domain directions")
    audit.check(abs(primary["exact_two_sided_sign_flip_p_descriptive"] - 0.0009765625) <= 1e-15, "V378 primary exact sign-flip value")
    audit.check(control["wins_ties_losses_for_pchp"] == [9, 0, 3], "V378 candidate-control domain directions")
    audit.check(abs(control["exact_two_sided_sign_flip_p_descriptive"] - 0.0166015625) <= 1e-15, "V378 candidate-control exact sign-flip value")
    audit.check(abs(control["exact_two_sided_sign_p_descriptive"] - 0.14599609375) <= 1e-15, "V378 candidate-control direction-only boundary")
    audit.check(budget["wins_ties_losses_for_pchp"] == [11, 0, 1], "V378 budget-path domain directions")
    audit.check(abs(budget["exact_two_sided_sign_flip_p_descriptive"] - 0.0078125) <= 1e-15, "V378 budget-path exact sign-flip value")
    audit.check(all(report["gates"].values()), "V378 frozen sensitivity gates")
    report_path = ROOT / "small_n_domain_inference_v378_report.json"
    audit.check(
        sha256_file(report_path) == EXPECTED_V368_SHA256["small_n_domain_inference_v378_report.json"],
        "V378 deterministic statistical report identity",
    )


def verify_submission_contract(audit: prior.prior.prior.core.Audit) -> None:
    manuscript_root = ROOT / "paper_q1/rccp_causal_manuscript_v368"
    main_en = (manuscript_root / "main_en.tex").read_text(encoding="utf-8")
    main_zh = (manuscript_root / "main_zh.tex").read_text(encoding="utf-8")
    supplement_en = (manuscript_root / "supplement_en.tex").read_text(encoding="utf-8")
    supplement_zh = (manuscript_root / "supplement_zh.tex").read_text(encoding="utf-8")
    references = (manuscript_root / "references.bib").read_text(encoding="utf-8")
    figure_source = (manuscript_root / "make_figures_v335.py").read_text(encoding="utf-8")
    graphical_source = (manuscript_root / "make_graphical_abstract_v368.py").read_text(encoding="utf-8")
    cover_letter = (ROOT / "paper_q1/APPLIED_ENERGY_COVER_LETTER_V368.md").read_text(encoding="utf-8")
    highlights = [
        line.strip()
        for line in (manuscript_root / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    word_count = abstract_word_count(main_en)
    audit.check(240 <= word_count <= 250, "V368 Applied Energy abstract word limit")
    audit.check("Lithium-Ion Battery State-of-Health" in main_en, "V368 title names lithium-ion battery SOH")
    audit.check("Prefix-Causal Harm-Budget Projection" in main_en, "V368 title names the distinctive method")
    audit.check(len(highlights) == 5, "V368 highlight count")
    audit.check(max(map(len, highlights)) <= 85, "V368 highlight character limit")
    audit.check(
        highlights
        == [
            "PCHP admits cross-domain SOH updates before capacity labels are available.",
            "Exact projection bounds per-record harm and never rewrites issued SOH.",
            "Exact viability kernel joins harm budgets with irreversible SOH trajectories.",
            "Nested evaluation improves 11 of 12 domains under the harm budget.",
            "Outcome-blind external confirmation improves all 45 eligible cells.",
        ],
        "V373 highlight identities",
    )
    audit.check(list(map(len, highlights)) == [74, 70, 77, 66, 67], "V373 highlight lengths")
    audit.check("Declaration of generative AI" in main_en, "V368 English AI-use declaration present")
    audit.check("生成式人工智能" in main_zh, "V368 Chinese AI-use declaration present")
    audit.check("no generative image model was used" in main_en, "V368 programmatic-figure declaration present")
    audit.check("unbounded outcome space" not in main_en, "V368 English unbounded-space overclaim absent")
    audit.check("无界结果空间" not in main_zh, "V368 Chinese unbounded-space overclaim absent")
    audit.check("prospectively locked external validation on held-out public data" in main_en, "V368 English external-validation label present")
    audit.check("结果盲、前瞻锁定的外部验证" in main_zh, "V368 Chinese external-validation label present")
    audit.check("not a prospectively collected laboratory study or an independent-laboratory replication" in main_en, "V368 English higher-tier evidence boundary present")
    audit.check("不是前瞻采集的实验室研究，也不构成独立实验室重复" in main_zh, "V368 Chinese higher-tier evidence boundary present")
    audit.check("harm-control--accuracy" in main_en, "V368 English harm-control boundary present")
    audit.check("损害控制—精度" in main_zh, "V368 Chinese harm-control boundary present")
    audit.check("outcome-isolated, protocol-frozen public-data evaluation" not in main_en, "V368 obsolete English public-evaluation label absent")
    audit.check("retrospective external evidence" not in main_en, "V368 over-restrictive English retrospective label absent")
    audit.check("safety--accuracy opportunity cost" not in main_en, "V368 ambiguous English safety label absent")
    audit.check("结果隔离、协议冻结的公开数据评估" not in main_zh, "V368 obsolete Chinese public-evaluation label absent")
    audit.check("回顾性外部证据" not in main_zh, "V368 over-restrictive Chinese retrospective label absent")
    audit.check("安全—精度" not in main_zh, "V368 ambiguous Chinese safety label absent")
    audit.check("vaskov2024donoharm" in main_en and "vaskov2024donoharm" in main_zh, "V369 closest conceptual neighbor cited bilingually")
    audit.check("vaskov2024donoharm" in references, "V369 closest conceptual neighbor identity present")
    audit.check("vovk2020conformal" in main_en and "vovk2020conformal" in main_zh, "V370 conformal do-no-harm neighbor cited bilingually")
    audit.check("deb2025conservative" in main_en and "deb2025conservative" in main_zh, "V370 conservative-bandit neighbor cited bilingually")
    audit.check("vovk2020conformal" in references and "proceedings.mlr.press/v128/vovk20a.html" in references, "V370 conformal-neighbor identity present")
    audit.check("deb2025conservative" in references and "dbca58f35bddc6e4003b2dd80e42f838" in references, "V370 conservative-bandit identity present")
    audit.check("Baseline-relative / constrained online decisions" in main_en and "相对基线 / 受约束在线决策" in main_zh, "V376 closest-policy row present bilingually")
    audit.check("Battery-specific operational meaning" in main_en and "面向电池运行的信息含义" in main_zh, "V371 battery-specific operational meaning present bilingually")
    audit.check("The confirmatory status of the BaSyTec analysis follows from design independence" in main_en, "V371 English design-independence confirmation boundary present")
    audit.check("BaSyTec 分析的确认属性来自设计独立性" in main_zh, "V371 Chinese design-independence confirmation boundary present")
    audit.check("new validation surface whose outcomes were unavailable across the team and project lineage" not in main_en, "V371 over-restrictive English validation requirement absent")
    audit.check("团队及项目谱系中结果均不可得的新验证面" not in main_zh, "V371 over-restrictive Chinese validation requirement absent")
    audit.check("partial charging records before reference capacity labels" in main_en, "V372 English abstract opens with battery information mismatch")
    audit.check("this intersection is the exact online viability kernel" in main_en and "unique closest-candidate solution" in main_en, "V373 English abstract states exact method contribution")
    audit.check("outcome-blind, protocol-locked external confirmation" in main_en, "V372 English abstract uses design-appropriate confirmation label")
    audit.check("部分充电记录，参考容量标签随后才可用" in main_zh, "V372 Chinese abstract opens with battery information mismatch")
    audit.check("该交集构成精确在线可行核" in main_zh and "核内唯一最接近候选的解" in main_zh, "V373 Chinese abstract states exact method contribution")
    audit.check("结果盲、协议锁定的外部确认" in main_zh, "V372 Chinese abstract uses design-appropriate confirmation label")
    audit.check("Exact online viability kernel" in main_en and "cor:viability_kernel" in main_en, "V373 English exact online viability theorem present")
    audit.check("精确在线可行核" in main_zh and "cor:viability_kernel_zh" in main_zh, "V373 Chinese exact online viability theorem present")
    audit.check("every admitted output remains extendable" in main_en, "V373 English viability interpretation present")
    audit.check("每个输出均可在未来继续执行" in main_zh, "V373 Chinese viability interpretation present")
    audit.check("wabersich2021predictive" in main_en and "wabersich2021predictive" in main_zh, "V373 predictive-safety-filter neighbor cited bilingually")
    audit.check("wabersich2021predictive" in references and "10.1016/j.automatica.2021.109597" in references, "V373 predictive-safety-filter source identity present")
    audit.check("Predictive safety filters" in main_en and "预测安全过滤器" in main_zh, "V373 predictive-safety-filter closest-method row present bilingually")
    audit.check("Its lower endpoint is non-increasing with $b_{it}$" not in main_en, "V373 misleading English asymmetric-proof wording absent")
    audit.check("该损害域的下端点关于 $b_{it}$ 非增" not in main_zh, "V373 misleading Chinese asymmetric-proof wording absent")
    audit.check("The output contract transfers across untuned secondary backbones" in main_en, "V375 English secondary-backbone result present")
    audit.check("输出契约可迁移至未额外调参的次级骨干" in main_zh, "V375 Chinese secondary-backbone result present")
    audit.check("Bonferroni-adjusted $98.33\\%$ intervals" in main_en, "V375 English multiplicity-adjusted interval present")
    audit.check("Bonferroni 调整后的 $98.33\\%$ 区间" in main_zh, "V375 Chinese multiplicity-adjusted interval present")
    audit.check("does not establish universal model independence, deep-model portability, or best-backbone superiority" in main_en, "V375 English portability boundary present")
    audit.check("不证明对所有模型普遍独立，也不证明深度模型可迁移或某个次级骨干更优" in main_zh, "V375 Chinese portability boundary present")
    for key in ("bernasconi2021conservative", "moradipari2020stagewise", "sridharan2025unknown"):
        audit.check(key in main_en and key in main_zh, f"V376 direct-neighbor citation used bilingually: {key}")
        audit.check(key in references, f"V376 direct-neighbor identity present: {key}")
    audit.check("cumulative realized loss stays within a multiplicative allowance" in main_en, "V376 English conservative-OCO distinction present")
    audit.check("同一记录上对所有可能结果成立的损害域" in main_zh, "V376 Chinese outcome-uniform distinction present")
    for key in ("samar2004moving", "gorinevsky2008efficient"):
        audit.check(key in main_en and key in main_zh, f"V377 monotone-filter stability prior cited bilingually: {key}")
        audit.check(key in references, f"V377 monotone-filter stability prior identity present: {key}")
    audit.check("Prefix-level nonexpansiveness of the complete operator" in main_en and "thm:nonexpansive" in main_en, "V377 English stability theorem present")
    audit.check("完整算子的前缀非扩张性" in main_zh and "thm:nonexpansive_zh" in main_zh, "V377 Chinese stability theorem present")
    audit.check("incremental algorithmic-stability statement" in main_en, "V377 English stability limitation present")
    audit.check("增量式算法稳定性结论" in main_zh, "V377 Chinese stability limitation present")
    audit.check("does not claim the first stable monotone filter" in main_en, "V377 English priority restraint present")
    audit.check("不声称首次提出稳定的单调滤波器" in main_zh, "V377 Chinese priority restraint present")
    audit.check("Untuned secondary-backbone portability audit" in supplement_en, "V375 English supplementary audit present")
    audit.check("未额外调参的次级骨干可移植性审计" in supplement_zh, "V375 Chinese supplementary audit present")
    audit.check("giving $34$ focused PCHP tests" in supplement_en and "合计为 $34$ 项聚焦 PCHP 测试" in supplement_zh, "V377 bilingual unit-test count synchronized")
    audit.check("Prefix nonexpansiveness falsification record" in supplement_en, "V377 English supplementary falsification record present")
    audit.check("前缀非扩张性证伪记录" in supplement_zh, "V377 Chinese supplementary falsification record present")
    audit.check("clearly post-hoc finite-sample sensitivity" in main_en, "V378 English finite-sample chronology present")
    audit.check("明确标注为事后的有限样本敏感性分析" in main_zh, "V378 Chinese finite-sample chronology present")
    audit.check("domain-equal mean utility contribution with heterogeneous domain effects" in main_en, "V378 English estimand boundary present")
    audit.check("域等权平均效用贡献" in main_zh and "随机抽取一个新域时改善概率大于一半" in main_zh, "V378 Chinese estimand boundary present")
    audit.check("Finite-sample domain sensitivity audit" in supplement_en, "V378 English supplementary audit present")
    audit.check("有限样本域级敏感性审计" in supplement_zh, "V378 Chinese supplementary audit present")
    audit.check((ROOT / "paper_q1/TOP_Q1_SMALL_N_STATISTICAL_STRESS_AUDIT_V378_ZH.md").is_file(), "V378 statistical stress audit present")
    for key, doi in {
        "liu2023rapid": "10.1016/j.apenergy.2023.121925",
        "zhang2024flexible": "10.1016/j.energy.2024.131009",
    }.items():
        audit.check(key in main_en and key in main_zh, f"V371 battery-operational citation used bilingually: {key}")
        audit.check(key in references and doi in references, f"V371 battery-operational source identity: {key}")
    audit.check("candidate-free adaptive-value control" in main_en, "V369 Figure 1 caption carries candidate-control evidence")
    audit.check("候选无关适应性价值对照" in main_zh, "V369 Chinese Figure 1 caption carries candidate-control evidence")
    audit.check("External validation" in figure_source and "basytec_body" in figure_source, "V369 Figure 1 carries BaSyTec external-validation evidence")
    audit.check((ROOT / "paper_q1/TOP_Q1_EDITOR_METHOD_STAT_NOVELTY_AUDIT_V369_ZH.md").is_file(), "V369 editor method-statistics-novelty audit present")
    audit.check((ROOT / "paper_q1/TOP_Q1_ADVERSARIAL_EDITOR_AUDIT_V370_ZH.md").is_file(), "V370 adversarial editor audit present")
    audit.check((ROOT / "paper_q1/EXTERNAL_CONFIRMATION_AND_ENERGY_RELEVANCE_AUDIT_V371_ZH.md").is_file(), "V371 external-confirmation and energy-relevance audit present")
    audit.check((ROOT / "paper_q1/APPLIED_ENERGY_EDITOR_FIRST_IMPRESSION_AUDIT_V372_ZH.md").is_file(), "V372 editor first-impression audit present")
    audit.check((ROOT / "paper_q1/TOP_Q1_EXACT_VIABILITY_AND_CLOSEST_NEIGHBOR_AUDIT_V373_ZH.md").is_file(), "V373 exact-viability and closest-neighbor audit present")
    audit.check("Operational SOH gap" in graphical_source, "V372 graphical abstract names the battery operational gap")
    audit.check("Outcome-blind external confirmation" in graphical_source, "V372 graphical abstract names the confirmation design")
    audit.check("Risk-bounded SOH updates" in graphical_source and "before reference capacity is available" in graphical_source, "V372 graphical abstract closes the problem-value loop")
    audit.check("Exact online viability set" in graphical_source and "closest viable output" in graphical_source, "V373 graphical abstract states viability contribution")
    audit.check("The central innovation is a recursively feasible prediction contract" in cover_letter, "V372 cover letter states the central innovation")
    audit.check("partial charging records can arrive" in cover_letter and "reference-capacity label is available" in cover_letter, "V372 cover letter states the battery information mismatch")
    audit.check("the exact online viability kernel" in cover_letter and "unique closest-candidate solution" in cover_letter, "V373 cover letter states exact viability and optimality")
    audit.check("non-increasing physical trajectory constraint" not in cover_letter, "V372 cover letter avoids overstating monotonicity as universal physics")
    unit_tests = (ROOT / "test_prefix_causal_harm_projection_v321.py").read_text(encoding="utf-8")
    v377_unit_tests = (ROOT / "test_prefix_nonexpansiveness_v377.py").read_text(encoding="utf-8")
    audit.check("test_every_current_viability_point_has_a_future_continuation" in unit_tests, "V373 future-continuation property test present")
    audit.check("test_every_point_outside_current_viability_interval_violates_contract" in unit_tests, "V373 maximal-kernel property test present")
    audit.check("test_protected_state_update_has_exact_local_proximal_form" in v377_unit_tests, "V377 exact local proximal-form test present")
    audit.check("test_protected_state_prefix_map_is_sup_norm_nonexpansive" in v377_unit_tests, "V377 protected-state nonexpansiveness test present")
    audit.check("test_full_pchp_prefix_map_is_sup_norm_nonexpansive" in v377_unit_tests, "V377 complete-operator nonexpansiveness test present")
    audit.check("test_full_pchp_nonexpansive_constant_is_tight" in v377_unit_tests, "V377 tightness test present")
    for key, doi in {
        "qiu2024multisource": "10.1016/j.apenergy.2024.124245",
        "zhang2025bayesian": "10.1016/j.apenergy.2024.125260",
        "hadzalic2025field": "10.1016/j.egyai.2025.100575",
    }.items():
        audit.check(key in main_en and key in main_zh, f"V368 concurrent-work citation used: {key}")
        audit.check(key in references and doi in references, f"V368 concurrent-work identity: {key}")
    png = manuscript_root / "graphical_abstract_applied_energy_v368.png"
    audit.check(png_dimensions(png) == (2340, 900), "V368 graphical abstract dimensions")
    audit.check(
        sha256_file(png) == EXPECTED_V368_SHA256["paper_q1/rccp_causal_manuscript_v368/graphical_abstract_applied_energy_v368.png"],
        "V368 graphical abstract PNG identity",
    )
    audit.check((manuscript_root / "graphical_abstract_applied_energy_v368.pdf").is_file(), "V368 graphical abstract PDF exists")
    checklist = (ROOT / "paper_q1/APPLIED_ENERGY_SUBMISSION_CHECKLIST_V368_ZH.md").read_text(encoding="utf-8")
    audit.check("OPEN_COMPLIANCE" in checklist and "OPEN_SCIENTIFIC" in checklist, "V368 open submission gates retained")


def regenerate_and_verify_figures(audit: prior.prior.prior.core.Audit) -> None:
    manuscript_root = ROOT / "paper_q1/rccp_causal_manuscript_v368"
    for script_name in (
        "make_figures_v335.py",
        "make_budget_path_figure_v363.py",
        "make_basytec_figure_v359.py",
    ):
        script = manuscript_root / script_name
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        audit.check(completed.returncode == 0, f"V368 figure generator executed: {script_name}")
    figure_root = manuscript_root / "figures"
    for name, expected in EXPECTED_V368_FIGURE_PNG_SHA256.items():
        path = figure_root / name
        audit.check(path.is_file(), f"V368 regenerated manuscript figure exists: {name}")
        audit.check(sha256_file(path) == expected, f"V368 regenerated manuscript figure hash: {name}")

    script = manuscript_root / "make_graphical_abstract_v368.py"
    with tempfile.TemporaryDirectory(prefix="pchp_v368_graphical_abstract_") as tmp:
        completed = subprocess.run(
            [sys.executable, str(script), "--output-dir", tmp],
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        audit.check(completed.returncode == 0, "V368 graphical abstract generator executed")
        png = Path(tmp) / "graphical_abstract_applied_energy_v368.png"
        pdf = Path(tmp) / "graphical_abstract_applied_energy_v368.pdf"
        audit.check(png.is_file(), "V368 regenerated graphical abstract PNG exists")
        audit.check(pdf.is_file(), "V368 regenerated graphical abstract PDF exists")
        audit.check(png_dimensions(png) == (2340, 900), "V368 regenerated graphical abstract dimensions")
        audit.check(
            sha256_file(png) == EXPECTED_V368_SHA256["paper_q1/rccp_causal_manuscript_v368/graphical_abstract_applied_energy_v368.png"],
            "V368 regenerated graphical abstract PNG hash",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = prior.prior.prior.core.Audit()
    verify_manifest(audit)
    verify_inherited_science(audit)
    verify_v368_pinned_artifacts(audit)
    verify_v374_rejected_extension(audit)
    verify_v375_backbone_portability(audit)
    verify_v377_prefix_nonexpansiveness(audit)
    verify_v378_small_n_domain_inference(audit)
    verify_submission_contract(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_V368_EXTERNAL_VALIDATION_REVIEW_LITE_VERIFICATION_PASSED",
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
            "The package verifies the inherited V366 scientific evidence and the V368 Applied Energy "
            "manuscript, external-validation labels, abstract, highlights, concurrent-work identities, "
            "V369--V378 method-statistics-novelty, closest-neighbor, external-confirmation, energy-relevance, editor-first-impression, exact-viability, rejected-recovery, untuned-backbone, design-independence, operator-stability, and finite-sample audits, "
            "AI-use declaration, "
            "manuscript figures, and graphical abstract. BaSyTec qualifies as outcome-blind, "
            "prospectively locked external validation because the held-out outcomes did not affect the method, "
            "hyperparameters, estimand, exclusions, or decision gates and predictions were frozen before capacity "
            "access. Confirmatory status follows from that design independence rather than new data collection or "
            "dataset count. V375 is retrospective development robustness, not an additional external confirmation. "
            "V377 establishes prefix-level incremental algorithmic stability, not accuracy, calibration, uncertainty, or physical safety. "
            "V378 is a post-hoc finite-sample sensitivity on twelve opened development domains; it supports domain-equal mean effects, not universal or majority-probability dominance. "
            "It is not a prospectively collected study or independent-laboratory replication. The package "
            "does not establish journal acceptance, institutional Q1/Top classification, author or affiliation "
            "facts, electrochemical safety, code licensing, or deployment-cost calibration."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v368.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
