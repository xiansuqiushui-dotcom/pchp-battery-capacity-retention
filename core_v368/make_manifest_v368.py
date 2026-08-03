"""Generate the deterministic V368 evidence-chronology review-lite manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest_v368.json"
EXCLUDED_NAMES = {MANIFEST.name, "verification_receipt_v368.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.name in EXCLUDED_NAMES:
        return False
    if "__pycache__" in relative.parts or "figures" in relative.parts:
        return False
    return True


def main() -> None:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(ROOT.rglob("*"))
        if tracked(path)
    ]
    payload = {
        "package": "PCHP reproducibility review-lite",
        "version": "v368",
        "scientific_evidence_inherited_from_v366": True,
        "contains_applied_energy_submission_assets": True,
        "contains_v368_venue_and_concurrent_work_audit": True,
        "contains_v368_method_and_evidence_audit": True,
        "contains_v369_editor_method_statistics_novelty_audit": True,
        "contains_v370_adversarial_editor_audit": True,
        "contains_v371_external_confirmation_energy_relevance_audit": True,
        "contains_v372_editor_first_impression_audit": True,
        "contains_v373_exact_viability_and_closest_neighbor_audit": True,
        "contains_v374_rejected_bounded_recovery_audit": True,
        "contains_v375_untuned_backbone_portability_audit": True,
        "contains_v376_novelty_and_reviewer_risk_audit": True,
        "contains_v377_method_elevation_and_stability_audit": True,
        "contains_prefix_nonexpansiveness_theorem_and_falsification": True,
        "v377_stability_is_algorithmic_not_accuracy_or_physical_safety": True,
        "contains_v378_small_n_domain_inference_audit": True,
        "v378_development_sensitivity_is_posthoc_not_confirmation": True,
        "v378_candidate_control_supports_mean_not_universal_or_majority": True,
        "v375_backbone_portability_is_retrospective_not_external_confirmation": True,
        "contains_v368_bilingual_manuscript": True,
        "contains_v368_graphical_abstract": True,
        "contains_v368_ai_use_declaration": True,
        "contains_outcome_blind_prospectively_locked_external_validation": True,
        "does_not_claim_prospective_data_collection_or_independent_lab_replication": True,
        "external_confirmation_uses_design_independence_not_absolute_lineage": True,
        "contains_claim_source_and_novelty_audit": True,
        "contains_loss_geometry_extension": True,
        "contains_theory_implementation_contract": True,
        "squared_loss_full_real_line_scope_only": True,
        "raw_third_party_archives_included": False,
        "contains_outcome_isolated_basytec_aggregate_evidence": True,
        "contains_protocol_locked_retrospective_budget_path": True,
        "retains_original_v361_narrow_record": True,
        "tracked_files": len(files),
        "files": files,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(MANIFEST), "tracked_files": len(files)}))


if __name__ == "__main__":
    main()
