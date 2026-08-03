"""Failure-closed verifier for the PCHP V367 Applied Energy review-lite package."""

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
EXCLUDED_MANIFEST_NAMES = {"manifest_v367.json", "verification_receipt_v367.json"}

EXPECTED_V367_SHA256 = {
    "paper_q1/rccp_causal_manuscript_v367/main_en.tex": "625200452c211bc5263c90863e3d4f411554eab0a84d0e7d1a9d453c26c79d78",
    "paper_q1/rccp_causal_manuscript_v367/main_zh.tex": "45b5c910f4320fa8e0660495f8a6ffed2b93ed839b4f28ec9e98c85dd07a962f",
    "paper_q1/rccp_causal_manuscript_v367/supplement_en.tex": "4e79a42a5a1041758dae326a0b29b151543960b18ff553b64a642933a18e8c69",
    "paper_q1/rccp_causal_manuscript_v367/supplement_zh.tex": "32bf8876cd9b888bbe2a522b53a1ec44f0e6dd18c55901c0b278d6b00dffc43e",
    "paper_q1/rccp_causal_manuscript_v367/references.bib": "d75320c117d3b820200092e59db82c77402620fd7b4b627d66bd1da4f98c1d99",
    "paper_q1/rccp_causal_manuscript_v367/highlights.txt": "911fd28f23af444d87235e55c6ad51413179ad05766f91f9795b4505cfadf80d",
    "paper_q1/rccp_causal_manuscript_v367/make_graphical_abstract_v367.py": "372ad211536b436d6a5a1768cb9b31cc4004a6961baab88eece2bf127eab3ec2",
    "paper_q1/rccp_causal_manuscript_v367/graphical_abstract_applied_energy_v367.png": "70dd7cd92a247864da8d8d3761d6c00f44af3c1b5a9eafcf2e9bcd34a5c2a32a",
    "paper_q1/VENUE_ROUTE_AND_CONCURRENT_WORK_AUDIT_V367_ZH.md": "0d0acbfa330f27bdec760d7866a12a7eee6c25859ee6766e8867a891051899de",
    "paper_q1/APPLIED_ENERGY_SUBMISSION_COMPONENTS_V367.md": "c82d53bc50c9345e16d637585580486d3d361fe77dd162e37056fdbbb379f20a",
    "paper_q1/APPLIED_ENERGY_COVER_LETTER_V367.md": "3030a0ac53d738904b0ac16a0bc0baa9982ccc80ea9353df9cae7080bdbd706b",
    "paper_q1/APPLIED_ENERGY_SUBMISSION_CHECKLIST_V367_ZH.md": "27426efb8c41b86257130b23a3a413c2a00d1322332e3713badf17980420659b",
    "paper_q1/RCCP_CAUSAL_PROJECT_HANDOFF_V367_ZH.md": "e945c917dcc187438510882c9041389e8a30dccc46763703a441a00cececca82",
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
    manifest = load_json("manifest_v367.json")
    audit.check(manifest["version"] == "v367", "V367 manifest version")
    audit.check(manifest["scientific_evidence_inherited_from_v366"] is True, "V367 inherits frozen V366 science")
    audit.check(manifest["contains_applied_energy_submission_assets"] is True, "V367 submission assets declared")
    audit.check(manifest["contains_v367_venue_and_concurrent_work_audit"] is True, "V367 venue audit declared")
    audit.check(manifest["contains_v367_bilingual_manuscript"] is True, "V367 bilingual manuscript declared")
    audit.check(manifest["contains_v367_graphical_abstract"] is True, "V367 graphical abstract declared")
    audit.check(manifest["contains_v367_ai_use_declaration"] is True, "V367 AI declaration declared")
    audit.check(manifest["raw_third_party_archives_included"] is False, "V367 raw-data boundary")
    expected_paths = {item["path"] for item in manifest["files"]}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if tracked(path)
    }
    audit.check(actual_paths == expected_paths, "V367 manifest has no untracked scientific files")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        audit.check(path.is_file(), f"V367 manifest file exists: {item['path']}")
        audit.check(path.stat().st_size == item["bytes"], f"V367 manifest size: {item['path']}")
        audit.check(sha256_file(path) == item["sha256"], f"V367 manifest hash: {item['path']}")
    audit.check(manifest["tracked_files"] == len(manifest["files"]), "V367 manifest file count")


def verify_inherited_science(audit: prior.prior.prior.core.Audit) -> None:
    core = prior.prior.prior.core
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


def verify_v367_pinned_artifacts(audit: prior.prior.prior.core.Audit) -> None:
    for relative, expected in EXPECTED_V367_SHA256.items():
        path = ROOT / relative
        audit.check(path.is_file(), f"V367 pinned artifact exists: {relative}")
        audit.check(sha256_file(path) == expected, f"V367 pinned artifact hash: {relative}")


def verify_submission_contract(audit: prior.prior.prior.core.Audit) -> None:
    manuscript_root = ROOT / "paper_q1/rccp_causal_manuscript_v367"
    main_en = (manuscript_root / "main_en.tex").read_text(encoding="utf-8")
    main_zh = (manuscript_root / "main_zh.tex").read_text(encoding="utf-8")
    references = (manuscript_root / "references.bib").read_text(encoding="utf-8")
    highlights = [
        line.strip()
        for line in (manuscript_root / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    word_count = abstract_word_count(main_en)
    audit.check(240 <= word_count <= 250, "V367 Applied Energy abstract word limit")
    audit.check("Lithium-Ion Battery State-of-Health" in main_en, "V367 title names lithium-ion battery SOH")
    audit.check("Prefix-Causal Harm-Budget Projection" in main_en, "V367 title names the distinctive method")
    audit.check(len(highlights) == 5, "V367 highlight count")
    audit.check(max(map(len, highlights)) <= 85, "V367 highlight character limit")
    audit.check("Declaration of generative AI" in main_en, "V367 English AI-use declaration present")
    audit.check("生成式人工智能" in main_zh, "V367 Chinese AI-use declaration present")
    audit.check("no generative image model was used" in main_en, "V367 programmatic-figure declaration present")
    audit.check("unbounded outcome space" not in main_en, "V367 English unbounded-space overclaim absent")
    audit.check("无界结果空间" not in main_zh, "V367 Chinese unbounded-space overclaim absent")
    for key, doi in {
        "qiu2024multisource": "10.1016/j.apenergy.2024.124245",
        "zhang2025bayesian": "10.1016/j.apenergy.2024.125260",
        "hadzalic2025field": "10.1016/j.egyai.2025.100575",
    }.items():
        audit.check(key in main_en and key in main_zh, f"V367 concurrent-work citation used: {key}")
        audit.check(key in references and doi in references, f"V367 concurrent-work identity: {key}")
    png = manuscript_root / "graphical_abstract_applied_energy_v367.png"
    audit.check(png_dimensions(png) == (2340, 900), "V367 graphical abstract dimensions")
    audit.check(
        sha256_file(png) == EXPECTED_V367_SHA256["paper_q1/rccp_causal_manuscript_v367/graphical_abstract_applied_energy_v367.png"],
        "V367 graphical abstract PNG identity",
    )
    audit.check((manuscript_root / "graphical_abstract_applied_energy_v367.pdf").is_file(), "V367 graphical abstract PDF exists")
    checklist = (ROOT / "paper_q1/APPLIED_ENERGY_SUBMISSION_CHECKLIST_V367_ZH.md").read_text(encoding="utf-8")
    audit.check("OPEN_COMPLIANCE" in checklist and "OPEN_SCIENTIFIC" in checklist, "V367 open submission gates retained")


def regenerate_and_verify_figures(audit: prior.prior.prior.core.Audit) -> None:
    prior.regenerate_and_verify_figures(audit)
    script = ROOT / "paper_q1/rccp_causal_manuscript_v367/make_graphical_abstract_v367.py"
    with tempfile.TemporaryDirectory(prefix="pchp_v367_graphical_abstract_") as tmp:
        completed = subprocess.run(
            [sys.executable, str(script), "--output-dir", tmp],
            cwd=script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        audit.check(completed.returncode == 0, "V367 graphical abstract generator executed")
        png = Path(tmp) / "graphical_abstract_applied_energy_v367.png"
        pdf = Path(tmp) / "graphical_abstract_applied_energy_v367.pdf"
        audit.check(png.is_file(), "V367 regenerated graphical abstract PNG exists")
        audit.check(pdf.is_file(), "V367 regenerated graphical abstract PDF exists")
        audit.check(png_dimensions(png) == (2340, 900), "V367 regenerated graphical abstract dimensions")
        audit.check(
            sha256_file(png) == EXPECTED_V367_SHA256["paper_q1/rccp_causal_manuscript_v367/graphical_abstract_applied_energy_v367.png"],
            "V367 regenerated graphical abstract PNG hash",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--regenerate-figures", action="store_true")
    args = parser.parse_args()
    audit = prior.prior.prior.core.Audit()
    verify_manifest(audit)
    verify_inherited_science(audit)
    verify_v367_pinned_artifacts(audit)
    verify_submission_contract(audit)
    if args.regenerate_figures:
        regenerate_and_verify_figures(audit)
    receipt = {
        "status": "PCHP_V367_APPLIED_ENERGY_REVIEW_LITE_VERIFICATION_PASSED",
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
            "The package verifies the inherited V366 scientific evidence and the V367 Applied Energy "
            "manuscript, abstract, highlights, concurrent-work identities, AI-use declaration, and graphical "
            "abstract. It does not establish journal acceptance, institutional Q1/Top classification, author "
            "or affiliation facts, a prospective laboratory trial, electrochemical safety, code licensing, "
            "or deployment-cost calibration."
        ),
    }
    if args.write_receipt:
        (ROOT / "verification_receipt_v367.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
