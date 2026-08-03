"""Create the deterministic V391 release manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "manifest_v391.json"
EXCLUDED_NAMES = {OUTPUT.name, "verification_receipt_v391.json"}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".spl",
    ".synctex",
    ".xdv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in {".pyc", ".pyo", *EXCLUDED_SUFFIXES}:
            continue
        if path.name.endswith(".synctex.gz"):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in tracked_files()
    ]
    manifest = {
        "version": "v391",
        "release_date": "2026-08-03",
        "title": "Risk-Controlled Online Updating of Cross-Domain Lithium-Ion Battery Capacity-Retention Estimates via Prefix-Causal Projection",
        "license_for_author_created_software": "MIT",
        "raw_third_party_archives_included": False,
        "contains_final_bilingual_manuscript": True,
        "contains_final_external_mechanism_confirmation": True,
        "contains_final_statistics_and_reviewer_risk_audits": True,
        "tracked_files": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT.name}: {len(records)} files, {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
