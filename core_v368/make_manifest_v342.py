"""Generate the deterministic V342 review-lite package manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest_v342.json"
EXCLUDED_NAMES = {MANIFEST.name, "verification_receipt_v342.json"}


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
        "version": "v342",
        "raw_third_party_archives_included": False,
        "contains_outcome_isolated_basytec_aggregate_evidence": True,
        "tracked_files": len(files),
        "files": files,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(MANIFEST), "tracked_files": len(files)}))


if __name__ == "__main__":
    main()
