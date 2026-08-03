"""Generate the deterministic review-lite package manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest_v341.json"
EXCLUDED_NAMES = {MANIFEST.name, "verification_receipt_v341.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path.name in EXCLUDED_NAMES or "__pycache__" in relative.parts:
        return False
    if relative.parts[-2:-1] == ("figures",):
        return False
    return path.is_file()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not tracked(path):
            continue
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "package": "PCHP reproducibility review-lite",
        "version": "v341",
        "raw_third_party_archives_included": False,
        "tracked_files": len(files),
        "files": files,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(MANIFEST), "tracked_files": len(files)}))


if __name__ == "__main__":
    main()
