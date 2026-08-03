"""Resume, verify, and freeze selected BatteryLife Zenodo archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from audit_batterylife_zenodo_v109 import (
    DEFAULT_MANIFEST,
    ROOT,
    STAGE_ONE_FILES,
    retrying_session,
)


DEFAULT_OUTPUT_DIR = ROOT / "public_data_batterylife_v109" / "archives"
DEFAULT_RESULT = ROOT / "batterylife_zenodo_v109_download.json"


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
    *,
    key: str,
    url: str,
    expected_bytes: int,
    checksum_algorithm: str,
    checksum_digest: str,
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / key
    partial_path = output_dir / f"{key}.part"
    if final_path.exists():
        observed_bytes = final_path.stat().st_size
        observed_digest = digest_file(final_path, checksum_algorithm)
        if (
            observed_bytes != expected_bytes
            or observed_digest.lower() != checksum_digest.lower()
        ):
            raise RuntimeError(f"existing archive failed checksum: {final_path}")
        return {
            "key": key,
            "path": str(final_path.resolve()),
            "bytes": int(observed_bytes),
            "checksum_algorithm": checksum_algorithm,
            "checksum_digest": observed_digest,
            "downloaded_this_run": False,
            "resumed_from_bytes": int(observed_bytes),
        }

    resumed_from = partial_path.stat().st_size if partial_path.exists() else 0
    if resumed_from > expected_bytes:
        raise RuntimeError(f"partial archive exceeds expected size: {partial_path}")
    headers = {"Range": f"bytes={resumed_from}-"} if resumed_from else {}
    session = retrying_session()
    response = session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(timeout_seconds, timeout_seconds),
    )
    response.raise_for_status()
    if resumed_from and response.status_code == 206:
        mode = "ab"
    elif resumed_from and response.status_code == 200:
        mode = "wb"
        resumed_from = 0
    elif not resumed_from and response.status_code == 200:
        mode = "wb"
    else:
        raise RuntimeError(
            f"unexpected HTTP status {response.status_code} for {key}"
        )

    with partial_path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 << 20):
            if chunk:
                handle.write(chunk)
    observed_bytes = partial_path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"{key}: expected {expected_bytes} bytes, observed {observed_bytes}"
        )
    observed_digest = digest_file(partial_path, checksum_algorithm)
    if observed_digest.lower() != checksum_digest.lower():
        raise RuntimeError(
            f"{key}: checksum mismatch; expected {checksum_digest}, "
            f"observed {observed_digest}"
        )
    os.replace(partial_path, final_path)
    return {
        "key": key,
        "path": str(final_path.resolve()),
        "bytes": int(observed_bytes),
        "checksum_algorithm": checksum_algorithm,
        "checksum_digest": observed_digest,
        "downloaded_this_run": True,
        "resumed_from_bytes": int(resumed_from),
    }


def run_download(
    manifest_path: Path,
    output_dir: Path,
    *,
    selected_files: Sequence[str],
    timeout_seconds: float,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    selected = tuple(dict.fromkeys(map(str, selected_files)))
    missing = sorted(set(selected) - set(manifest["key"]))
    if missing:
        raise ValueError(f"selected files are absent from manifest: {missing}")
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for key in selected:
        row = manifest.loc[manifest["key"] == key]
        if len(row) != 1:
            raise RuntimeError(f"manifest key is nonunique: {key}")
        item = row.iloc[0]
        record = download_one(
            key=key,
            url=str(item["download_url"]),
            expected_bytes=int(item["bytes"]),
            checksum_algorithm=str(item["checksum_algorithm"]),
            checksum_digest=str(item["checksum_digest"]),
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        records.append(record)
        print(
            f"verified {key}: {record['bytes']} bytes "
            f"({record['checksum_algorithm']}:{record['checksum_digest']})",
            flush=True,
        )
    return {
        "status": "STAGE_ONE_ARCHIVES_DOWNLOADED_AND_CHECKSUM_VERIFIED",
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_files": list(selected),
        "files": records,
        "total_bytes": int(sum(int(row["bytes"]) for row in records)),
        "all_checksums_verified": True,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(STAGE_ONE_FILES),
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_download(
        args.manifest,
        args.output_dir,
        selected_files=args.files,
        timeout_seconds=args.timeout_seconds,
    )
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
