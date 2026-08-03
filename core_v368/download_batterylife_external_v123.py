"""Resume-safe downloader for frozen external BatteryLife v11 domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent
RECORD_ID = "19688272"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DEFAULT_ARCHIVE_DIR = ROOT / "public_data_batterylife_v109" / "archives"
DEFAULT_JSON = ROOT / "batterylife_external_v123_download_results.json"
DEFAULT_DOMAINS = ("HUST", "RWTH")


def retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _md5(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _record_files(session: requests.Session) -> dict[str, dict[str, object]]:
    response = session.get(API_URL, timeout=60)
    response.raise_for_status()
    record = response.json()
    files = record.get("files")
    if not isinstance(files, list):
        raise ValueError("Zenodo record has no file list")
    return {str(item["key"]): item for item in files}


def download_file(
    session: requests.Session,
    *,
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_md5: str,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"existing file has wrong size: {destination}")
        observed_md5 = _md5(destination)
        if observed_md5 != expected_md5:
            raise RuntimeError(f"existing file has wrong checksum: {destination}")
        return {
            "path": str(destination.resolve()),
            "bytes": int(expected_bytes),
            "md5": observed_md5,
            "downloaded_now": False,
        }

    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_bytes:
        raise RuntimeError(f"partial file is larger than expected: {partial}")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(60, 300),
    ) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            raise RuntimeError("server ignored byte-range resume request")
        mode = "ab" if existing else "wb"
        downloaded = existing
        next_report = downloaded + 128 * 1024 * 1024
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(
                        json.dumps(
                            {
                                "file": destination.name,
                                "downloaded_bytes": downloaded,
                                "expected_bytes": expected_bytes,
                                "fraction": downloaded / expected_bytes,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    next_report += 128 * 1024 * 1024
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"download size mismatch for {destination.name}: "
            f"{partial.stat().st_size} != {expected_bytes}"
        )
    observed_md5 = _md5(partial)
    if observed_md5 != expected_md5:
        raise RuntimeError(
            f"checksum mismatch for {destination.name}: "
            f"{observed_md5} != {expected_md5}"
        )
    partial.replace(destination)
    return {
        "path": str(destination.resolve()),
        "bytes": int(expected_bytes),
        "md5": observed_md5,
        "downloaded_now": True,
    }


def run_download(
    archive_dir: Path,
    *,
    domains: Sequence[str] = DEFAULT_DOMAINS,
) -> dict[str, object]:
    started = time.perf_counter()
    frozen_domains = tuple(dict.fromkeys(str(domain) for domain in domains))
    if not frozen_domains:
        raise ValueError("at least one external domain is required")
    session = retrying_session()
    record_files = _record_files(session)
    downloaded: list[dict[str, object]] = []
    for domain in frozen_domains:
        key = f"{domain}.zip"
        if key not in record_files:
            raise KeyError(f"{key} is absent from BatteryLife v11")
        item = record_files[key]
        checksum = str(item["checksum"])
        algorithm, digest = checksum.split(":", maxsplit=1)
        if algorithm.lower() != "md5":
            raise ValueError(f"unsupported checksum for {key}: {checksum}")
        links = item["links"]
        if not isinstance(links, dict):
            raise ValueError(f"missing download link for {key}")
        result = download_file(
            session,
            url=str(links["self"]),
            destination=archive_dir / key,
            expected_bytes=int(item["size"]),
            expected_md5=digest.lower(),
        )
        downloaded.append({"domain": domain, "zenodo_key": key, **result})
    return {
        "status": "FROZEN_EXTERNAL_DOMAINS_DOWNLOADED_V123",
        "record_id": RECORD_ID,
        "record_url": f"https://zenodo.org/records/{RECORD_ID}",
        "version": "BatteryLife v11",
        "domains": list(frozen_domains),
        "files": downloaded,
        "scientific_role": (
            "previously unused external battery domains; not used to choose "
            "the V119 method structure or V120 acquisition baselines"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_download(args.archive_dir, domains=args.domains)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
