"""Run label-blind predictions after freezing the complete header family."""

from __future__ import annotations

import json
from pathlib import Path

import basytec_external_v352_common as common
import build_basytec_label_blind_predictions_v344 as engine


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_basytec_v352"


def configure() -> None:
    engine.detect_schema = common.detect_schema
    engine.read_prediction_fields = common.read_prediction_fields
    engine.time_seconds = common.time_seconds
    engine.EXPECTED_ZIP_COUNT = 47
    engine.EXTERNAL = EXTERNAL
    engine.DOWNLOADS = ROOT / "external_basytec_v343" / "downloads"
    engine.DOWNLOAD_RECEIPT = (
        ROOT / "external_basytec_v347" / "confirmation_roster_v347.json"
    )
    engine.OUT = EXTERNAL / "label_blind_v353"


def main() -> int:
    configure()
    result = engine.main()
    source_report = engine.OUT / "basytec_label_blind_prediction_report_v344.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    report["protocol_version"] = "V352_HEADER_FAMILY_COMPLETE"
    report["development_cells_excluded"] = ["F0001.zip"]
    report["confirmation_cells"] = 47
    report["header_only_inventory_sha256"] = common.sha256_file(
        ROOT / "external_basytec_v347" / "header_inventory_v351.json"
    )
    report["capacity_amendment"] = (
        "Ah[Ah] recognized for later frozen scoring; no Ah[Ah] values selected "
        "into this prediction stage"
    )
    report["amendment_artifacts_sha256"] = {
        "wrapper": common.sha256_file(Path(__file__).resolve()),
        "common": common.sha256_file(ROOT / "basytec_external_v352_common.py"),
        "roster": common.sha256_file(engine.DOWNLOAD_RECEIPT),
    }
    amended = engine.OUT / "basytec_label_blind_prediction_report_v353.json"
    amended.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote V353 provenance report {amended}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
