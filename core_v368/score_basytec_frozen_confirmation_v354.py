"""Score the 47-cell V352 confirmation after V353 predictions are frozen."""

from __future__ import annotations

import json
from pathlib import Path

import basytec_external_v352_common as common
import score_basytec_frozen_confirmation_v345 as engine


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_basytec_v352"


def configure() -> None:
    engine.detect_schema = common.detect_schema
    engine.read_scoring_fields = common.read_scoring_fields
    engine.capacity_map = common.capacity_map
    engine.EXTERNAL = EXTERNAL
    engine.DOWNLOADS = ROOT / "external_basytec_v343" / "downloads"
    engine.PREDICTION_DIR = EXTERNAL / "label_blind_v353"
    engine.PREDICTION_REPORT = (
        engine.PREDICTION_DIR / "basytec_label_blind_prediction_report_v353.json"
    )
    engine.OUT = EXTERNAL / "scored_v354"
    engine.RECORD_PATH = engine.OUT / "basytec_scored_records_v354.parquet"
    engine.CELL_PATH = engine.OUT / "basytec_cell_metrics_v354.csv"
    engine.REPORT_PATH = engine.OUT / "basytec_frozen_confirmation_v354_report.json"


def main() -> int:
    configure()
    result = engine.main()
    report = json.loads(engine.REPORT_PATH.read_text(encoding="utf-8"))
    report["protocol_version"] = "V352_HEADER_FAMILY_COMPLETE"
    report["development_cells_excluded"] = ["F0001.zip"]
    report["amendment_artifacts_sha256"] = {
        "wrapper": common.sha256_file(Path(__file__).resolve()),
        "common": common.sha256_file(ROOT / "basytec_external_v352_common.py"),
        "prediction_report": common.sha256_file(engine.PREDICTION_REPORT),
    }
    engine.REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"updated V354 provenance in {engine.REPORT_PATH}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
