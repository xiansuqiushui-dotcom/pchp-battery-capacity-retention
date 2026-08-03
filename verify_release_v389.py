"""Portable end-to-end verifier for the frozen V389 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest_v389.json"
EXCLUDED_NAMES = {MANIFEST.name, "verification_receipt_v389.json"}
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
FORBIDDEN_SUFFIXES = {".zip", ".7z", ".rar", ".mat", ".h5", ".hdf5", ".pkl", ".pickle", ".joblib", ".sav"}


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
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def check(condition: bool, label: str, passed: list[str]) -> None:
    if not condition:
        raise AssertionError(label)
    passed.append(label)
    print(f"PASS: {label}")


def run_checked(label: str, args: list[str], cwd: Path, passed: list[str]) -> None:
    completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise AssertionError(f"{label} failed with exit code {completed.returncode}")
    check(True, label, passed)


def two_stage_bootstrap(values: dict[str, np.ndarray], repetitions: int, seed: int) -> tuple[float, tuple[float, float]]:
    names = sorted(values)
    point = float(np.mean([np.mean(values[name]) for name in names]))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    chunk = 2_500
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        count = stop - start
        selected = rng.integers(0, len(names), size=(count, len(names)))
        totals = np.zeros(count, dtype=float)
        for position in range(len(names)):
            chosen = selected[:, position]
            contribution = np.zeros(count, dtype=float)
            for surface_index, name in enumerate(names):
                rows = np.flatnonzero(chosen == surface_index)
                if not len(rows):
                    continue
                array = values[name]
                indices = rng.integers(0, len(array), size=(len(rows), len(array)))
                contribution[rows] = array[indices].mean(axis=1)
            totals += contribution
        draws[start:stop] = totals / len(names)
    interval = tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))
    return point, interval


def verify_manifest(passed: list[str]) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {path.relative_to(ROOT).as_posix(): path for path in tracked_files()}
    check(manifest["version"] == "v389", "manifest version", passed)
    check(set(actual) == set(expected), "manifest has no missing or untracked files", passed)
    for relative, path in actual.items():
        item = expected[relative]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"manifest mismatch: {relative}")
    check(True, f"all {len(actual)} file hashes", passed)


def verify_release_boundary(passed: list[str]) -> None:
    files = tracked_files()
    forbidden = [path for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    oversized = [path for path in files if path.stat().st_size >= 100_000_000]
    check(not forbidden, "no forbidden raw archive or model-bundle suffix", passed)
    check(not oversized, "no file reaches the GitHub 100 MB hard limit", passed)
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    check("MIT License" in license_text and "Yuyang Wu" in license_text, "MIT license present", passed)


def verify_external_mechanism(passed: list[str]) -> None:
    base = ROOT / "updates_v389" / "external_mechanism_decision_v380"
    report = json.loads((base / "external_mechanism_decision_v380_report.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "updates_v389" / "PCHP_EXTERNAL_MECHANISM_DECISION_PREFREEZE_V380.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(base / "external_mechanism_predictions_v380.parquet")
    check(len(predictions) == 9_712, "external record count", passed)
    check(predictions["cell_id"].nunique() == 659, "external physical-cell count", passed)
    check(predictions["domain"].nunique() == 6, "external dataset count", passed)

    predictions = predictions.copy()
    predictions["effect"] = (predictions["pchp_method"] - predictions["truth"]).abs() - (predictions["fixed_shift"] - predictions["truth"]).abs()
    cell = predictions.groupby(["domain", "cell_id"], sort=True)["effect"].mean().reset_index()
    values = {domain: frame["effect"].to_numpy(float) for domain, frame in cell.groupby("domain", sort=True)}
    repetitions = int(frozen["primary_estimand"]["bootstrap_repetitions"])
    seed = int(frozen["primary_estimand"]["bootstrap_seed"])
    point, interval = two_stage_bootstrap(values, repetitions, seed)
    expected_point = float(report["primary_mechanism_estimand"]["pchp_minus_fixed_shift"])
    expected_interval = tuple(float(x) for x in report["primary_mechanism_estimand"]["two_stage_95_percent_interval"])
    check(abs(point - expected_point) <= 1e-14, "external dataset-equal mechanism estimand", passed)
    check(np.max(np.abs(np.asarray(interval) - np.asarray(expected_interval))) <= 1e-14, "external two-stage bootstrap interval", passed)

    displacement = float(np.max(np.abs(predictions["pchp_method"] - predictions["protected_state"])))
    realized_harm = float(np.max((predictions["pchp_method"] - predictions["truth"]).abs() - (predictions["protected_state"] - predictions["truth"]).abs()))
    max_increase = 0.0
    for _, frame in predictions.sort_values(["domain", "cell_id", "target_cycle_number"]).groupby(["domain", "cell_id"], sort=False):
        if len(frame) > 1:
            max_increase = max(max_increase, float(np.max(np.diff(frame["pchp_method"].to_numpy(float)))))
    check(displacement <= 0.01 + 1e-12, "external displacement budget", passed)
    check(realized_harm <= 0.01 + 1e-12, "external realized absolute-loss budget", passed)
    check(max_increase <= 1e-12, "external non-increasing output", passed)


def verify_manuscript(passed: list[str]) -> None:
    manuscript = ROOT / "manuscript"
    en = (manuscript / "main_en.tex").read_text(encoding="utf-8")
    zh = (manuscript / "main_zh.tex").read_text(encoding="utf-8")
    bib = (manuscript / "references.bib").read_text(encoding="utf-8")
    check("Prefix-Causal Harm-Budget Projection for Cross-Domain Lithium-Ion Battery Capacity-Retention Estimation" in en, "final English title", passed)
    check("面向跨域锂离子电池容量保持率估计的" in zh and "前缀因果损害预算投影" in zh, "final Chinese title", passed)
    check("Yuyang Wu" in en and "Aiping Jiang" in en, "author order", passed)
    check("0.00690" in en and "0.00690" in zh and "7.54" in en and "7.54" in zh, "bilingual headline numbers", passed)
    check("\\iffalse" not in en and "\\iffalse" not in zh, "no hidden manuscript branch", passed)
    cited = set()
    for text in [en, zh, (manuscript / "supplement_en.tex").read_text(encoding="utf-8"), (manuscript / "supplement_zh.tex").read_text(encoding="utf-8")]:
        for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", text):
            cited.update(key.strip() for key in group.split(","))
    entries = set(re.findall(r"@\w+\{([^,]+),", bib))
    check(not (cited - entries), "all citation keys resolve", passed)
    for name in ["main_en.pdf", "main_zh.pdf", "supplement_en.pdf", "supplement_zh.pdf"]:
        check((manuscript / name).is_file(), f"compiled artifact present: {name}", passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    passed: list[str] = []

    verify_manifest(passed)
    verify_release_boundary(passed)
    verify_external_mechanism(passed)
    verify_manuscript(passed)

    run_checked("legacy V368 reproducibility core", [sys.executable, "verify_reproducibility_v368.py"], ROOT / "core_v368", passed)
    updates = ROOT / "updates_v389"
    for test in [
        "test_external_robustness_v384.py",
        "test_monotonicity_ablation_v385.py",
        "test_output_range_sensitivity_v386.py",
        "test_data_flow_v387.py",
    ]:
        run_checked(test, [sys.executable, test], updates, passed)

    receipt = {
        "version": "v389",
        "status": "PASS",
        "checks_passed": len(passed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "manifest_sha256": sha256(MANIFEST),
        "check_labels": passed,
    }
    if args.write_receipt:
        path = ROOT / "verification_receipt_v389.json"
        path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path.name}")
    print(f"V389 RELEASE VERIFICATION PASSED: {len(passed)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
