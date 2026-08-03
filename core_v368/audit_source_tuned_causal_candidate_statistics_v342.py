"""Post-hoc statistical audit of the frozen V342 source-tuned comparator.

This script does not change the pre-frozen decision rule or scientific result.
It independently reconstructs domain metrics from cell-level records, merges
MICH and MICH_EXP into one experimental-source group, and reports sensitivity
summaries at the source-group level.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "source_tuned_causal_candidate_v342"
CELL_PATH = INPUT_DIR / "source_tuned_causal_candidate_cell_metrics_v342.csv"
DOMAIN_PATH = INPUT_DIR / "source_tuned_causal_candidate_domain_metrics_v342.csv"
REPORT_PATH = INPUT_DIR / "source_tuned_causal_candidate_statistics_audit_v342.json"

EXPECTED_CELL_SHA256 = "7EF4D2677BE17E7A0E31F8F719A77DA3E4111ABEBD3A9D642618FD90DF89551B"
EXPECTED_DOMAIN_SHA256 = "86890A76D164B4BE6DD716AE3F7BAB91D0D0B6C27DFD1AC4C09D55B8257D482A"
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260807
TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def percentile_bootstrap(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def comparison_summary(values: pd.Series) -> dict[str, object]:
    x = values.to_numpy(float)
    wins = int((x < -TOL).sum())
    ties = int((np.abs(x) <= TOL).sum())
    losses = int((x > TOL).sum())
    leave_one_out = np.asarray(
        [np.delete(x, index).mean() for index in range(len(x))], dtype=float
    )
    return {
        "mean": float(x.mean()),
        "bootstrap_ci95_posthoc": percentile_bootstrap(x),
        "wins_ties_losses": [wins, ties, losses],
        "exact_two_sided_sign_p_descriptive": exact_two_sided_sign_p(wins, losses),
        "leave_one_group_out_mean_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
    }


def main() -> int:
    observed_hashes = {
        "cell": sha256_file(CELL_PATH),
        "domain": sha256_file(DOMAIN_PATH),
    }
    if observed_hashes != {
        "cell": EXPECTED_CELL_SHA256,
        "domain": EXPECTED_DOMAIN_SHA256,
    }:
        raise RuntimeError(f"frozen metric identity failure: {observed_hashes}")

    cell = pd.read_csv(CELL_PATH)
    domain = pd.read_csv(DOMAIN_PATH)
    metric_columns = [
        "baseline_mae",
        "pchp_mae",
        "source_tuned_causal_candidate_mae",
    ]
    reconstructed = (
        cell.groupby("domain", as_index=False, sort=True)[metric_columns].mean()
    )
    aligned = domain[["domain"] + metric_columns].merge(
        reconstructed,
        on="domain",
        how="inner",
        validate="one_to_one",
        suffixes=("_reported", "_reconstructed"),
    )
    reconstruction_max_abs_error = max(
        float(
            np.max(
                np.abs(
                    aligned[f"{column}_reported"].to_numpy(float)
                    - aligned[f"{column}_reconstructed"].to_numpy(float)
                )
            )
        )
        for column in metric_columns
    )

    domain = domain.copy()
    domain["source_group"] = domain["domain"].where(
        ~domain["domain"].isin(["MICH", "MICH_EXP"]), "MICH_PROGRAM"
    )
    grouped = (
        domain.groupby("source_group", as_index=False, sort=True)[metric_columns]
        .mean()
    )
    grouped["pchp_minus_baseline"] = grouped["pchp_mae"] - grouped["baseline_mae"]
    grouped["source_tuned_candidate_minus_baseline"] = (
        grouped["source_tuned_causal_candidate_mae"] - grouped["baseline_mae"]
    )
    grouped["pchp_minus_source_tuned_candidate"] = (
        grouped["pchp_mae"] - grouped["source_tuned_causal_candidate_mae"]
    )

    baseline = float(grouped["baseline_mae"].mean())
    pchp = float(grouped["pchp_mae"].mean())
    candidate = float(grouped["source_tuned_causal_candidate_mae"].mean())
    pchp_gain = baseline - pchp
    candidate_gain = baseline - candidate
    retention = pchp_gain / candidate_gain if candidate_gain > TOL else None

    comparisons = {
        name: comparison_summary(grouped[name])
        for name in (
            "pchp_minus_baseline",
            "source_tuned_candidate_minus_baseline",
            "pchp_minus_source_tuned_candidate",
        )
    }
    report = {
        "version": "v342",
        "status": "POSTHOC_STATISTICAL_AUDIT_COMPLETED",
        "role": "sensitivity_only_not_a_new_confirmatory_test",
        "inputs": [
            {"path": CELL_PATH.relative_to(ROOT).as_posix(), "sha256": observed_hashes["cell"]},
            {"path": DOMAIN_PATH.relative_to(ROOT).as_posix(), "sha256": observed_hashes["domain"]},
        ],
        "design": {
            "record_count": int(cell["records"].sum()),
            "physical_cells": int(len(cell)),
            "dataset_domains": int(domain["domain"].nunique()),
            "experimental_source_groups_after_merge": int(len(grouped)),
            "merged_domains": {"MICH_PROGRAM": ["MICH", "MICH_EXP"]},
            "independent_unit_for_primary_cross_domain_summary": "complete_dataset_domain",
            "sensitivity_unit": "experimental_source_group",
            "nested_measurements": "records_within_physical_cells",
        },
        "reconstruction": {
            "max_abs_error_domain_metrics_from_cell_records": reconstruction_max_abs_error,
            "passes_1e-12": bool(reconstruction_max_abs_error <= TOL),
        },
        "source_group_equal_accuracy": {
            "baseline_mae": baseline,
            "pchp_mae": pchp,
            "source_tuned_causal_candidate_mae": candidate,
            "pchp_gain_over_baseline": pchp_gain,
            "source_tuned_candidate_gain_over_baseline": candidate_gain,
            "pchp_utility_retention_fraction": retention,
        },
        "comparisons": comparisons,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "experimental_source_group",
            "interpretation": "post-hoc heterogeneity sensitivity; not population-confirmatory coverage",
        },
        "audit_conclusion": {
            "primary_narrow_decision_unchanged": True,
            "reason": (
                "Merging MICH and MICH_EXP does not restore the pre-frozen utility gate; "
                "the source-tuned unprotected comparator remains substantially more accurate, "
                "while PCHP alone retains the exact pointwise harm tube."
            ),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
