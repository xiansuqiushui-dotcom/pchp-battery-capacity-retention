"""Post-hoc source-group sensitivity for the PCHP development effects.

MICH and MICH_EXP originate from the same experimental program.  The primary
analysis treats complete dataset domains as the cross-domain units.  This
conservative sensitivity merges those two domains before equal weighting,
bootstrap resampling, and leave-one-source-group-out analysis.  It is explicitly
post-hoc and does not replace or retroactively redefine the frozen estimand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRIMARY_INPUT = (
    ROOT
    / "nested_prefix_causal_outer_v327"
    / "nested_outer_domain_metrics_v327.csv"
)
CONTROL_INPUT = (
    ROOT
    / "candidate_information_control_v333"
    / "candidate_information_domain_metrics_v333.csv"
)
OUT = ROOT / "source_group_sensitivity_v339"
BOOTSTRAP_REPLICATES = 100_000
TOLERANCE = 1e-12
SEEDS = {"primary_development": 20_260_803, "candidate_control": 20_260_804}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_group(domain: str) -> str:
    return "MICH_PROGRAM" if domain in {"MICH", "MICH_EXP"} else domain


def summarize(
    rows: pd.DataFrame,
    *,
    analysis: str,
    baseline_column: str,
    method_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    grouped = (
        rows.assign(source_group=rows["domain"].astype(str).map(source_group))
        .groupby("source_group", as_index=False, sort=True)[
            [baseline_column, method_column]
        ]
        .mean()
    )
    grouped = grouped.rename(
        columns={baseline_column: "baseline_mae", method_column: "pchp_mae"}
    )
    grouped["pchp_minus_baseline"] = (
        grouped["pchp_mae"] - grouped["baseline_mae"]
    )
    differences = grouped["pchp_minus_baseline"].to_numpy(float)
    rng = np.random.default_rng(SEEDS[analysis])
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPLICATES, len(differences)),
    )
    replicates = differences[indices].mean(axis=1)
    leave_one_out = np.asarray(
        [np.delete(differences, index).mean() for index in range(len(differences))],
        dtype=float,
    )
    summary = {
        "analysis": analysis,
        "status": "POSTHOC_CONSERVATIVE_SOURCE_GROUP_SENSITIVITY",
        "independent_sensitivity_unit": "experimental source group",
        "source_groups": int(len(grouped)),
        "merged_domains": {"MICH_PROGRAM": ["MICH", "MICH_EXP"]},
        "source_group_equal_baseline_mae": float(grouped["baseline_mae"].mean()),
        "source_group_equal_pchp_mae": float(grouped["pchp_mae"].mean()),
        "source_group_equal_pchp_minus_baseline": float(differences.mean()),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": SEEDS[analysis],
            "ci95": np.percentile(replicates, [2.5, 97.5]).tolist(),
        },
        "source_group_wins_ties_losses": [
            int((differences < -TOLERANCE).sum()),
            int((np.abs(differences) <= TOLERANCE).sum()),
            int((differences > TOLERANCE).sum()),
        ],
        "leave_one_source_group_out_mean_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
        "all_leave_one_source_group_out_means_negative": bool(
            (leave_one_out < 0.0).all()
        ),
    }
    grouped.insert(0, "analysis", analysis)
    return summary, grouped


def main() -> None:
    for path in (PRIMARY_INPUT, CONTROL_INPUT):
        if not path.exists():
            raise FileNotFoundError(path)
    primary = pd.read_csv(PRIMARY_INPUT)
    pivot = primary.pivot(
        index="domain", columns="method", values="cell_macro_mae"
    ).reset_index()
    primary_summary, primary_groups = summarize(
        pivot,
        analysis="primary_development",
        baseline_column="selected_causal_baseline",
        method_column="selected_causal_method",
    )
    control = pd.read_csv(CONTROL_INPUT)
    control_summary, control_groups = summarize(
        control,
        analysis="candidate_control",
        baseline_column="control_cell_macro_mae",
        method_column="pchp_cell_macro_mae",
    )
    groups = pd.concat([primary_groups, control_groups], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    groups_path = OUT / "source_group_effects_v339.csv"
    report_path = OUT / "source_group_sensitivity_v339_report.json"
    groups.to_csv(groups_path, index=False)
    report = {
        "status": "POSTHOC_SOURCE_GROUP_SENSITIVITY_COMPLETED",
        "reason": (
            "MICH and MICH_EXP share an experimental program; merging them "
            "tests whether treating them as separate development domains "
            "materially drives either effect."
        ),
        "primary_estimator_unchanged": True,
        "analyses": [primary_summary, control_summary],
        "inputs": [
            {
                "path": PRIMARY_INPUT.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(PRIMARY_INPUT),
            },
            {
                "path": CONTROL_INPUT.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(CONTROL_INPUT),
            },
        ],
        "output": {
            "path": groups_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(groups_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
