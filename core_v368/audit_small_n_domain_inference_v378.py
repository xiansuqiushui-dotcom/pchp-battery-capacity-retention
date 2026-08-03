"""Finite-sample sensitivity audit for the three domain-level PCHP effects.

This post-hoc audit reads only frozen domain aggregates.  It does not change
the estimands, comparisons, model selection, external confirmation, or the
original percentile-bootstrap reports.  Its purpose is to test whether the
reported directions survive complementary small-sample analyses over the
twelve complete development domains.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
PRIMARY_PATH = (
    ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_domain_metrics_v327.csv"
)
CONTROL_PATH = (
    ROOT
    / "candidate_information_control_v333"
    / "candidate_information_domain_metrics_v333.csv"
)
BUDGET_PATH = (
    ROOT / "budget_path_candidate_control_v361" / "budget_path_domain_auc_v361.csv"
)
REPORT_PATH = ROOT / "small_n_domain_inference_v378_report.json"

EXPECTED_SHA256 = {
    PRIMARY_PATH: "724c65d212bea87b4c35e99d6fdf1a72708c9991d51dc51ec1777592ab29a7c3",
    CONTROL_PATH: "d9c48cf22f5602029a7cd06d6b0e884eb667b2a10c6ea9c6aedffa54655fef70",
    BUDGET_PATH: "f3c1275be42e9de2d38a4d516c3518a3777bad5a0341f4d07a2852d28693c416",
}
TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    tail = min(wins, losses)
    return float(
        min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    )


def exact_two_sided_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceedances = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.dot(np.asarray(signs), values) / len(values)))
        exceedances += int(statistic >= observed - TOL)
        assignments += 1
    return float(exceedances / assignments)


def summarize(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if len(values) != 12 or not np.isfinite(values).all():
        raise RuntimeError("expected twelve finite complete-domain effects")
    wins = int((values < -TOL).sum())
    ties = int((np.abs(values) <= TOL).sum())
    losses = int((values > TOL).sum())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    t_interval = stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=float(values.mean()),
        scale=standard_error,
    )
    wilcoxon = stats.wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    return {
        "n_complete_domains": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "wins_ties_losses_for_pchp": [wins, ties, losses],
        "student_t_interval_95": [float(t_interval[0]), float(t_interval[1])],
        "exact_two_sided_sign_flip_p_descriptive": exact_two_sided_sign_flip_p(values),
        "exact_two_sided_wilcoxon_p_descriptive": float(wilcoxon.pvalue),
        "exact_two_sided_sign_p_descriptive": exact_two_sided_sign_p(wins, losses),
        "sign_flip_assignments": int(2 ** len(values)),
    }


def main() -> int:
    observed_hashes = {path: sha256_file(path) for path in EXPECTED_SHA256}
    if observed_hashes != EXPECTED_SHA256:
        printable = {
            path.relative_to(ROOT).as_posix(): value
            for path, value in observed_hashes.items()
        }
        raise RuntimeError(f"frozen input identity failure: {printable}")

    primary = pd.read_csv(PRIMARY_PATH)
    primary_wide = primary.pivot(
        index="domain", columns="method", values="cell_macro_mae"
    ).sort_index()
    primary_effect = (
        primary_wide["selected_causal_method"]
        - primary_wide["selected_causal_baseline"]
    ).to_numpy(float)

    control = pd.read_csv(CONTROL_PATH).sort_values("domain")
    control_effect = control["pchp_minus_control"].to_numpy(float)

    budget = pd.read_csv(BUDGET_PATH).sort_values("domain")
    budget_effect = budget[
        "budget_normalized_auc_pchp_minus_control"
    ].to_numpy(float)

    comparisons = {
        "primary_pchp_minus_protected_baseline": summarize(primary_effect),
        "pchp_minus_candidate_free_shift": summarize(control_effect),
        "budget_path_auc_pchp_minus_shift": summarize(budget_effect),
    }

    primary_summary = comparisons["primary_pchp_minus_protected_baseline"]
    control_summary = comparisons["pchp_minus_candidate_free_shift"]
    budget_summary = comparisons["budget_path_auc_pchp_minus_shift"]
    gates = {
        "primary_mean_negative": primary_summary["mean"] < 0,
        "primary_t_interval_upper_below_zero": primary_summary[
            "student_t_interval_95"
        ][1]
        < 0,
        "primary_exact_sign_flip_below_0_01": primary_summary[
            "exact_two_sided_sign_flip_p_descriptive"
        ]
        < 0.01,
        "control_mean_negative": control_summary["mean"] < 0,
        "control_t_interval_upper_below_zero": control_summary[
            "student_t_interval_95"
        ][1]
        < 0,
        "control_exact_sign_flip_below_0_05": control_summary[
            "exact_two_sided_sign_flip_p_descriptive"
        ]
        < 0.05,
        "budget_path_mean_negative": budget_summary["mean"] < 0,
        "budget_path_t_interval_upper_below_zero": budget_summary[
            "student_t_interval_95"
        ][1]
        < 0,
        "budget_path_exact_sign_flip_below_0_01": budget_summary[
            "exact_two_sided_sign_flip_p_descriptive"
        ]
        < 0.01,
    }
    if not all(gates.values()):
        raise RuntimeError(f"small-sample sensitivity gate failure: {gates}")

    report = {
        "version": "v378",
        "status": "SMALL_N_DOMAIN_INFERENCE_SENSITIVITY_PASSED",
        "role": "posthoc_small_sample_sensitivity_not_new_confirmation",
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": observed_hashes[path],
            }
            for path in EXPECTED_SHA256
        ],
        "design": {
            "independent_unit": "complete_dataset_domain",
            "n": 12,
            "nested_measurements": "records_within_cells_within_domains",
            "estimand": "domain_equal_mean_paired_difference",
            "chronology": "posthoc_sensitivity_on_opened_development_domains",
            "primary_analysis_unchanged": True,
        },
        "comparisons": comparisons,
        "gates": gates,
        "interpretation": {
            "supported": (
                "The negative domain-equal mean directions are not artifacts of the "
                "reported percentile-bootstrap calculation alone."
            ),
            "candidate_control_boundary": (
                "The candidate-free comparison has nine wins and three losses; its pure "
                "direction sign test is not below 0.05. The supported estimand is the "
                "domain-equal mean effect, not universal or majority-probability dominance."
            ),
            "not_supported": (
                "These post-hoc checks do not create confirmatory population coverage, "
                "repair dependence between data sources, or establish universal transfer."
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
