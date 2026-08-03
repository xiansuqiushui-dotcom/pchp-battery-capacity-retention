from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "budget_path_anchor_audit_v362"
OUT.mkdir(parents=True, exist_ok=True)

TOL = 1e-12

FILES = {
    "v361_analysis_script": ROOT / "analyze_budget_path_candidate_control_v361.py",
    "v361_prefreeze": ROOT / "paper_q1" / "BUDGET_PATH_CANDIDATE_CONTROL_PREFREEZE_V361.json",
    "v361_report": ROOT / "budget_path_candidate_control_v361" / "budget_path_candidate_control_v361_report.json",
    "v361_summary": ROOT / "budget_path_candidate_control_v361" / "budget_path_summary_v361.csv",
    "v361_domain_metrics": ROOT / "budget_path_candidate_control_v361" / "budget_path_domain_metrics_v361.csv",
    "v361_pchp_selections": ROOT / "budget_path_candidate_control_v361" / "budget_path_pchp_selections_v361.csv",
    "v326_alpha_selections": ROOT / "nested_prefix_causal_selection_v326" / "nested_alpha_selections_v326.csv",
    "v327_report": ROOT / "nested_prefix_causal_outer_v327" / "nested_prefix_causal_outer_v327_report.json",
    "v327_summary": ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_summary_v327.csv",
    "v327_domain_metrics": ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_domain_metrics_v327.csv",
}

EXPECTED_SHA256 = {
    "v361_analysis_script": "bdd0a1aeaf715d25fe574feb52b1679551a608999ab49848a3ff8c6dfd701f79",
    "v361_prefreeze": "7bc32a42d2c20e61471357126f6281e3a11ac976979d56056b68b243f6a6c4d6",
    "v361_report": "6e8d09554f73f1df7cdf6cf8affdaaf1c13eec80b6778b5e58616038ec02781b",
    "v361_summary": "340adf8ed60a68c748b5b03aa0fa20ede6ed068a93561bd2a8ca164d747e4240",
    "v361_domain_metrics": "9e9798814776c22cd0542247ceb86b3abb0e1421ee02ad9985e86c88728fe191",
    "v361_pchp_selections": "338748019aad99d9e8901dc54a1f8415ee80dc1538486a5e950a873886e6f1fe",
    "v326_alpha_selections": "7440327383458225bd8b92f2998620eec4d8d924b8e31cfab05428faaca81eab",
    "v327_report": "32b855617954088371fad07620cdc9c90d93cca541b952a2ec1b96f6444f668b",
    "v327_summary": "79323655efb23e523ff05d1187ef6d8038e0b295a9e58ca5545fda266a7ad8a3",
    "v327_domain_metrics": "724c65d212bea87b4c35e99d6fdf1a72708c9991d51dc51ec1777592ab29a7c3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a: float, b: float, atol: float = TOL) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def main() -> None:
    observed_hashes = {name: sha256(path) for name, path in FILES.items()}
    hash_checks = {
        name: observed_hashes[name] == expected
        for name, expected in EXPECTED_SHA256.items()
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"Frozen input hash mismatch: {hash_checks}")

    report361 = json.loads(FILES["v361_report"].read_text(encoding="utf-8"))
    report327 = json.loads(FILES["v327_report"].read_text(encoding="utf-8"))
    summary361 = pd.read_csv(FILES["v361_summary"])
    domains361 = pd.read_csv(FILES["v361_domain_metrics"])
    selections361 = pd.read_csv(FILES["v361_pchp_selections"])
    selections326 = pd.read_csv(FILES["v326_alpha_selections"])
    summary327 = pd.read_csv(FILES["v327_summary"])
    domains327 = pd.read_csv(FILES["v327_domain_metrics"])

    anchor361 = summary361.loc[np.isclose(summary361["budget"], 0.01, rtol=0.0, atol=TOL)].iloc[0]
    method327 = summary327.loc[summary327["method"] == "selected_causal_method"].iloc[0]
    baseline327 = summary327.loc[summary327["method"] == "selected_causal_baseline"].iloc[0]
    json_method327 = next(
        item for item in report327["summary"]
        if item["method"] == "selected_causal_method"
    )

    authoritative_aggregate_checks = {
        "pchp_mae_matches_v327_high_precision_csv": close(
            anchor361["domain_equal_pchp_mae"],
            method327["domain_equal_cell_macro_mae"],
        ),
        "baseline_mae_matches_v327_high_precision_csv": close(
            anchor361["domain_equal_baseline_mae"],
            baseline327["domain_equal_cell_macro_mae"],
        ),
        "pchp_minus_baseline_matches_high_precision_csv_difference": close(
            anchor361["domain_equal_pchp_minus_baseline"],
            float(method327["domain_equal_cell_macro_mae"])
            - float(baseline327["domain_equal_cell_macro_mae"]),
        ),
    }

    d361 = domains361.loc[np.isclose(domains361["budget"], 0.01, rtol=0.0, atol=TOL)].copy()
    d327 = domains327.loc[
        domains327["method"].isin(["selected_causal_method", "selected_causal_baseline"])
    ].pivot(index="domain", columns="method", values=["physical_cells", "cell_macro_mae"])

    domain_rows: list[dict[str, object]] = []
    for row in d361.sort_values("domain").itertuples(index=False):
        pchp327 = float(d327.loc[row.domain, ("cell_macro_mae", "selected_causal_method")])
        baseline_domain327 = float(d327.loc[row.domain, ("cell_macro_mae", "selected_causal_baseline")])
        cells_method327 = int(d327.loc[row.domain, ("physical_cells", "selected_causal_method")])
        cells_baseline327 = int(d327.loc[row.domain, ("physical_cells", "selected_causal_baseline")])
        domain_rows.append(
            {
                "domain": row.domain,
                "v361_pchp_mae": float(row.pchp_cell_macro_mae),
                "v327_pchp_mae": pchp327,
                "absolute_pchp_difference": abs(float(row.pchp_cell_macro_mae) - pchp327),
                "pchp_matches": close(row.pchp_cell_macro_mae, pchp327),
                "v361_baseline_mae": float(row.baseline_cell_macro_mae),
                "v327_baseline_mae": baseline_domain327,
                "absolute_baseline_difference": abs(float(row.baseline_cell_macro_mae) - baseline_domain327),
                "baseline_matches": close(row.baseline_cell_macro_mae, baseline_domain327),
                "physical_cells_match": int(row.physical_cells) == cells_method327 == cells_baseline327,
            }
        )
    domain_comparison = pd.DataFrame(domain_rows)
    domain_comparison.to_csv(OUT / "v361_v327_domain_anchor_comparison_v362.csv", index=False)

    s361 = selections361.loc[np.isclose(selections361["budget"], 0.01, rtol=0.0, atol=TOL)].copy()
    s361 = s361.rename(columns={"outer_target_domain": "domain"})
    s326 = selections326.rename(columns={"outer_target_domain": "domain"})
    selection_join = s361.merge(s326, on="domain", how="outer", validate="one_to_one", suffixes=("_v361", "_v326"))
    selection_checks = {
        "domain_roster_matches_v326": len(selection_join) == 12 and not selection_join.isna().any().any(),
        "selected_alpha_roster_matches_v326": bool(
            np.isclose(
                selection_join["selected_pchp_alpha"],
                selection_join["selected_alpha"],
                rtol=0.0,
                atol=TOL,
            ).all()
        ),
        "inner_method_score_matches_v326": bool(
            np.isclose(
                selection_join["selected_inner_domain_equal_pchp_mae"],
                selection_join["selected_inner_domain_equal_method_mae"],
                rtol=0.0,
                atol=TOL,
            ).all()
        ),
        "outer_target_labels_unused_in_both": bool(
            (~selection_join["outer_target_labels_used_for_selection_v361"].astype(bool)).all()
            and (~selection_join["outer_target_labels_used_for_selection_v326"].astype(bool)).all()
        ),
    }

    operational_diagnostic = {
        "authoritative_v327_csv_value": float(method327["domain_equal_cell_macro_mae"]),
        "rounded_v327_json_value_used_by_v361_anchor": float(json_method327["domain_equal_cell_macro_mae"]),
        "absolute_rounding_difference": abs(
            float(method327["domain_equal_cell_macro_mae"])
            - float(json_method327["domain_equal_cell_macro_mae"])
        ),
        "v361_original_anchor_tolerance": TOL,
        "rounded_json_comparison_fails_at_original_tolerance": not close(
            method327["domain_equal_cell_macro_mae"],
            json_method327["domain_equal_cell_macro_mae"],
        ),
        "v361_recorded_pchp_mae_anchor_false": report361["anchor_checks"]["pchp_mae_matches_v327"] is False,
        "v361_recorded_status_preserved": report361["status"] == "PROTOCOL_LOCKED_RETROSPECTIVE_BUDGET_PATH_GATE_NOT_PASSED",
        "v361_recorded_decision_preserved": report361["decision"] == "NARROW",
    }

    scientific_gates_preserved = {
        key: bool(value)
        for key, value in report361["gates"].items()
        if key != "all_anchor_checks"
    }
    scientific_gates_preserved["v361_primary_auc_values_unchanged"] = (
        float(report361["primary"]["domain_equal_mean"]) == -0.0069422304526310374
        and list(report361["primary"]["domain_bootstrap_ci95"])
        == [-0.011082909073005586, -0.0029569289946119034]
    )

    corrected_anchor_checks = {
        **authoritative_aggregate_checks,
        "all_domain_pchp_metrics_match_v327": bool(domain_comparison["pchp_matches"].all()),
        "all_domain_baseline_metrics_match_v327": bool(domain_comparison["baseline_matches"].all()),
        "all_domain_physical_cell_counts_match_v327": bool(domain_comparison["physical_cells_match"].all()),
        **selection_checks,
    }

    audit_pass = all(hash_checks.values()) and all(operational_diagnostic.values()) and all(
        corrected_anchor_checks.values()
    ) and all(scientific_gates_preserved.values())

    output = {
        "version": "V362",
        "audit_scope": "independent operational audit of the V361 delta=0.01 historical anchor",
        "scientific_outputs_recomputed": False,
        "v361_files_modified": False,
        "frozen_input_hashes": observed_hashes,
        "hash_checks": hash_checks,
        "operational_diagnostic": operational_diagnostic,
        "authoritative_aggregate_checks": authoritative_aggregate_checks,
        "domain_checks": {
            "domains": int(len(domain_comparison)),
            "maximum_absolute_pchp_difference": float(domain_comparison["absolute_pchp_difference"].max()),
            "maximum_absolute_baseline_difference": float(domain_comparison["absolute_baseline_difference"].max()),
        },
        "selection_checks": selection_checks,
        "corrected_anchor_checks": corrected_anchor_checks,
        "scientific_gates_preserved": scientific_gates_preserved,
        "audit_pass": audit_pass,
        "status": (
            "V361_ANCHOR_FAILURE_CONFIRMED_AS_JSON_PRECISION_FALSE_NEGATIVE"
            if audit_pass
            else "V362_ANCHOR_AUDIT_NOT_PASSED"
        ),
        "decision": (
            "RETAIN_V361_SCIENTIFIC_RESULT_AS_PROTOCOL_LOCKED_RETROSPECTIVE_SENSITIVITY_WITH_V362_AUDIT_TRAIL"
            if audit_pass
            else "NARROW"
        ),
        "interpretation_boundary": (
            "This audit repairs only the provenance and numerical-precision validation of the historical anchor. "
            "It does not change the V361 data, estimand, budget grid, model selection, bootstrap, scientific gates, "
            "or retrospective status, and it does not convert the analysis into external confirmation."
        ),
    }
    (OUT / "budget_path_anchor_audit_v362_report.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
