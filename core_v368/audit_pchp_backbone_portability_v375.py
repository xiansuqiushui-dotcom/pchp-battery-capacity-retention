"""Audit current PCHP across three untuned secondary regression backbones."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_method_ablation_v315 import RAW_ABSOLUTE_CHANGE_FEATURES
from develop_anchor_invariant_soh_v306 import (
    RAW_FEATURES,
    TARGET,
    balanced_source_rows,
    load_data,
)
from develop_context_change_soh_v310 import add_context_change
from evaluate_backbone_generality_v314 import fit_predict_backbone
from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
PREFREEZE = ROOT / "PCHP_BACKBONE_PORTABILITY_PREFREEZE_V375.json"
SELECTIONS = (
    ROOT
    / "nested_prefix_causal_selection_v326"
    / "nested_alpha_selections_v326.csv"
)
OUT = ROOT / "pchp_backbone_portability_v375"
CACHE = OUT / "raw_prediction_cache"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prefreeze() -> dict[str, object]:
    prefreeze = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if prefreeze["status"] != "PREFROZEN_BEFORE_V375_EXECUTION":
        raise RuntimeError("V375 prefreeze status is not executable")
    for relative, expected in prefreeze["inputs"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V375 frozen input identity mismatch: {relative}")
    alpha_contract = prefreeze["alpha_schedule"]
    if sha256_file(ROOT / alpha_contract["path"]) != alpha_contract["sha256"]:
        raise RuntimeError("V375 frozen alpha schedule identity mismatch")
    return prefreeze


def cache_path(backbone: str, target_domain: str) -> Path:
    return CACHE / f"{backbone}__{target_domain}.parquet"


def valid_cache(path: Path, backbone: str, target_domain: str) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    required = {
        "domain",
        "cell_id",
        "target_cycle_number",
        "truth",
        "raw_baseline",
        "raw_candidate",
    }
    return bool(
        set(frame.columns) == required
        and set(frame["domain"].astype(str).unique()) == {target_domain}
        and not frame.isna().any().any()
    )


def train_or_load_fold(
    backbone: str,
    target_domain: str,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = cache_path(backbone, target_domain)
    if not valid_cache(path, backbone, target_domain):
        source = evaluation.loc[
            evaluation["domain"].astype(str) != target_domain
        ].copy()
        target = evaluation.loc[
            evaluation["domain"].astype(str) == target_domain
        ].copy()
        source_fit = balanced_source_rows(source)
        raw_baseline = fit_predict_backbone(
            backbone,
            source_fit,
            target,
            RAW_FEATURES,
            domain_equal=False,
        )
        raw_candidate = fit_predict_backbone(
            backbone,
            source_fit,
            target,
            RAW_ABSOLUTE_CHANGE_FEATURES,
            domain_equal=True,
        )
        frame = target[
            ["domain", "cell_id", "target_cycle_number", TARGET]
        ].rename(columns={TARGET: "truth"})
        frame["raw_baseline"] = raw_baseline
        frame["raw_candidate"] = raw_candidate
        frame.to_parquet(path, index=False)
        trained = True
    else:
        frame = pd.read_parquet(path)
        source = evaluation.loc[
            evaluation["domain"].astype(str) != target_domain
        ]
        source_fit = balanced_source_rows(source)
        trained = False
    metadata = {
        "backbone": backbone,
        "target_domain": target_domain,
        "source_domains": sorted(source["domain"].astype(str).unique()),
        "source_fit_rows": int(len(source_fit)),
        "target_rows": int(len(frame)),
        "target_cells": int(frame["cell_id"].nunique()),
        "outer_target_labels_used_for_training_or_alpha_selection": False,
        "cache_retrained_this_run": trained,
        "cache": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
    }
    return frame, metadata


def prefix_replay_max_difference(
    rows: pd.DataFrame,
    state: np.ndarray,
    method: np.ndarray,
    alpha: float,
) -> float:
    identifiers = rows["cell_id"].astype(str).to_numpy()
    cycles = rows["target_cycle_number"].to_numpy(float)
    raw = rows["raw_baseline"].to_numpy(float)
    candidate = rows["raw_candidate"].to_numpy(float)
    maximum = 0.0
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        length = max(1, len(order) // 2)
        prefix = order[:length]
        prefix_state, prefix_method = prefix_causal_cellwise_projection(
            np.repeat(identifier, length),
            cycles[prefix],
            raw[prefix],
            candidate[prefix],
            0.01,
            assimilation=alpha,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(prefix_state - state[prefix]))),
            float(np.max(np.abs(prefix_method - method[prefix]))),
        )
    return maximum


def trajectory_maximum_increase(
    rows: pd.DataFrame,
    prediction: np.ndarray,
) -> float:
    identifiers = rows["cell_id"].astype(str).to_numpy()
    cycles = rows["target_cycle_number"].to_numpy(float)
    maximum = -np.inf
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        if len(order) > 1:
            maximum = max(maximum, float(np.max(np.diff(prediction[order]))))
    return maximum


def apply_pchp(
    rows: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    identifiers = rows["cell_id"].astype(str).to_numpy()
    cycles = rows["target_cycle_number"].to_numpy(float)
    raw = rows["raw_baseline"].to_numpy(float)
    candidate = rows["raw_candidate"].to_numpy(float)
    truth = rows["truth"].to_numpy(float)
    state, method = prefix_causal_cellwise_projection(
        identifiers,
        cycles,
        raw,
        candidate,
        0.01,
        assimilation=alpha,
    )
    output = rows.copy()
    output["selected_alpha"] = alpha
    output["protected_state"] = state
    output["pchp_method"] = method
    certificate = {
        "maximum_absolute_displacement": float(np.max(np.abs(method - state))),
        "maximum_observed_absolute_loss_regret": float(
            np.max(np.abs(method - truth) - np.abs(state - truth))
        ),
        "maximum_protected_state_increase": trajectory_maximum_increase(rows, state),
        "maximum_pchp_output_increase": trajectory_maximum_increase(rows, method),
        "minimum_protected_state": float(np.min(state)),
        "maximum_protected_state": float(np.max(state)),
        "minimum_pchp_output": float(np.min(method)),
        "maximum_pchp_output": float(np.max(method)),
        "maximum_prefix_replay_difference": prefix_replay_max_difference(
            rows, state, method, alpha
        ),
    }
    return output, certificate


def aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roles = ("protected_state", "pchp_method", "raw_candidate")
    cell_blocks: list[pd.DataFrame] = []
    for (backbone, domain), rows in predictions.groupby(
        ["backbone", "domain"], sort=True
    ):
        for role in roles:
            absolute_error = np.abs(
                rows[role].to_numpy(float) - rows["truth"].to_numpy(float)
            )
            cells = rows[["cell_id"]].copy()
            cells["absolute_error"] = absolute_error
            cells = (
                cells.groupby("cell_id", as_index=False)
                .agg(cell_mae=("absolute_error", "mean"))
            )
            cells["backbone"] = backbone
            cells["domain"] = domain
            cells["method"] = role
            cell_blocks.append(cells)
    cell_metrics = pd.concat(cell_blocks, ignore_index=True)
    domain_metrics = (
        cell_metrics.groupby(["backbone", "domain", "method"], as_index=False)
        .agg(
            cell_macro_mae=("cell_mae", "mean"),
            physical_cells=("cell_id", "nunique"),
        )
    )
    return cell_metrics, domain_metrics


def compare_backbone(
    domain_metrics: pd.DataFrame,
    backbone: str,
    prefreeze: dict[str, object],
) -> dict[str, object]:
    subset = domain_metrics.loc[domain_metrics["backbone"] == backbone]
    method = subset.loc[
        subset["method"] == "pchp_method", ["domain", "cell_macro_mae"]
    ]
    baseline = subset.loc[
        subset["method"] == "protected_state", ["domain", "cell_macro_mae"]
    ]
    paired = method.merge(
        baseline,
        on="domain",
        suffixes=("_method", "_baseline"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["cell_macro_mae_method"] - paired["cell_macro_mae_baseline"]
    )
    differences = paired["difference"].to_numpy(float)
    bootstrap = prefreeze["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    indices = rng.integers(
        0,
        len(differences),
        size=(int(bootstrap["repetitions"]), len(differences)),
    )
    replicates = differences[indices].mean(axis=1)
    wins = int((differences < -TOLERANCE).sum())
    ties = int((np.abs(differences) <= TOLERANCE).sum())
    losses = int((differences > TOLERANCE).sum())
    return {
        "backbone": backbone,
        "difference_direction": "negative favors PCHP",
        "domain_equal_mean_difference": float(differences.mean()),
        "bonferroni_adjusted_interval": [
            float(np.quantile(replicates, bootstrap["lower_quantile"])),
            float(np.quantile(replicates, bootstrap["upper_quantile"])),
        ],
        "domain_wins_ties_losses": [wins, ties, losses],
        "per_domain": json.loads(paired.to_json(orient="records")),
    }


def main() -> None:
    started = time.perf_counter()
    prefreeze = verify_prefreeze()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(SELECTIONS)
    if (
        len(selections) != 12
        or selections["inner_domains"].min() != 11
        or selections["outer_target_labels_used_for_selection"].any()
    ):
        raise RuntimeError("V375 frozen alpha schedule is invalid")
    alpha_map = dict(
        zip(
            selections["outer_target_domain"].astype(str),
            selections["selected_alpha"].astype(float),
        )
    )

    enriched = add_context_change(load_data())
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    if (
        evaluation["domain"].nunique()
        != prefreeze["information_boundary"]["outer_domains"]
        or evaluation["cell_id"].nunique()
        != prefreeze["information_boundary"]["physical_cells"]
        or len(evaluation)
        != prefreeze["information_boundary"]["post_reference_rows_per_backbone"]
    ):
        raise RuntimeError("V375 development roster differs from the prefreeze")

    domains = sorted(evaluation["domain"].astype(str).unique())
    prediction_blocks: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []
    certificate_by_backbone: dict[str, dict[str, float]] = {}
    for backbone in prefreeze["secondary_backbones"]:
        certificate_by_backbone[backbone] = {
            "maximum_absolute_displacement": -np.inf,
            "maximum_observed_absolute_loss_regret": -np.inf,
            "maximum_protected_state_increase": -np.inf,
            "maximum_pchp_output_increase": -np.inf,
            "minimum_protected_state": np.inf,
            "maximum_protected_state": -np.inf,
            "minimum_pchp_output": np.inf,
            "maximum_pchp_output": -np.inf,
            "maximum_prefix_replay_difference": 0.0,
        }
        for fold_index, target_domain in enumerate(domains, start=1):
            raw, metadata = train_or_load_fold(
                backbone, target_domain, evaluation
            )
            projected, certificate = apply_pchp(
                raw,
                alpha=float(alpha_map[target_domain]),
            )
            projected["backbone"] = backbone
            prediction_blocks.append(projected)
            fold_metadata.append(metadata)
            aggregate = certificate_by_backbone[backbone]
            for key in (
                "maximum_absolute_displacement",
                "maximum_observed_absolute_loss_regret",
                "maximum_protected_state_increase",
                "maximum_pchp_output_increase",
                "maximum_protected_state",
                "maximum_pchp_output",
                "maximum_prefix_replay_difference",
            ):
                aggregate[key] = max(aggregate[key], certificate[key])
            for key in ("minimum_protected_state", "minimum_pchp_output"):
                aggregate[key] = min(aggregate[key], certificate[key])
            print(
                f"V375 {backbone} fold {fold_index}/{len(domains)}: "
                f"{target_domain}",
                flush=True,
            )

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    cells, domain_metrics = aggregate_metrics(predictions)
    comparisons = [
        compare_backbone(domain_metrics, backbone, prefreeze)
        for backbone in prefreeze["secondary_backbones"]
    ]

    gates: dict[str, dict[str, bool]] = {}
    for comparison in comparisons:
        backbone = comparison["backbone"]
        certificate = certificate_by_backbone[backbone]
        gates[backbone] = {
            "mean_difference_below_zero": (
                comparison["domain_equal_mean_difference"] < 0.0
            ),
            "adjusted_interval_upper_below_zero": (
                comparison["bonferroni_adjusted_interval"][1] < 0.0
            ),
            "minimum_improved_outer_domains": (
                comparison["domain_wins_ties_losses"][0]
                >= prefreeze["retain_gates_per_secondary_backbone"][
                    "minimum_improved_outer_domains"
                ]
            ),
            "absolute_displacement_budget": (
                certificate["maximum_absolute_displacement"] <= 0.01 + TOLERANCE
            ),
            "observed_absolute_loss_regret": (
                certificate["maximum_observed_absolute_loss_regret"]
                <= 0.01 + TOLERANCE
            ),
            "protected_state_nonincreasing": (
                certificate["maximum_protected_state_increase"] <= TOLERANCE
            ),
            "pchp_output_nonincreasing": (
                certificate["maximum_pchp_output_increase"] <= TOLERANCE
            ),
            "physical_range": (
                certificate["minimum_protected_state"] >= -TOLERANCE
                and certificate["maximum_protected_state"] <= 1.3 + TOLERANCE
                and certificate["minimum_pchp_output"] >= -TOLERANCE
                and certificate["maximum_pchp_output"] <= 1.3 + TOLERANCE
            ),
            "prefix_replay_exact": (
                certificate["maximum_prefix_replay_difference"] == 0.0
            ),
            "outer_target_information_isolated": True,
        }
    retained = all(all(backbone_gates.values()) for backbone_gates in gates.values())

    summary = (
        domain_metrics.groupby(["backbone", "method"], as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
        )
        .sort_values(["backbone", "domain_equal_cell_macro_mae"])
    )
    paths = {
        "summary": OUT / "pchp_backbone_portability_summary_v375.csv",
        "domain_metrics": OUT / "pchp_backbone_portability_domain_metrics_v375.csv",
        "cell_metrics": OUT / "pchp_backbone_portability_cell_metrics_v375.csv",
        "predictions": OUT / "pchp_backbone_portability_predictions_v375.parquet",
        "report": OUT / "pchp_backbone_portability_v375_report.json",
    }
    summary.to_csv(paths["summary"], index=False)
    domain_metrics.to_csv(paths["domain_metrics"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    report = {
        "version": "v375",
        "status": (
            "PCHP_BACKBONE_PORTABILITY_RETAINED"
            if retained
            else "PCHP_BACKBONE_PORTABILITY_REJECTED"
        ),
        "decision": "RETAIN" if retained else "REJECT",
        "scope": prefreeze["scope"],
        "prefreeze": {"path": str(PREFREEZE), "sha256": sha256_file(PREFREEZE)},
        "information_boundary": prefreeze["information_boundary"],
        "model_contract": prefreeze["model_contract"],
        "alpha_schedule": {
            "path": str(SELECTIONS),
            "sha256": sha256_file(SELECTIONS),
            "selected_alpha_by_outer_domain": alpha_map,
            "secondary_backbone_specific_tuning": False,
        },
        "summary": json.loads(summary.to_json(orient="records")),
        "comparisons": comparisons,
        "deterministic_certificate_by_backbone": certificate_by_backbone,
        "gates": gates,
        "conjunction_gate_passed": retained,
        "folds": fold_metadata,
        "limitations": [
            "All twelve domains were historically opened; this is retrospective development robustness evidence.",
            "Secondary hyperparameters and the alpha schedule were not tuned for the secondary backbones.",
            "The audit covers one linear and two tree-boosting families, not neural sequence models or every candidate class.",
            "No external result or V368 primary estimand is changed by this audit.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
