"""Fit and serialize the source-only RCCP models before external data access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from analyze_method_ablation_v315 import RAW_ABSOLUTE_CHANGE_FEATURES
from develop_anchor_invariant_soh_v306 import (
    DATA_PATHS,
    MINIMUM_SAMPLES_LEAF,
    RAW_FEATURES,
    SEED,
    TARGET,
    TREES,
    balanced_source_rows,
    domain_equal_weights,
    load_data,
    raw_matrix,
)
from develop_context_change_soh_v310 import add_context_change


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rccp_source_model_freeze_v316"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_bundle(source, features: list[str], *, domain_equal: bool) -> dict[str, object]:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    matrix = imputer.fit_transform(raw_matrix(source, features))
    model = ExtraTreesRegressor(
        n_estimators=TREES,
        min_samples_leaf=MINIMUM_SAMPLES_LEAF,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    weights = domain_equal_weights(source) if domain_equal else None
    model.fit(matrix, source[TARGET].to_numpy(float), sample_weight=weights)
    return {
        "features": tuple(features),
        "imputer": imputer,
        "model": model,
        "prediction_clip": (0.0, 1.3),
        "domain_equal_training_weights": domain_equal,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enriched = add_context_change(load_data())
    source = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    source_fit = balanced_source_rows(source)
    baseline = fit_bundle(source_fit, list(RAW_FEATURES), domain_equal=False)
    candidate = fit_bundle(
        source_fit,
        list(RAW_ABSOLUTE_CHANGE_FEATURES),
        domain_equal=True,
    )
    model_path = OUT / "rccp_source_models_v316.joblib"
    joblib.dump(
        {
            "version": "v316",
            "target": TARGET,
            "protected_baseline": baseline,
            "absolute_change_candidate": candidate,
            "monotone_projection": "per-cell non-increasing isotonic regression",
            "regret_budget_soh_units": 0.01,
        },
        model_path,
        compress=3,
    )
    metadata_path = OUT / "rccp_source_models_v316_metadata.json"
    metadata = {
        "status": "SOURCE_ONLY_MODELS_FROZEN_BEFORE_OXFORD_DATA_ACCESS",
        "source_domains": sorted(source_fit["domain"].astype(str).unique()),
        "source_physical_cells": int(source_fit["cell_id"].nunique()),
        "source_fit_rows": int(len(source_fit)),
        "target": TARGET,
        "baseline_features": list(RAW_FEATURES),
        "candidate_features": list(RAW_ABSOLUTE_CHANGE_FEATURES),
        "candidate_relative_change_features_used": False,
        "model": {
            "family": "ExtraTreesRegressor",
            "n_estimators": TREES,
            "min_samples_leaf": MINIMUM_SAMPLES_LEAF,
            "max_features": 0.8,
            "random_state": SEED,
        },
        "budget_soh_units": 0.01,
        "training_data": [
            {"path": str(path), "sha256": sha256_file(path)} for path in DATA_PATHS
        ],
        "artifact": {"path": str(model_path), "sha256": sha256_file(model_path)},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
