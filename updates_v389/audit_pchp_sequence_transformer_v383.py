"""Battery-specific sequence-Transformer portability audit for PCHP V383.

The candidate is a compact Transformer operating directly on three 0--600 s
charge-curve channels sampled every 10 s.  Its architecture, optimizer,
source domains, epoch count, and seeds are inherited from the frozen V180/V181
source-only protocol.  The model is fitted on eight source domains and tested
zero-shot on the complete SDU and MATR trajectories.  No target labels are
used for fitting, calibration, model selection, or projection.

The protected prefix-causal states and the original PCHP predictions are read
from the strict LODO V327 ledger.  The new Transformer proposal is projected
through the same delta=0.01 PCHP output contract.  Results must be retained
regardless of direction and are a post-hoc portability audit, not independent
external confirmation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_postconfirmatory_sequence_baselines_v161 as engine
from evaluate_postconfirmatory_transformer_baselines_v181 import (
    SequenceTransformerRegressor,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pchp_sequence_transformer_v383"
SOURCE_PATH = ROOT / "batterylife_early_charge_soh_v109.parquet"
TARGETS = {
    "SDU": ROOT / "batterylife_sdu_early_charge_soh_v145.parquet",
    "MATR": ROOT / "batterylife_matr_early_charge_soh_v151.parquet",
}
CACHE = ROOT / "cg3t_sequence_cache_v161"
V327 = ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_predictions_v327.parquet"
DELTA = 0.01
SEEDS = (20_260_760, 20_260_761, 20_260_762)
EPOCHS = 15
MAXIMUM_ROWS_PER_SOURCE_CELL = 100
BOOTSTRAP_REPETITIONS = 100_000
BOOTSTRAP_SEED = 20_260_804
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_transformer_ensemble() -> tuple[list[torch.nn.Module], engine.Normalization, dict[str, object]]:
    source = engine._load_post_reference(SOURCE_PATH)
    source["_sequence_index"] = np.arange(len(source), dtype=int)
    source_sequences, cache_metadata = engine.load_or_build_sequence_cache(
        source,
        table_path=SOURCE_PATH,
        cache_name="source",
        cache_dir=CACHE,
    )
    balanced = engine.balanced_source_rows(
        source,
        maximum_rows_per_cell=MAXIMUM_ROWS_PER_SOURCE_CELL,
    )
    sequence_positions = balanced["_sequence_index"].to_numpy(int)
    normalization = engine._normalization(
        source_sequences,
        balanced,
        sequence_positions,
    )
    balanced_raw = np.asarray(source_sequences[sequence_positions], dtype=np.float32)
    sequences, age = engine._apply_normalization(
        balanced_raw,
        balanced,
        normalization,
    )
    device = engine._device()
    original_model = engine.SequenceRegressor
    engine.SequenceRegressor = SequenceTransformerRegressor
    models: list[torch.nn.Module] = []
    try:
        for seed in SEEDS:
            model, _, _ = engine._train_model(
                sequences,
                age,
                balanced.reset_index(drop=True),
                seed=int(seed),
                epochs=EPOCHS,
                device=device,
            )
            models.append(model)
            print(f"trained Transformer ensemble member seed={seed}", flush=True)
    finally:
        engine.SequenceRegressor = original_model
    metadata = {
        "device": str(device),
        "source_domains": sorted(balanced["domain"].astype(str).unique().tolist()),
        "source_cells": int(balanced["cell_id"].nunique()),
        "balanced_source_rows": int(len(balanced)),
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "cache": cache_metadata,
    }
    return models, normalization, metadata


def transformer_predict(
    models: list[torch.nn.Module],
    normalization: engine.Normalization,
    domain: str,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = engine._load_post_reference(path, expected_domain=domain)
    sequences, cache_metadata = engine.load_or_build_sequence_cache(
        target,
        table_path=path,
        cache_name=domain.lower(),
        cache_dir=CACHE,
    )
    normalized_sequence, normalized_age = engine._apply_normalization(
        sequences,
        target,
        normalization,
    )
    device = engine._device()
    original_model = engine.SequenceRegressor
    engine.SequenceRegressor = SequenceTransformerRegressor
    try:
        prediction, _ = engine._ensemble_predict(
            models,
            normalized_sequence,
            normalized_age,
            device=device,
        )
    finally:
        engine.SequenceRegressor = original_model
    output = target[["domain", "cell_id", "target_cycle_number", engine.TARGET]].copy()
    output = output.rename(columns={engine.TARGET: "truth"})
    output["sequence_transformer_candidate"] = np.clip(prediction, 0.0, 1.3)
    return output, cache_metadata


def attach_v327(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    ledger = pd.read_parquet(V327)
    ledger = ledger.loc[
        ledger["domain"].astype(str) == domain,
        [
            "domain",
            "cell_id",
            "target_cycle_number",
            "truth",
            "selected_causal_baseline",
            "selected_causal_method",
        ],
    ].copy()
    keys = ["domain", "cell_id", "target_cycle_number"]
    if frame.duplicated(keys).any() or ledger.duplicated(keys).any():
        raise RuntimeError(f"{domain}: duplicated target key")
    merged = frame.merge(
        ledger,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_sequence", "_v327"),
    )
    if len(merged) != len(frame) or len(merged) != len(ledger):
        raise RuntimeError(f"{domain}: sequence/V327 roster mismatch")
    if not np.allclose(
        merged["truth_sequence"].to_numpy(float),
        merged["truth_v327"].to_numpy(float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(f"{domain}: target mismatch")
    return merged.drop(columns=["truth_v327"]).rename(columns={"truth_sequence": "truth"})


def pchp_project(frame: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for _, rows in frame.groupby("cell_id", sort=False):
        ordered = rows.sort_values("target_cycle_number").copy()
        previous = 1.3
        projected: list[float] = []
        for baseline, candidate in zip(
            ordered["selected_causal_baseline"].to_numpy(float),
            ordered["sequence_transformer_candidate"].to_numpy(float),
        ):
            lower = max(0.0, baseline - DELTA)
            upper = min(1.3, baseline + DELTA, previous)
            if lower > upper + 1e-12:
                raise RuntimeError("PCHP interval is empty")
            value = float(np.clip(candidate, lower, upper))
            projected.append(value)
            previous = value
        ordered["pchp_sequence_transformer"] = projected
        blocks.append(ordered)
    output = pd.concat(blocks, ignore_index=True)
    if (output.groupby("cell_id")["pchp_sequence_transformer"].diff() > 1e-12).any():
        raise RuntimeError("projected Transformer trajectory is not non-increasing")
    return output


def cell_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "selected_causal_baseline",
        "selected_causal_method",
        "sequence_transformer_candidate",
        "pchp_sequence_transformer",
    ]
    blocks: list[pd.DataFrame] = []
    truth = frame["truth"].to_numpy(float)
    for method in methods:
        work = frame[["domain", "cell_id"]].copy()
        error = frame[method].to_numpy(float) - truth
        work["absolute_error"] = np.abs(error)
        work["squared_error"] = error**2
        cells = (
            work.groupby(["domain", "cell_id"], as_index=False)
            .agg(
                records=("absolute_error", "size"),
                cell_mae=("absolute_error", "mean"),
                cell_mse=("squared_error", "mean"),
            )
        )
        cells["cell_rmse"] = np.sqrt(cells.pop("cell_mse"))
        cells["method"] = method
        blocks.append(cells)
    return pd.concat(blocks, ignore_index=True)


def comparisons(cells: pd.DataFrame) -> dict[str, object]:
    wide = cells.pivot(index=["domain", "cell_id"], columns="method", values="cell_mae")
    output: dict[str, object] = {}
    for reference in ("selected_causal_baseline", "selected_causal_method"):
        difference = wide["pchp_sequence_transformer"] - wide[reference]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        domain_results: dict[str, object] = {}
        for domain, values in difference.groupby(level="domain"):
            array = values.to_numpy(float)
            draws = rng.integers(0, len(array), size=(BOOTSTRAP_REPETITIONS, len(array)))
            bootstrap = array[draws].mean(axis=1)
            domain_results[str(domain)] = {
                "cells": int(len(array)),
                "mean_cell_mae_difference": float(array.mean()),
                "ci95_cell_bootstrap": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                "cell_wins_ties_losses": [
                    int(np.sum(array < -TOLERANCE)),
                    int(np.sum(np.abs(array) <= TOLERANCE)),
                    int(np.sum(array > TOLERANCE)),
                ],
            }
        domain_means = np.asarray(
            [item["mean_cell_mae_difference"] for item in domain_results.values()],
            dtype=float,
        )
        output[f"pchp_sequence_transformer_vs_{reference}"] = {
            "dataset_equal_mean_difference": float(domain_means.mean()),
            "per_domain": domain_results,
        }
    return output


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    models, normalization, training = fit_transformer_ensemble()
    prediction_blocks: list[pd.DataFrame] = []
    target_caches: dict[str, object] = {}
    for domain, path in TARGETS.items():
        target, cache_metadata = transformer_predict(models, normalization, domain, path)
        target = attach_v327(target, domain)
        prediction_blocks.append(pchp_project(target))
        target_caches[domain] = cache_metadata
        print(f"completed zero-shot full-trajectory replay: {domain}", flush=True)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    cells = cell_metrics(predictions)
    summary = (
        cells.groupby(["domain", "method"], as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            records=("records", "sum"),
            cell_macro_mae=("cell_mae", "mean"),
            cell_macro_rmse=("cell_rmse", "mean"),
        )
    )

    truth = predictions["truth"].to_numpy(float)
    baseline = predictions["selected_causal_baseline"].to_numpy(float)
    projected = predictions["pchp_sequence_transformer"].to_numpy(float)
    realized_harm = np.abs(projected - truth) - np.abs(baseline - truth)
    deterministic = {
        "maximum_absolute_displacement": float(np.max(np.abs(projected - baseline))),
        "maximum_realized_absolute_loss_increase": float(np.max(realized_harm)),
        "budget_violations": int(np.sum(realized_harm > DELTA + 1e-12)),
        "range_violations": int(np.sum((projected < -1e-12) | (projected > 1.3 + 1e-12))),
        "monotonicity_violations": int(
            (predictions.sort_values(["cell_id", "target_cycle_number"]).groupby("cell_id")["pchp_sequence_transformer"].diff() > 1e-12).sum()
        ),
    }
    if any(deterministic[key] for key in ("budget_violations", "range_violations", "monotonicity_violations")):
        raise RuntimeError(f"deterministic audit failed: {deterministic}")

    paths = {
        "predictions": OUT / "pchp_sequence_transformer_predictions_v383.parquet",
        "cell_metrics": OUT / "pchp_sequence_transformer_cell_metrics_v383.csv",
        "summary": OUT / "pchp_sequence_transformer_summary_v383.csv",
        "report": OUT / "pchp_sequence_transformer_v383_report.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    report = {
        "status": "PCHP_SEQUENCE_TRANSFORMER_PORTABILITY_AUDIT_COMPLETED",
        "evidence_tier": "post-hoc source-only battery-sequence backbone portability audit",
        "retain_regardless_of_direction": True,
        "target_labels_used_for_training_calibration_or_selection": False,
        "architecture": {
            "family": "compact pre-normalized Transformer encoder",
            "channels": ["voltage", "current_C_rate", "charge_increment_per_nominal"],
            "time_grid_seconds": "0:10:600",
            "model_dimension": 64,
            "attention_heads": 4,
            "encoder_layers": 2,
            "feedforward_dimension": 128,
            "trainable_parameters": int(sum(p.numel() for p in models[0].parameters())),
        },
        "training": training,
        "target_caches": target_caches,
        "summary": json.loads(summary.to_json(orient="records")),
        "comparisons": comparisons(cells),
        "deterministic_certificate": deterministic,
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
