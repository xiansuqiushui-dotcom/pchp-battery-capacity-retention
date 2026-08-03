"""Independent replay verifier for the PCHP V375 backbone-portability audit."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pchp_backbone_portability_v375"
REPORT = OUT / "pchp_backbone_portability_v375_report.json"
RECEIPT = OUT / "pchp_backbone_portability_v375_verification_receipt.json"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[str] = []

    def check(self, condition: bool, name: str) -> None:
        if not condition:
            raise AssertionError(name)
        self.checks.append(name)

    def close(self, left: float, right: float, name: str) -> None:
        self.check(bool(np.isclose(left, right, rtol=0.0, atol=TOLERANCE)), name)


def recompute_domain_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for (backbone, domain), rows in predictions.groupby(
        ["backbone", "domain"], sort=True
    ):
        for role in ("protected_state", "pchp_method", "raw_candidate"):
            working = rows[["cell_id"]].copy()
            working["absolute_error"] = np.abs(
                rows[role].to_numpy(float) - rows["truth"].to_numpy(float)
            )
            cells = (
                working.groupby("cell_id", as_index=False)
                .agg(cell_mae=("absolute_error", "mean"))
            )
            blocks.append(
                pd.DataFrame(
                    {
                        "backbone": backbone,
                        "domain": domain,
                        "method": role,
                        "cell_macro_mae": [float(cells["cell_mae"].mean())],
                        "physical_cells": [int(cells["cell_id"].nunique())],
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["backbone", "domain", "method"]
    ).reset_index(drop=True)


def recompute_comparison(
    domains: pd.DataFrame,
    backbone: str,
    bootstrap: dict[str, object],
) -> dict[str, object]:
    subset = domains.loc[domains["backbone"] == backbone]
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
    differences = (
        paired["cell_macro_mae_method"] - paired["cell_macro_mae_baseline"]
    ).to_numpy(float)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    indices = rng.integers(
        0,
        len(differences),
        size=(int(bootstrap["repetitions"]), len(differences)),
    )
    replicates = differences[indices].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "lower": float(np.quantile(replicates, bootstrap["lower_quantile"])),
        "upper": float(np.quantile(replicates, bootstrap["upper_quantile"])),
        "wins": int((differences < -TOLERANCE).sum()),
        "ties": int((np.abs(differences) <= TOLERANCE).sum()),
        "losses": int((differences > TOLERANCE).sum()),
    }


def maximum_trajectory_increase(
    rows: pd.DataFrame,
    values: np.ndarray,
) -> float:
    identifiers = rows["cell_id"].astype(str).to_numpy()
    cycles = rows["target_cycle_number"].to_numpy(float)
    maximum = -np.inf
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        if len(order) > 1:
            maximum = max(maximum, float(np.max(np.diff(values[order]))))
    return maximum


def verify_prefix_replay(
    audit: Audit,
    predictions: pd.DataFrame,
) -> None:
    maximum = 0.0
    for (backbone, domain), rows in predictions.groupby(
        ["backbone", "domain"], sort=True
    ):
        identifiers = rows["cell_id"].astype(str).to_numpy()
        cycles = rows["target_cycle_number"].to_numpy(float)
        raw = rows["raw_baseline"].to_numpy(float)
        candidate = rows["raw_candidate"].to_numpy(float)
        state = rows["protected_state"].to_numpy(float)
        method = rows["pchp_method"].to_numpy(float)
        alpha_values = rows["selected_alpha"].astype(float).unique()
        audit.check(
            len(alpha_values) == 1,
            f"V375 one frozen alpha in {backbone}/{domain}",
        )
        alpha = float(alpha_values[0])
        for identifier in np.unique(identifiers):
            positions = np.flatnonzero(identifiers == identifier)
            order = positions[np.argsort(cycles[positions], kind="mergesort")]
            length = max(1, len(order) // 2)
            prefix = order[:length]
            replay_state, replay_method = prefix_causal_cellwise_projection(
                np.repeat(identifier, length),
                cycles[prefix],
                raw[prefix],
                candidate[prefix],
                0.01,
                assimilation=alpha,
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(replay_state - state[prefix]))),
                float(np.max(np.abs(replay_method - method[prefix]))),
            )
    audit.check(maximum == 0.0, "V375 independent prefix replay exact")


def main() -> None:
    audit = Audit()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit.check(report["version"] == "v375", "V375 report version")
    audit.check(report["decision"] == "RETAIN", "V375 retained decision")
    audit.check(
        report["status"] == "PCHP_BACKBONE_PORTABILITY_RETAINED",
        "V375 retained status",
    )
    audit.check(report["conjunction_gate_passed"] is True, "V375 conjunction gate")
    for gate_group in report["gates"].values():
        for name, passed in gate_group.items():
            audit.check(passed is True, f"V375 reported gate: {name}")

    for name, identity in report["files"].items():
        path = Path(identity["path"])
        audit.check(path.is_file(), f"V375 file exists: {name}")
        audit.check(path.stat().st_size == identity["bytes"], f"V375 file size: {name}")
        audit.check(sha256_file(path) == identity["sha256"], f"V375 file hash: {name}")

    predictions_path = Path(report["files"]["predictions"]["path"])
    predictions = pd.read_parquet(predictions_path)
    audit.check(len(predictions) == 3 * 601_932, "V375 prediction row count")
    audit.check(predictions["backbone"].nunique() == 3, "V375 secondary backbone count")
    audit.check(not predictions.isna().any().any(), "V375 predictions contain no missing values")
    for backbone, rows in predictions.groupby("backbone"):
        audit.check(rows["domain"].nunique() == 12, f"V375 domain roster: {backbone}")
        audit.check(rows["cell_id"].nunique() == 586, f"V375 cell roster: {backbone}")
        audit.check(len(rows) == 601_932, f"V375 row roster: {backbone}")

    recomputed_domains = recompute_domain_metrics(predictions)
    saved_domains = pd.read_csv(report["files"]["domain_metrics"]["path"]).sort_values(
        ["backbone", "domain", "method"]
    ).reset_index(drop=True)
    audit.check(
        recomputed_domains[["backbone", "domain", "method", "physical_cells"]].equals(
            saved_domains[["backbone", "domain", "method", "physical_cells"]]
        ),
        "V375 independently recomputed domain identities",
    )
    audit.check(
        bool(
            np.allclose(
                recomputed_domains["cell_macro_mae"].to_numpy(float),
                saved_domains["cell_macro_mae"].to_numpy(float),
                rtol=0.0,
                atol=TOLERANCE,
            )
        ),
        "V375 independently recomputed domain MAE",
    )

    bootstrap = json.loads(
        (ROOT / "PCHP_BACKBONE_PORTABILITY_PREFREEZE_V375.json").read_text(
            encoding="utf-8"
        )
    )["bootstrap"]
    reported_comparisons = {
        item["backbone"]: item for item in report["comparisons"]
    }
    for backbone in ("ridge", "hist_gradient_boosting", "lightgbm"):
        recomputed = recompute_comparison(recomputed_domains, backbone, bootstrap)
        reported = reported_comparisons[backbone]
        audit.close(
            recomputed["mean"],
            reported["domain_equal_mean_difference"],
            f"V375 independently recomputed mean difference: {backbone}",
        )
        audit.close(
            recomputed["lower"],
            reported["bonferroni_adjusted_interval"][0],
            f"V375 independently recomputed adjusted lower endpoint: {backbone}",
        )
        audit.close(
            recomputed["upper"],
            reported["bonferroni_adjusted_interval"][1],
            f"V375 independently recomputed adjusted upper endpoint: {backbone}",
        )
        audit.check(
            [recomputed["wins"], recomputed["ties"], recomputed["losses"]]
            == reported["domain_wins_ties_losses"],
            f"V375 independently recomputed wins/ties/losses: {backbone}",
        )

        rows = predictions.loc[predictions["backbone"] == backbone]
        state = rows["protected_state"].to_numpy(float)
        method = rows["pchp_method"].to_numpy(float)
        truth = rows["truth"].to_numpy(float)
        certificate = report["deterministic_certificate_by_backbone"][backbone]
        audit.close(
            float(np.max(np.abs(method - state))),
            certificate["maximum_absolute_displacement"],
            f"V375 independent displacement certificate: {backbone}",
        )
        audit.close(
            float(np.max(np.abs(method - truth) - np.abs(state - truth))),
            certificate["maximum_observed_absolute_loss_regret"],
            f"V375 independent observed-regret certificate: {backbone}",
        )
        audit.close(
            maximum_trajectory_increase(rows, state),
            certificate["maximum_protected_state_increase"],
            f"V375 independent state monotonicity: {backbone}",
        )
        audit.close(
            maximum_trajectory_increase(rows, method),
            certificate["maximum_pchp_output_increase"],
            f"V375 independent output monotonicity: {backbone}",
        )
        audit.check(
            float(np.min(state)) >= 0.0
            and float(np.max(state)) <= 1.3
            and float(np.min(method)) >= 0.0
            and float(np.max(method)) <= 1.3,
            f"V375 independent physical range: {backbone}",
        )

    verify_prefix_replay(audit, predictions)
    for fold in report["folds"]:
        audit.check(
            fold["target_domain"] not in fold["source_domains"],
            f"V375 target excluded from sources: {fold['backbone']}/{fold['target_domain']}",
        )
        audit.check(
            fold["outer_target_labels_used_for_training_or_alpha_selection"] is False,
            f"V375 outer-label isolation: {fold['backbone']}/{fold['target_domain']}",
        )

    receipt = {
        "status": "PCHP_BACKBONE_PORTABILITY_V375_VERIFICATION_PASSED",
        "named_checks_passed": len(audit.checks),
        "checks": audit.checks,
        "report_sha256": sha256_file(REPORT),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "boundary": (
            "This receipt independently verifies the protocol-locked retrospective "
            "V375 secondary-backbone audit. It does not establish external confirmation, "
            "universal model independence, deep-model portability, or best-backbone superiority."
        ),
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
