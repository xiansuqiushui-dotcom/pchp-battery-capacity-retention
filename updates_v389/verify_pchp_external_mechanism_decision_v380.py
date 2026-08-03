"""Independent replay verifier for the frozen PCHP V380 audit."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "external_mechanism_decision_v380"
REPORT = OUT / "external_mechanism_decision_v380_report.json"
RECEIPT = OUT / "external_mechanism_decision_v380_verification_receipt.json"
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


def cell_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    truth = predictions["truth"].to_numpy(float)
    for method in ("protected_state", "fixed_shift", "pchp_method", "raw_candidate"):
        working = predictions[["domain", "cell_id"]].copy()
        working["method"] = method
        working["absolute_error"] = np.abs(
            predictions[method].to_numpy(float) - truth
        )
        blocks.append(
            working.groupby(["domain", "cell_id", "method"], as_index=False)
            .agg(mae=("absolute_error", "mean"), records=("absolute_error", "size"))
        )
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["domain", "cell_id", "method"]
    ).reset_index(drop=True)


def difference_map(
    long_metrics: pd.DataFrame, left: str, right: str, value: str
) -> dict[str, np.ndarray]:
    selected = long_metrics.loc[
        long_metrics["method"].isin([left, right]),
        ["domain", "cell_id", "method", value],
    ]
    wide = selected.pivot(
        index=["domain", "cell_id"], columns="method", values=value
    ).reset_index()
    wide["difference"] = wide[left] - wide[right]
    return {
        domain: rows["difference"].to_numpy(float)
        for domain, rows in wide.groupby("domain", sort=True)
    }


def bootstrap(
    values: dict[str, np.ndarray], repetitions: int, seed: int
) -> tuple[float, list[float]]:
    names = sorted(values)
    point = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    replicates = np.empty(repetitions, dtype=float)
    chunk = 2500
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        count = stop - start
        sampled_surfaces = rng.integers(0, len(names), size=(count, len(names)))
        totals = np.zeros(count, dtype=float)
        for position in range(len(names)):
            chosen = sampled_surfaces[:, position]
            contribution = np.zeros(count, dtype=float)
            for surface_index, name in enumerate(names):
                rows = np.flatnonzero(chosen == surface_index)
                if len(rows):
                    vector = values[name]
                    indices = rng.integers(
                        0, len(vector), size=(len(rows), len(vector))
                    )
                    contribution[rows] = vector[indices].mean(axis=1)
            totals += contribution
        replicates[start:stop] = totals / len(names)
    return point, [
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    ]


def maximum_increase(predictions: pd.DataFrame, values: np.ndarray) -> float:
    identifiers = predictions["cell_id"].astype(str).to_numpy()
    cycles = predictions["target_cycle_number"].to_numpy(float)
    maximum = 0.0
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        if len(order) > 1:
            maximum = max(maximum, float(np.max(np.diff(values[order]))))
    return maximum


def verify_prefix_replay(audit: Audit, predictions: pd.DataFrame) -> None:
    identifiers = predictions["cell_id"].astype(str).to_numpy()
    cycles = predictions["target_cycle_number"].to_numpy(float)
    raw = predictions["raw_baseline"].to_numpy(float)
    candidate = predictions["raw_candidate"].to_numpy(float)
    state = predictions["protected_state"].to_numpy(float)
    method = predictions["pchp_method"].to_numpy(float)
    maximum = 0.0
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        prefix = order[: max(1, len(order) // 2)]
        replay_state, replay_method = prefix_causal_cellwise_projection(
            np.repeat(identifier, len(prefix)),
            cycles[prefix],
            raw[prefix],
            candidate[prefix],
            0.01,
            assimilation=0.01,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(replay_state - state[prefix]))),
            float(np.max(np.abs(replay_method - method[prefix]))),
        )
    audit.check(maximum == 0.0, "V380 independent prefix replay exact")


def decision_cells(predictions: pd.DataFrame) -> pd.DataFrame:
    truth = predictions["truth"].to_numpy(float)
    blocks = []
    for method in ("protected_state", "fixed_shift", "pchp_method"):
        values = predictions[method].to_numpy(float)
        premature = (truth > 0.8) & (values <= 0.8)
        missed = (truth <= 0.8) & (values > 0.8)
        working = predictions[["domain", "cell_id"]].copy()
        working["method"] = method
        working["binary_cost"] = premature.astype(float) + 5.0 * missed.astype(float)
        blocks.append(
            working.groupby(["domain", "cell_id", "method"], as_index=False)
            .agg(binary_cost=("binary_cost", "mean"))
        )
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    audit = Audit()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    audit.check(report["version"] == "v380", "V380 report version")
    audit.check(report["decision"] == "RETAIN", "V380 mechanism retained")
    audit.check(
        report["status"] == "V380_EXTERNAL_MECHANISM_RETAINED",
        "V380 retained status",
    )
    audit.check(
        report["prefreeze"]["sha256"]
        == "128fa29a2b21ad8022adf3afecc6351e8b480e770c4c8833a032c7097575b5cc",
        "V380 immutable prefreeze identity",
    )
    for name, passed in report["primary_mechanism_gates"].items():
        audit.check(passed is True, f"V380 reported mechanism gate: {name}")
    audit.check(
        report["primary_decision_cost"]["decision_claim_retained"] is False,
        "V380 threshold decision non-claim preserved",
    )

    for name, identity in report["files"].items():
        path = Path(identity["path"])
        audit.check(path.is_file(), f"V380 file exists: {name}")
        audit.check(path.stat().st_size == identity["bytes"], f"V380 size: {name}")
        audit.check(sha256_file(path) == identity["sha256"], f"V380 hash: {name}")

    predictions = pd.read_parquet(report["files"]["predictions"]["path"])
    audit.check(len(predictions) == 9712, "V380 external prediction rows")
    audit.check(predictions["domain"].nunique() == 6, "V380 external surfaces")
    audit.check(predictions["cell_id"].nunique() == 659, "V380 external cells")
    audit.check(not predictions.isna().any().any(), "V380 predictions complete")

    recomputed_cells = cell_metric_table(predictions)
    saved_cells = pd.read_csv(report["files"]["cell_metrics"]["path"]).sort_values(
        ["domain", "cell_id", "method"]
    ).reset_index(drop=True)
    audit.check(
        recomputed_cells[["domain", "cell_id", "method", "records"]].equals(
            saved_cells[["domain", "cell_id", "method", "records"]]
        ),
        "V380 independently recomputed cell identities",
    )
    audit.check(
        bool(
            np.allclose(
                recomputed_cells["mae"].to_numpy(float),
                saved_cells["mae"].to_numpy(float),
                rtol=0.0,
                atol=TOLERANCE,
            )
        ),
        "V380 independently recomputed cell MAE",
    )
    values = difference_map(recomputed_cells, "pchp_method", "fixed_shift", "mae")
    point, interval = bootstrap(values, 100000, 20260803)
    reported = report["primary_mechanism_estimand"]
    audit.close(point, reported["pchp_minus_fixed_shift"], "V380 mechanism effect")
    audit.close(interval[0], reported["two_stage_95_percent_interval"][0], "V380 mechanism lower")
    audit.close(interval[1], reported["two_stage_95_percent_interval"][1], "V380 mechanism upper")

    truth = predictions["truth"].to_numpy(float)
    state = predictions["protected_state"].to_numpy(float)
    method = predictions["pchp_method"].to_numpy(float)
    certificate = report["deterministic_certificate"]
    audit.close(
        float(np.max(np.abs(method - state))),
        certificate["maximum_absolute_displacement"],
        "V380 displacement certificate",
    )
    audit.close(
        float(np.max(np.abs(method - truth) - np.abs(state - truth))),
        certificate["maximum_observed_absolute_loss_regret"],
        "V380 observed regret certificate",
    )
    audit.close(
        maximum_increase(predictions, state),
        certificate["maximum_protected_state_increase"],
        "V380 protected state monotonicity",
    )
    audit.close(
        maximum_increase(predictions, method),
        certificate["maximum_pchp_output_increase"],
        "V380 PCHP monotonicity",
    )
    verify_prefix_replay(audit, predictions)

    decisions = decision_cells(predictions)
    fixed_map = difference_map(decisions, "pchp_method", "fixed_shift", "binary_cost")
    protected_map = difference_map(
        decisions, "pchp_method", "protected_state", "binary_cost"
    )
    fixed_point, fixed_interval = bootstrap(fixed_map, 100000, 20260803)
    protected_point, protected_interval = bootstrap(protected_map, 100000, 20260803)
    decision = report["primary_decision_cost"]
    audit.close(fixed_point, decision["pchp_minus_fixed_shift"], "V380 threshold fixed effect")
    audit.close(fixed_interval[0], decision["pchp_minus_fixed_shift_two_stage_95_percent_interval"][0], "V380 threshold fixed lower")
    audit.close(fixed_interval[1], decision["pchp_minus_fixed_shift_two_stage_95_percent_interval"][1], "V380 threshold fixed upper")
    audit.close(protected_point, decision["pchp_minus_protected_state"], "V380 threshold protected effect")
    audit.close(protected_interval[0], decision["pchp_minus_protected_two_stage_95_percent_interval"][0], "V380 threshold protected lower")
    audit.close(protected_interval[1], decision["pchp_minus_protected_two_stage_95_percent_interval"][1], "V380 threshold protected upper")

    receipt = {
        "status": "PCHP_EXTERNAL_MECHANISM_DECISION_V380_VERIFICATION_PASSED",
        "named_checks_passed": len(audit.checks),
        "checks": audit.checks,
        "report_sha256": sha256_file(REPORT),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "audit_classification": {
            "mechanism_inference": "PASS",
            "deterministic_certificate": "PASS",
            "threshold_decision_benefit": "PASS_NON_CLAIM",
        },
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
