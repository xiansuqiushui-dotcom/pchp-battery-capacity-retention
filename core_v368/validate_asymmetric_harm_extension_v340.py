"""Failure-closed validation for the asymmetric PCHP decision-loss extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from prefix_causal_harm_projection_v321 import (
    prefix_causal_asymmetric_harm_projection,
    prefix_causal_harm_projection,
    worst_case_asymmetric_absolute_loss_increase,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_q1" / "asymmetric_harm_extension_v340"
SEED = 20_260_802
TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def asymmetric_loss(
    prediction: np.ndarray,
    outcome: np.ndarray,
    underestimation_cost: np.ndarray,
    overestimation_cost: np.ndarray,
) -> np.ndarray:
    return underestimation_cost * np.maximum(outcome - prediction, 0.0) + (
        overestimation_cost * np.maximum(prediction - outcome, 0.0)
    )


def legacy_symmetric_reference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    budget: float,
    *,
    y_min: float = 0.0,
    y_max: float = 1.3,
) -> np.ndarray:
    output = np.empty_like(baseline)
    previous = y_max
    for index, (base, candidate_value) in enumerate(zip(baseline, candidate)):
        lower = max(y_min, base - budget)
        upper = min(y_max, base + budget, previous)
        if lower > upper + TOL:
            raise RuntimeError("legacy reference interval is empty")
        if lower > upper:
            lower = upper
        output[index] = np.clip(candidate_value, lower, upper)
        previous = output[index]
    return output


def main() -> None:
    rng = np.random.default_rng(SEED)
    checks: dict[str, object] = {}

    sample_count = 10_000
    baseline = rng.normal(0.8, 0.5, sample_count)
    prediction = rng.normal(0.8, 0.5, sample_count)
    under = rng.uniform(0.1, 10.0, sample_count)
    over = rng.uniform(0.1, 10.0, sample_count)
    analytic = np.where(
        prediction >= baseline,
        over * (prediction - baseline),
        under * (baseline - prediction),
    )
    api = worst_case_asymmetric_absolute_loss_increase(
        baseline,
        prediction,
        underestimation_cost=1.0,
        overestimation_cost=1.0,
    )
    unit_cost_expected = np.abs(prediction - baseline)
    checks["unit_cost_api_recovers_absolute_displacement"] = bool(
        np.allclose(api, unit_cost_expected, rtol=0.0, atol=TOL)
    )

    outcomes = np.stack(
        [
            np.minimum(baseline, prediction) - 1.0,
            baseline,
            prediction,
            np.maximum(baseline, prediction) + 1.0,
        ],
        axis=1,
    )
    direct = (
        asymmetric_loss(
            prediction[:, None], outcomes, under[:, None], over[:, None]
        )
        - asymmetric_loss(baseline[:, None], outcomes, under[:, None], over[:, None])
    ).max(axis=1)
    supremum_error = float(np.max(np.abs(analytic - direct)))
    checks["exact_supremum_piecewise_replay"] = supremum_error <= TOL

    budgets = rng.uniform(1e-4, 0.2, sample_count)
    direction = rng.choice(np.array([-1.0, 1.0]), size=sample_count)
    inside_fraction = rng.uniform(0.0, 1.0, sample_count)
    directional_radius = np.where(direction < 0.0, budgets / under, budgets / over)
    inside = baseline + direction * inside_fraction * directional_radius
    outside = baseline + direction * 1.0001 * directional_radius
    inside_harm = np.where(
        inside >= baseline, over * (inside - baseline), under * (baseline - inside)
    )
    outside_harm = np.where(
        outside >= baseline, over * (outside - baseline), under * (baseline - outside)
    )
    checks["inside_exact_tube_satisfies_budget"] = bool(
        (inside_harm <= budgets + TOL).all()
    )
    checks["outside_exact_tube_constructively_violates_budget"] = bool(
        (outside_harm > budgets).all()
    )

    equal_cost_trials = 100
    equal_cost_exact = True
    generalized_properties = True
    prefix_invariance = True
    for _ in range(equal_cost_trials):
        raw = rng.uniform(0.1, 1.2, 80)
        state = np.minimum.accumulate(raw)
        candidate = rng.uniform(-0.2, 1.5, 80)
        budget = float(rng.uniform(0.0, 0.1))
        legacy = legacy_symmetric_reference(state, candidate, budget)
        symmetric = prefix_causal_harm_projection(state, candidate, budget)
        generalized = prefix_causal_asymmetric_harm_projection(
            state,
            candidate,
            budget,
            underestimation_cost=1.0,
            overestimation_cost=1.0,
        )
        equal_cost_exact &= bool(
            np.array_equal(legacy, symmetric)
            and np.array_equal(symmetric, generalized)
        )

        under_cost = float(rng.uniform(0.2, 8.0))
        over_cost = float(rng.uniform(0.2, 8.0))
        eta = float(rng.uniform(0.0, 0.2))
        projected = prefix_causal_asymmetric_harm_projection(
            state,
            candidate,
            eta,
            underestimation_cost=under_cost,
            overestimation_cost=over_cost,
        )
        harm = worst_case_asymmetric_absolute_loss_increase(
            state,
            projected,
            underestimation_cost=under_cost,
            overestimation_cost=over_cost,
        )
        generalized_properties &= bool(
            (np.diff(projected) <= TOL).all()
            and (projected >= -TOL).all()
            and (projected <= 1.3 + TOL).all()
            and (harm <= eta + TOL).all()
        )
        for length in (1, 2, 5, 31, 80):
            prefix = prefix_causal_asymmetric_harm_projection(
                state[:length],
                candidate[:length],
                eta,
                underestimation_cost=under_cost,
                overestimation_cost=over_cost,
            )
            prefix_invariance &= bool(np.array_equal(prefix, projected[:length]))
    checks["equal_cost_exact_legacy_recovery"] = equal_cost_exact
    checks["generalized_online_properties"] = generalized_properties
    checks["generalized_prefix_invariance"] = prefix_invariance

    base = np.array([0.9])
    eta = 0.08
    up = prefix_causal_asymmetric_harm_projection(
        base,
        np.array([2.0]),
        eta,
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    down = prefix_causal_asymmetric_harm_projection(
        base,
        np.array([-2.0]),
        eta,
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    checks["directional_cost_radii"] = bool(
        np.isclose(up[0] - base[0], eta / 8.0, rtol=0.0, atol=TOL)
        and np.isclose(base[0] - down[0], eta / 2.0, rtol=0.0, atol=TOL)
    )
    zero = prefix_causal_asymmetric_harm_projection(
        np.array([1.0, 0.9, 0.8]),
        np.array([0.2, 1.2, 0.1]),
        0.0,
        underestimation_cost=2.0,
        overestimation_cost=8.0,
    )
    checks["zero_budget_identity"] = bool(
        np.array_equal(zero, np.array([1.0, 0.9, 0.8]))
    )

    invalid_contracts_fail = True
    for under_cost, over_cost in (
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, np.inf),
    ):
        try:
            prefix_causal_asymmetric_harm_projection(
                np.array([0.9, 0.8]),
                np.array([0.9, 0.8]),
                0.01,
                underestimation_cost=under_cost,
                overestimation_cost=over_cost,
            )
        except ValueError:
            pass
        else:
            invalid_contracts_fail = False
    checks["invalid_cost_contracts_fail_closed"] = invalid_contracts_fail

    passed = all(bool(value) for value in checks.values())
    if not passed:
        failed = [name for name, value in checks.items() if not bool(value)]
        raise AssertionError(f"asymmetric extension gate failed: {failed}")

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "asymmetric_harm_extension_v340_report.json"
    report = {
        "status": "ASYMMETRIC_HARM_EXTENSION_RETAINED",
        "decision": "RETAIN",
        "seed": SEED,
        "random_scalar_tuples": sample_count,
        "random_trajectory_trials": equal_cost_trials,
        "maximum_direct_supremum_error": supremum_error,
        "checks": checks,
        "exact_result": {
            "loss": "c_under*max(y-p,0)+c_over*max(p-y,0)",
            "supremum": "c_over*(p-b) for p>=b; c_under*(b-p) for p<b",
            "harm_set": "[b-eta/c_under, b+eta/c_over]",
            "equal_cost_recovery": "c_under=c_over=1 recovers absolute-loss PCHP",
        },
        "scope": (
            "Costs and budget are positive constants within a trajectory; "
            "the extension does not claim empirical deployment cost calibration."
        ),
        "artifacts": [
            {
                "path": "paper_q1/ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json",
                "sha256": sha256_file(
                    ROOT / "paper_q1/ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json"
                ),
            },
            {
                "path": "prefix_causal_harm_projection_v321.py",
                "sha256": sha256_file(ROOT / "prefix_causal_harm_projection_v321.py"),
            },
            {
                "path": "test_prefix_causal_harm_projection_v321.py",
                "sha256": sha256_file(
                    ROOT / "test_prefix_causal_harm_projection_v321.py"
                ),
            },
            {
                "path": "validate_asymmetric_harm_extension_v340.py",
                "sha256": sha256_file(Path(__file__)),
            },
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
