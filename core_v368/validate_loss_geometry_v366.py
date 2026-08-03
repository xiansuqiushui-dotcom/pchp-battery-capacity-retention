"""Deterministic falsification tests for the frozen V366 loss-geometry extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parent
PAPER_ROOT = ROOT / "paper_q1"
PREFREEZE = PAPER_ROOT / "LOSS_GEOMETRY_EXTENSION_PREFREEZE_V366_ZH.md"
OUTPUT = ROOT / "loss_geometry_extension_v366"
REPORT = OUTPUT / "loss_geometry_extension_v366_report.json"

SEED = 20260802
TOL = 1e-12
N_FORMULA = 200_000
N_METRIC = 20_000
N_TRAJECTORIES = 2_000
TRAJECTORY_LENGTH = 128


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
        if not bool(condition):
            raise AssertionError(name)
        self.checks.append(name)

    def close(self, observed: np.ndarray | float, expected: np.ndarray | float, name: str) -> float:
        difference = float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
        self.check(difference <= TOL, name)
        return difference


def squared_harm(p: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    at_lower = (p - lower) ** 2 - (b - lower) ** 2
    at_upper = (p - upper) ** 2 - (b - upper) ** 2
    return np.maximum(at_lower, at_upper)


def squared_interval(
    b: np.ndarray, budget: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_lower = upper - np.sqrt((upper - b) ** 2 + budget)
    raw_upper = lower + np.sqrt((b - lower) ** 2 + budget)
    return (
        np.maximum(lower, raw_lower),
        np.minimum(upper, raw_upper),
        raw_lower,
        raw_upper,
    )


def symbolic_checks(audit: Audit) -> dict[str, str]:
    p, b, y, lower, upper = sp.symbols("p b y lower upper", real=True)
    difference = sp.expand((p - y) ** 2 - (b - y) ** 2)
    audit.check(sp.simplify(difference - (p - b) * (p + b - 2 * y)) == 0, "symbolic squared-loss affine identity")
    audit.check(
        sp.simplify((p - b) * (p + b - 2 * lower) - ((p - lower) ** 2 - (b - lower) ** 2)) == 0,
        "symbolic lower-endpoint branch identity",
    )
    audit.check(
        sp.simplify((b - p) * (2 * upper - p - b) - ((upper - p) ** 2 - (upper - b) ** 2)) == 0,
        "symbolic upper-endpoint branch identity",
    )
    return {
        "squared_difference": str(sp.factor(difference)),
        "positive_branch": "(p-lower)^2-(b-lower)^2",
        "negative_branch": "(upper-p)^2-(upper-b)^2",
    }


def metric_checks(rng: np.random.Generator, audit: Audit) -> dict[str, float]:
    scalar_b = rng.normal(size=N_METRIC)
    scalar_p = rng.normal(size=N_METRIC)
    scalar_distance = np.abs(scalar_p - scalar_b)
    scalar_y = rng.normal(size=(N_METRIC, 16))
    scalar_y[:, 0] = scalar_b
    scalar_sup = np.max(
        np.abs(scalar_p[:, None] - scalar_y) - np.abs(scalar_b[:, None] - scalar_y),
        axis=1,
    )
    scalar_error = audit.close(scalar_sup, scalar_distance, "scalar metric identity")

    dimension = 7
    vector_b = rng.normal(size=(N_METRIC, dimension))
    vector_p = rng.normal(size=(N_METRIC, dimension))
    vector_distance = np.linalg.norm(vector_p - vector_b, axis=1)
    vector_y = rng.normal(size=(N_METRIC, 12, dimension))
    vector_y[:, 0, :] = vector_b
    vector_sup = np.max(
        np.linalg.norm(vector_p[:, None, :] - vector_y, axis=2)
        - np.linalg.norm(vector_b[:, None, :] - vector_y, axis=2),
        axis=1,
    )
    vector_error = audit.close(vector_sup, vector_distance, "Euclidean metric identity")
    return {
        "scalar_max_absolute_error": scalar_error,
        "euclidean_max_absolute_error": vector_error,
    }


def squared_formula_checks(rng: np.random.Generator, audit: Audit) -> dict[str, float | int]:
    lower = rng.uniform(-1.0, 0.25, size=N_FORMULA)
    width = rng.uniform(0.05, 2.0, size=N_FORMULA)
    upper = lower + width
    b = lower + rng.random(N_FORMULA) * width
    p = lower + rng.random(N_FORMULA) * width
    budget = rng.random(N_FORMULA) * (1.5 * width**2)

    endpoint_harm = squared_harm(p, b, lower, upper)
    formula_harm = np.where(
        p >= b,
        (p - b) * (p + b - 2 * lower),
        (b - p) * (2 * upper - p - b),
    )
    formula_error = audit.close(endpoint_harm, formula_harm, "bounded squared-loss supremum formula")

    feasible_lower, feasible_upper, raw_lower, raw_upper = squared_interval(b, budget, lower, upper)
    audit.check(np.all(feasible_lower <= b + TOL), "squared feasible lower contains baseline")
    audit.check(np.all(feasible_upper + TOL >= b), "squared feasible upper contains baseline")
    audit.check(np.all(feasible_lower <= feasible_upper + TOL), "squared feasible intervals nonempty")

    boundary_lower_harm = squared_harm(feasible_lower, b, lower, upper)
    boundary_upper_harm = squared_harm(feasible_upper, b, lower, upper)
    audit.check(np.all(boundary_lower_harm <= budget + TOL), "lower boundary respects squared budget")
    audit.check(np.all(boundary_upper_harm <= budget + TOL), "upper boundary respects squared budget")

    active_lower = raw_lower >= lower
    active_upper = raw_upper <= upper
    lower_boundary_error = audit.close(
        boundary_lower_harm[active_lower], budget[active_lower], "active lower boundary is exact"
    )
    upper_boundary_error = audit.close(
        boundary_upper_harm[active_upper], budget[active_upper], "active upper boundary is exact"
    )

    midpoint = 0.5 * (feasible_lower + feasible_upper)
    audit.check(
        np.all(squared_harm(midpoint, b, lower, upper) <= budget + TOL),
        "squared feasible interval interior respects budget",
    )

    lower_step = np.maximum(1e-10, width * 1e-7)
    lower_exterior = feasible_lower - lower_step
    lower_testable = lower_exterior >= lower
    audit.check(
        np.all(squared_harm(lower_exterior[lower_testable], b[lower_testable], lower[lower_testable], upper[lower_testable]) > budget[lower_testable]),
        "every testable lower exterior point violates budget",
    )

    upper_step = np.maximum(1e-10, width * 1e-7)
    upper_exterior = feasible_upper + upper_step
    upper_testable = upper_exterior <= upper
    audit.check(
        np.all(squared_harm(upper_exterior[upper_testable], b[upper_testable], lower[upper_testable], upper[upper_testable]) > budget[upper_testable]),
        "every testable upper exterior point violates budget",
    )

    return {
        "supremum_formula_max_absolute_error": formula_error,
        "active_lower_boundary_max_absolute_error": lower_boundary_error,
        "active_upper_boundary_max_absolute_error": upper_boundary_error,
        "active_lower_boundaries": int(np.sum(active_lower)),
        "active_upper_boundaries": int(np.sum(active_upper)),
        "testable_lower_exterior_points": int(np.sum(lower_testable)),
        "testable_upper_exterior_points": int(np.sum(upper_testable)),
    }


def recursive_checks(rng: np.random.Generator, audit: Audit) -> dict[str, float | int]:
    maximum_budget_excess = -np.inf
    minimum_interval_width = np.inf
    maximum_increase = -np.inf
    all_intervals_nonempty = True
    all_outputs_in_range = True
    all_outputs_non_increasing = True
    all_budgets_respected = True
    for _ in range(N_TRAJECTORIES):
        lower = float(rng.uniform(-0.2, 0.1))
        upper = float(lower + rng.uniform(0.5, 1.8))
        budget = float(rng.uniform(0.0, 0.4 * (upper - lower) ** 2))
        protected = np.empty(TRAJECTORY_LENGTH)
        protected[0] = rng.uniform(lower, upper)
        for t in range(1, TRAJECTORY_LENGTH):
            protected[t] = max(lower, protected[t - 1] - rng.exponential((upper - lower) / 40.0))
        candidate = rng.uniform(lower - 0.5 * (upper - lower), upper + 0.5 * (upper - lower), size=TRAJECTORY_LENGTH)

        previous = upper
        for t in range(TRAJECTORY_LENGTH):
            interval_lower, standalone_upper, _, _ = squared_interval(
                np.array([protected[t]]),
                np.array([budget]),
                np.array([lower]),
                np.array([upper]),
            )
            current_lower = float(interval_lower[0])
            current_upper = min(float(standalone_upper[0]), previous)
            all_intervals_nonempty &= current_lower <= current_upper + TOL
            output = float(np.clip(candidate[t], current_lower, current_upper))
            harm = float(
                squared_harm(
                    np.array([output]),
                    np.array([protected[t]]),
                    np.array([lower]),
                    np.array([upper]),
                )[0]
            )
            maximum_budget_excess = max(maximum_budget_excess, harm - budget)
            minimum_interval_width = min(minimum_interval_width, current_upper - current_lower)
            maximum_increase = max(maximum_increase, output - previous)
            all_outputs_in_range &= lower - TOL <= output <= upper + TOL
            all_outputs_non_increasing &= output <= previous + TOL
            all_budgets_respected &= harm <= budget + TOL
            previous = output

    audit.check(all_intervals_nonempty, "all recursive squared intervals nonempty")
    audit.check(all_outputs_in_range, "all recursive squared outputs in physical range")
    audit.check(all_outputs_non_increasing, "all recursive squared outputs non-increasing")
    audit.check(all_budgets_respected, "all recursive squared outputs respect budget")
    audit.check(maximum_budget_excess <= TOL, "all recursive squared budgets pass")
    audit.check(minimum_interval_width >= -TOL, "all recursive squared widths pass")
    audit.check(maximum_increase <= TOL, "all recursive squared monotonicity checks pass")
    return {
        "trajectories": N_TRAJECTORIES,
        "records_per_trajectory": TRAJECTORY_LENGTH,
        "maximum_budget_excess": float(maximum_budget_excess),
        "minimum_interval_width": float(minimum_interval_width),
        "maximum_output_increase": float(maximum_increase),
    }


def unbounded_squared_checks(audit: Audit) -> dict[str, list[float]]:
    magnitudes = np.array([1.0, 10.0, 100.0, 1_000.0, 10_000.0])
    b = 0.2
    p_above = 0.7
    y_negative = -magnitudes
    above_harm = (p_above - y_negative) ** 2 - (b - y_negative) ** 2
    audit.check(np.all(np.diff(above_harm) > 0.0), "unbounded squared harm diverges for upward update")

    p_below = -0.4
    y_positive = magnitudes
    below_harm = (p_below - y_positive) ** 2 - (b - y_positive) ** 2
    audit.check(np.all(np.diff(below_harm) > 0.0), "unbounded squared harm diverges for downward update")
    audit.check(above_harm[-1] > 1_000.0, "upward unbounded counterexample exceeds finite budget")
    audit.check(below_harm[-1] > 1_000.0, "downward unbounded counterexample exceeds finite budget")
    return {
        "magnitudes": magnitudes.tolist(),
        "upward_update_harm": above_harm.tolist(),
        "downward_update_harm": below_harm.tolist(),
    }


def main() -> int:
    if not PREFREEZE.is_file():
        raise FileNotFoundError(PREFREEZE)
    rng = np.random.default_rng(SEED)
    audit = Audit()
    symbolic = symbolic_checks(audit)
    metric = metric_checks(rng, audit)
    squared = squared_formula_checks(rng, audit)
    recursive = recursive_checks(rng, audit)
    unbounded = unbounded_squared_checks(audit)

    report = {
        "status": "RETAIN_FOR_V366",
        "seed": SEED,
        "tolerance": TOL,
        "prefreeze_path": PREFREEZE.relative_to(ROOT).as_posix(),
        "prefreeze_sha256": sha256_file(PREFREEZE),
        "named_checks_passed": len(audit.checks),
        "checks": audit.checks,
        "symbolic": symbolic,
        "metric_identity": metric,
        "bounded_squared_loss": squared,
        "recursive_squared_projection": recursive,
        "unbounded_squared_loss": unbounded,
        "interpretation_boundary": (
            "The tests validate the frozen formulas and recursion only. They do not prove literature priority, "
            "empirical utility under squared loss, deployment cost calibration, or electrochemical safety."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
