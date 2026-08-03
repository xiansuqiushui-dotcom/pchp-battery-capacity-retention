from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    prefix_causal_harm_projection,
)


SEED = 20260802
TOLERANCE = 2e-12
RANDOM_TRAJECTORIES = 30_000
GRID = np.array([-0.2, 0.0, 0.4, 0.9, 1.3, 1.6], dtype=float)
ALPHAS = (0.05, 0.5, 1.0)
BUDGETS = (0.0, 0.01, 0.2)
Y_MIN = 0.0
Y_MAX = 1.3


def protected_step(previous: float, raw: float, alpha: float) -> float:
    innovation = min(raw - previous, 0.0)
    return float(np.clip(previous + alpha * innovation, Y_MIN, Y_MAX))


def projection_step(
    baseline: float, candidate: float, previous: float, budget: float
) -> float:
    lower = max(Y_MIN, baseline - budget)
    upper = min(Y_MAX, baseline + budget, previous)
    if lower > upper + TOLERANCE:
        raise ValueError("infeasible local state")
    return float(np.clip(candidate, lower, upper))


def exhaustive_local_protected_checks() -> int:
    count = 0
    states = tuple(itertools.product(GRID, repeat=2))
    for alpha in ALPHAS:
        for left in states:
            left_output = protected_step(left[0], left[1], alpha)
            for right in states:
                right_output = protected_step(right[0], right[1], alpha)
                input_radius = max(
                    abs(left[0] - right[0]), abs(left[1] - right[1])
                )
                if abs(left_output - right_output) > input_radius + TOLERANCE:
                    raise AssertionError(
                        ("protected_local", alpha, left, right, input_radius)
                    )
                count += 1
    return count


def exhaustive_local_projection_checks() -> int:
    count = 0
    for budget in BUDGETS:
        states: list[tuple[float, float, float]] = []
        for baseline, candidate, previous in itertools.product(GRID, repeat=3):
            lower = max(Y_MIN, baseline - budget)
            upper = min(Y_MAX, baseline + budget, previous)
            if Y_MIN <= baseline <= Y_MAX and lower <= upper + TOLERANCE:
                states.append((baseline, candidate, previous))
        for left in states:
            left_output = projection_step(*left, budget)
            for right in states:
                right_output = projection_step(*right, budget)
                input_radius = max(abs(a - b) for a, b in zip(left, right))
                if abs(left_output - right_output) > input_radius + TOLERANCE:
                    raise AssertionError(
                        ("projection_local", budget, left, right, input_radius)
                    )
                count += 1
    return count


def random_prefix_trajectory_checks() -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    maximum_baseline_excess = -np.inf
    maximum_output_excess = -np.inf
    for trial in range(RANDOM_TRAJECTORIES):
        length = int(rng.integers(1, 80))
        alpha = float(rng.uniform(1e-5, 1.0))
        budget = float(rng.uniform(0.0, 0.2))
        perturbation_radius = 10.0 ** float(rng.uniform(-9.0, -0.7))
        raw = rng.uniform(-0.5, 1.8, length)
        candidate = rng.uniform(-0.5, 1.8, length)
        raw_shift = rng.uniform(
            -perturbation_radius, perturbation_radius, length
        )
        candidate_shift = rng.uniform(
            -perturbation_radius, perturbation_radius, length
        )

        baseline = causal_nonincreasing_state(raw, assimilation=alpha)
        changed_baseline = causal_nonincreasing_state(
            raw + raw_shift, assimilation=alpha
        )
        projected = prefix_causal_harm_projection(baseline, candidate, budget)
        changed_projected = prefix_causal_harm_projection(
            changed_baseline, candidate + candidate_shift, budget
        )

        raw_prefix_radius = np.maximum.accumulate(np.abs(raw_shift))
        joint_prefix_radius = np.maximum(
            raw_prefix_radius,
            np.maximum.accumulate(np.abs(candidate_shift)),
        )
        baseline_prefix_radius = np.maximum.accumulate(
            np.abs(baseline - changed_baseline)
        )
        output_prefix_radius = np.maximum.accumulate(
            np.abs(projected - changed_projected)
        )
        baseline_excess = float(
            np.max(baseline_prefix_radius - raw_prefix_radius)
        )
        output_excess = float(np.max(output_prefix_radius - joint_prefix_radius))
        maximum_baseline_excess = max(maximum_baseline_excess, baseline_excess)
        maximum_output_excess = max(maximum_output_excess, output_excess)
        if baseline_excess > TOLERANCE or output_excess > TOLERANCE:
            raise AssertionError(
                (
                    "trajectory",
                    trial,
                    baseline_excess,
                    output_excess,
                    perturbation_radius,
                )
            )
    return {
        "random_trajectory_pairs": RANDOM_TRAJECTORIES,
        "maximum_baseline_excess": maximum_baseline_excess,
        "maximum_output_excess": maximum_output_excess,
    }


def proximal_equivalence_checks() -> int:
    count = 0
    for alpha in ALPHAS:
        for previous, raw in itertools.product(GRID, repeat=2):
            previous_in_range = float(np.clip(previous, Y_MIN, Y_MAX))
            actual = protected_step(previous_in_range, raw, alpha)
            unconstrained = (1.0 - alpha) * previous_in_range + alpha * raw
            proximal = float(
                np.clip(unconstrained, Y_MIN, min(Y_MAX, previous_in_range))
            )
            if abs(actual - proximal) > TOLERANCE:
                raise AssertionError(
                    ("proximal", alpha, previous_in_range, raw, actual, proximal)
                )
            count += 1
    return count


def tightness_check() -> float:
    epsilon = 0.03
    raw = np.array([0.90, 0.85, 0.80])
    candidate = raw.copy()
    baseline = causal_nonincreasing_state(raw, assimilation=0.5)
    changed_baseline = causal_nonincreasing_state(
        raw + epsilon, assimilation=0.5
    )
    projected = prefix_causal_harm_projection(baseline, candidate, 0.1)
    changed_projected = prefix_causal_harm_projection(
        changed_baseline, candidate + epsilon, 0.1
    )
    attained = float(np.max(np.abs(changed_projected - projected)))
    if abs(attained - epsilon) > TOLERANCE:
        raise AssertionError(("tightness", attained, epsilon))
    return attained


def main() -> None:
    report = {
        "schema": "pchp_prefix_nonexpansiveness_v377",
        "seed": SEED,
        "absolute_tolerance": TOLERANCE,
        "exhaustive_local_protected_checks": exhaustive_local_protected_checks(),
        "exhaustive_local_projection_checks": exhaustive_local_projection_checks(),
        "proximal_equivalence_checks": proximal_equivalence_checks(),
        **random_prefix_trajectory_checks(),
        "tightness_attained": tightness_check(),
        "status": "PCHP_PREFIX_NONEXPANSIVENESS_V377_PASSED",
    }
    output = Path(__file__).with_name(
        "prefix_nonexpansiveness_v377_report.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
