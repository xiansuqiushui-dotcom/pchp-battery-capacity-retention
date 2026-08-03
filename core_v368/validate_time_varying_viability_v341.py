"""Independent falsification harness for the V341 time-varying PCHP claim.

This script implements the gates frozen in
``paper_q1/TIME_VARYING_VIABILITY_PREFREEZE_V341.json``.  It deliberately
combines exhaustive finite-grid checks, randomized property tests, and
constructive counterexamples.  Any failed gate produces a non-zero exit code
and a report with status ``REJECT``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prefix_causal_harm_projection_v321 import (
    minimum_viable_asymmetric_budget,
    prefix_causal_asymmetric_harm_projection,
    prefix_causal_harm_projection,
    prefix_causal_time_varying_asymmetric_harm_projection,
    time_varying_asymmetric_harm_tube_bounds,
)


ROOT = Path(__file__).resolve().parent
PREFREEZE = ROOT / "paper_q1" / "TIME_VARYING_VIABILITY_PREFREEZE_V341.json"
REPORT_DIR = ROOT / "paper_q1" / "time_varying_viability_v341"
REPORT_PATH = REPORT_DIR / "time_varying_viability_v341_report.json"
SEED = 20260802
TOLERANCE = 1e-12
Y_MIN = 0.0
Y_MAX = 1.3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def failure_index(error: RuntimeError) -> int:
    match = re.search(r"index (\d+)", str(error))
    if match is None:
        raise AssertionError(f"runtime failure did not expose its index: {error}")
    return int(match.group(1))


def directional_harm(
    baseline: np.ndarray,
    prediction: np.ndarray,
    under: np.ndarray,
    over: np.ndarray,
) -> np.ndarray:
    return np.where(
        prediction >= baseline,
        over * (prediction - baseline),
        under * (baseline - prediction),
    )


def exhaustive_small_grid() -> dict[str, Any]:
    baseline_grid = (0.4, 0.8, 1.2)
    under_grid = (0.5, 2.0)
    budget_grid = (0.0, 0.2, 0.6)
    candidate_grid = (-1.0, 0.65, 2.0)
    over = np.array([0.7, 3.0, 1.5])
    schedules = 0
    trajectories = 0
    viable_schedules = 0
    nonviable_schedules = 0

    baselines = [
        np.asarray(values, dtype=float)
        for values in itertools.product(baseline_grid, repeat=3)
        if all(values[index] >= values[index + 1] for index in range(2))
    ]
    for baseline in baselines:
        for under_values in itertools.product(under_grid, repeat=3):
            under = np.asarray(under_values, dtype=float)
            for budget_values in itertools.product(budget_grid, repeat=3):
                budget = np.asarray(budget_values, dtype=float)
                lower, _ = time_varying_asymmetric_harm_tube_bounds(
                    baseline,
                    budget,
                    underestimation_cost=under,
                    overestimation_cost=over,
                    tolerance=TOLERANCE,
                )
                analytic_universal = bool(np.all(np.diff(lower) <= TOLERANCE))
                empirical_universal = True
                for candidate_values in itertools.product(candidate_grid, repeat=3):
                    candidate = np.asarray(candidate_values, dtype=float)
                    trajectories += 1
                    try:
                        prefix_causal_time_varying_asymmetric_harm_projection(
                            baseline,
                            candidate,
                            budget,
                            underestimation_cost=under,
                            overestimation_cost=over,
                            tolerance=TOLERANCE,
                        )
                    except RuntimeError:
                        empirical_universal = False
                require(
                    empirical_universal == analytic_universal,
                    "finite-grid universal verdict disagrees with lower-endpoint condition",
                )
                schedules += 1
                viable_schedules += int(analytic_universal)
                nonviable_schedules += int(not analytic_universal)

    require(viable_schedules > 0, "grid contains no viable schedules")
    require(nonviable_schedules > 0, "grid contains no nonviable schedules")
    return {
        "schedules": schedules,
        "candidate_trajectories": trajectories,
        "viable_schedules": viable_schedules,
        "nonviable_schedules": nonviable_schedules,
    }


def randomized_viable_schedules() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 1)
    schedules = 300
    candidates_per_schedule = 40
    checked_prefixes = 0
    checked_records = 0
    for _ in range(schedules):
        length = int(rng.integers(8, 81))
        baseline = np.minimum.accumulate(rng.uniform(0.2, 1.2, length))
        proposed_lower = baseline - rng.uniform(0.0, 0.18, length)
        lower = np.maximum(Y_MIN, np.minimum.accumulate(proposed_lower))
        under = rng.uniform(0.1, 12.0, length)
        over = rng.uniform(0.1, 12.0, length)
        budget = under * (baseline - lower)
        observed_lower, _ = time_varying_asymmetric_harm_tube_bounds(
            baseline,
            budget,
            underestimation_cost=under,
            overestimation_cost=over,
            tolerance=TOLERANCE,
        )
        require(
            bool(np.all(np.diff(observed_lower) <= TOLERANCE)),
            "constructed viable lower schedule is not non-increasing",
        )
        for _ in range(candidates_per_schedule):
            candidate = rng.uniform(-0.5, 1.8, length)
            output = prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
                tolerance=TOLERANCE,
            )
            require(bool(np.all(np.diff(output) <= TOLERANCE)), "output increased")
            require(
                bool(np.all((output >= Y_MIN - TOLERANCE) & (output <= Y_MAX + TOLERANCE))),
                "output left physical bounds",
            )
            harm = directional_harm(baseline, output, under, over)
            require(bool(np.all(harm <= budget + TOLERANCE)), "harm budget violated")
            prefix_length = int(rng.integers(1, length + 1))
            prefix = prefix_causal_time_varying_asymmetric_harm_projection(
                baseline[:prefix_length],
                candidate[:prefix_length],
                budget[:prefix_length],
                underestimation_cost=under[:prefix_length],
                overestimation_cost=over[:prefix_length],
                tolerance=TOLERANCE,
            )
            require(np.array_equal(prefix, output[:prefix_length]), "prefix invariance failed")
            checked_prefixes += 1
            checked_records += length
    return {
        "schedules": schedules,
        "candidate_trajectories": schedules * candidates_per_schedule,
        "records": checked_records,
        "prefix_replays": checked_prefixes,
    }


def constructive_failures() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 2)
    trials = 1000
    matched_failure_indices = 0
    for _ in range(trials):
        length = int(rng.integers(3, 41))
        baseline = np.linspace(1.2, 0.8, length)
        lower = np.minimum.accumulate(rng.uniform(0.15, 0.45, length))
        index = int(rng.integers(1, length))
        lower[index] = lower[index - 1] + rng.uniform(1e-4, 0.10)
        require(lower[index] < baseline[index], "counterexample lower endpoint left tube domain")
        under = rng.uniform(0.2, 10.0, length)
        over = rng.uniform(0.2, 10.0, length)
        budget = under * (baseline - lower)
        candidate = np.full(length, -10.0)
        try:
            prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
                tolerance=TOLERANCE,
            )
        except RuntimeError as error:
            observed_index = failure_index(error)
            require(observed_index == index, "constructive failure occurred at wrong index")
            matched_failure_indices += 1
        else:
            raise AssertionError("upward lower endpoint did not produce frozen counterexample")
    return {"counterexamples": trials, "matched_failure_indices": matched_failure_indices}


def realized_prefix_equivalence() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 3)
    trials = 3000
    feasible = 0
    infeasible = 0
    for _ in range(trials):
        length = int(rng.integers(1, 61))
        baseline = np.minimum.accumulate(rng.uniform(0.05, 1.25, length))
        budget = rng.uniform(0.0, 0.7, length)
        under = rng.uniform(0.1, 15.0, length)
        over = rng.uniform(0.1, 15.0, length)
        candidate = rng.uniform(-0.5, 1.8, length)
        lower, standalone_upper = time_varying_asymmetric_harm_tube_bounds(
            baseline,
            budget,
            underestimation_cost=under,
            overestimation_cost=over,
            tolerance=TOLERANCE,
        )
        manual = np.empty(length)
        previous = Y_MAX
        expected_failure: int | None = None
        for index in range(length):
            if lower[index] > previous + TOLERANCE:
                expected_failure = index
                break
            upper = min(standalone_upper[index], previous)
            current_lower = min(lower[index], upper)
            manual[index] = np.clip(candidate[index], current_lower, upper)
            previous = manual[index]
        try:
            output = prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                budget,
                underestimation_cost=under,
                overestimation_cost=over,
                tolerance=TOLERANCE,
            )
        except RuntimeError as error:
            require(expected_failure is not None, "implementation failed despite feasible prefix")
            require(failure_index(error) == expected_failure, "realized failure index mismatch")
            infeasible += 1
        else:
            require(expected_failure is None, "implementation accepted an empty interval")
            require(
                np.allclose(output, manual, atol=TOLERANCE, rtol=0.0),
                "implementation output differs from manual recursion",
            )
            feasible += 1
    require(feasible > 0 and infeasible > 0, "random audit did not exercise both verdicts")
    return {"trajectories": trials, "feasible": feasible, "infeasible": infeasible}


def minimum_budget_exactness() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 4)
    active_trials = 1000
    inactive_trials = 500
    rejected_below = 0
    for _ in range(active_trials):
        baseline_value = float(rng.uniform(0.2, 1.2))
        previous = float(rng.uniform(0.0, baseline_value - 0.01))
        baseline = np.array([baseline_value, baseline_value])
        under = rng.uniform(0.1, 10.0, 2)
        over = rng.uniform(0.1, 10.0, 2)
        floor = float(
            minimum_viable_asymmetric_budget(
                baseline[1:],
                np.array([previous]),
                underestimation_cost=under[1:],
            )[0]
        )
        expected = under[1] * (baseline_value - previous)
        require(abs(floor - expected) <= TOLERANCE, "minimum-budget formula mismatch")
        first_budget = under[0] * (baseline_value - previous)
        candidate = np.array([previous, Y_MAX])
        equality = prefix_causal_time_varying_asymmetric_harm_projection(
            baseline,
            candidate,
            np.array([first_budget, floor]),
            underestimation_cost=under,
            overestimation_cost=over,
            tolerance=TOLERANCE,
        )
        require(abs(equality[0] - previous) <= TOLERANCE, "failed to realize prior output")
        require(abs(equality[1] - previous) <= TOLERANCE, "budget equality was not feasible")
        decrement = max(1e-9, floor * 1e-8)
        require(decrement < floor, "active floor too small for below-bound audit")
        try:
            prefix_causal_time_varying_asymmetric_harm_projection(
                baseline,
                candidate,
                np.array([first_budget, floor - decrement]),
                underestimation_cost=under,
                overestimation_cost=over,
                tolerance=TOLERANCE,
            )
        except RuntimeError as error:
            require(failure_index(error) == 1, "below-floor failure occurred at wrong index")
            rejected_below += 1
        else:
            raise AssertionError("budget immediately below exact floor remained feasible")

    for _ in range(inactive_trials):
        baseline_value = float(rng.uniform(0.1, 1.0))
        previous = float(rng.uniform(baseline_value, 1.2))
        under = rng.uniform(0.1, 10.0, 2)
        over = rng.uniform(0.1, 10.0, 2)
        floor = minimum_viable_asymmetric_budget(
            np.array([baseline_value]),
            np.array([previous]),
            underestimation_cost=under[1:],
        )
        require(float(floor[0]) == 0.0, "inactive minimum budget was not zero")
        first_budget = over[0] * (previous - baseline_value)
        output = prefix_causal_time_varying_asymmetric_harm_projection(
            np.array([baseline_value, baseline_value]),
            np.array([previous, baseline_value]),
            np.array([first_budget, 0.0]),
            underestimation_cost=under,
            overestimation_cost=over,
            tolerance=TOLERANCE,
        )
        require(abs(output[1] - baseline_value) <= TOLERANCE, "zero floor was infeasible")
    return {
        "active_trials": active_trials,
        "rejected_immediately_below": rejected_below,
        "inactive_zero_floor_trials": inactive_trials,
    }


def upper_cost_invariance() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 5)
    trials = 2000
    feasible = 0
    infeasible = 0
    for _ in range(trials):
        length = int(rng.integers(2, 61))
        baseline = np.minimum.accumulate(rng.uniform(0.05, 1.25, length))
        budget = rng.uniform(0.0, 0.5, length)
        under = rng.uniform(0.1, 12.0, length)
        over_a = rng.uniform(0.05, 0.5, length)
        over_b = rng.uniform(20.0, 200.0, length)
        candidate = rng.uniform(-0.5, 1.8, length)
        lower_a, _ = time_varying_asymmetric_harm_tube_bounds(
            baseline,
            budget,
            underestimation_cost=under,
            overestimation_cost=over_a,
            tolerance=TOLERANCE,
        )
        lower_b, _ = time_varying_asymmetric_harm_tube_bounds(
            baseline,
            budget,
            underestimation_cost=under,
            overestimation_cost=over_b,
            tolerance=TOLERANCE,
        )
        require(np.array_equal(lower_a, lower_b), "upper cost changed lower endpoint")

        verdicts: list[tuple[bool, int | None, np.ndarray | None]] = []
        for over in (over_a, over_b):
            try:
                output = prefix_causal_time_varying_asymmetric_harm_projection(
                    baseline,
                    candidate,
                    budget,
                    underestimation_cost=under,
                    overestimation_cost=over,
                    tolerance=TOLERANCE,
                )
            except RuntimeError as error:
                verdicts.append((False, failure_index(error), None))
            else:
                harm = directional_harm(baseline, output, under, over)
                require(bool(np.all(harm <= budget + TOLERANCE)), "directional certificate failed")
                verdicts.append((True, None, output))
        require(verdicts[0][0] == verdicts[1][0], "upper cost changed feasibility verdict")
        require(verdicts[0][1] == verdicts[1][1], "upper cost changed failure index")
        if verdicts[0][0]:
            feasible += 1
        else:
            infeasible += 1
    require(feasible > 0 and infeasible > 0, "invariance audit lacked both verdicts")
    return {"trajectories": trials, "feasible": feasible, "infeasible": infeasible}


def constant_schedule_recovery() -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 6)
    trials = 1000
    for _ in range(trials):
        length = int(rng.integers(1, 101))
        baseline = np.minimum.accumulate(rng.uniform(0.0, Y_MAX, length))
        candidate = rng.uniform(-0.5, 1.8, length)
        budget = float(rng.uniform(0.0, 0.4))
        under = float(rng.uniform(0.05, 20.0))
        over = float(rng.uniform(0.05, 20.0))
        legacy_asymmetric = prefix_causal_asymmetric_harm_projection(
            baseline,
            candidate,
            budget,
            underestimation_cost=under,
            overestimation_cost=over,
            tolerance=TOLERANCE,
        )
        scheduled_asymmetric = prefix_causal_time_varying_asymmetric_harm_projection(
            baseline,
            candidate,
            np.full(length, budget),
            underestimation_cost=np.full(length, under),
            overestimation_cost=np.full(length, over),
            tolerance=TOLERANCE,
        )
        require(
            np.array_equal(legacy_asymmetric, scheduled_asymmetric),
            "constant asymmetric schedule did not recover V340 exactly",
        )
        legacy_symmetric = prefix_causal_harm_projection(
            baseline,
            candidate,
            budget,
            tolerance=TOLERANCE,
        )
        scheduled_symmetric = prefix_causal_time_varying_asymmetric_harm_projection(
            baseline,
            candidate,
            np.full(length, budget),
            underestimation_cost=np.ones(length),
            overestimation_cost=np.ones(length),
            tolerance=TOLERANCE,
        )
        require(
            np.array_equal(legacy_symmetric, scheduled_symmetric),
            "constant symmetric schedule did not recover original API exactly",
        )
    return {"random_trajectories": trials, "exact_array_equalities": 2 * trials}


def invalid_contracts_fail_closed() -> dict[str, Any]:
    baseline = np.array([0.9, 0.8])
    candidate = np.array([0.9, 0.8])
    cases: list[Callable[[], Any]] = [
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, np.array([0.01]), underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, np.array([0.01, -0.01]), underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, np.array([0.01, np.nan]), underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, 0.01, underestimation_cost=np.array([1.0, 0.0]), overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, 0.01, underestimation_cost=1.0, overestimation_cost=np.array([1.0, np.inf])
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            np.array([0.8, 0.9]), candidate, 0.01, underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            np.array([0.9, 1.4]), candidate, 0.01, underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, np.array([0.9]), 0.01, underestimation_cost=1.0, overestimation_cost=1.0
        ),
        lambda: prefix_causal_time_varying_asymmetric_harm_projection(
            baseline, candidate, 0.01, underestimation_cost=1.0, overestimation_cost=1.0, tolerance=-1.0
        ),
        lambda: minimum_viable_asymmetric_budget(
            baseline, np.array([0.9]), underestimation_cost=1.0
        ),
        lambda: minimum_viable_asymmetric_budget(
            baseline, baseline, underestimation_cost=np.array([1.0, -1.0])
        ),
    ]
    rejected = 0
    for case in cases:
        try:
            case()
        except (ValueError, RuntimeError):
            rejected += 1
        else:
            raise AssertionError("invalid contract was accepted")
    return {"invalid_cases": len(cases), "rejected": rejected}


def existing_unit_suite() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", "test_prefix_causal_harm_projection_v321.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    require(completed.returncode == 0, completed.stdout + completed.stderr)
    match = re.search(r"Ran (\d+) tests", completed.stderr + completed.stdout)
    require(match is not None, "unit-test output did not report test count")
    return {"tests": int(match.group(1)), "return_code": completed.returncode}


def main() -> int:
    with PREFREEZE.open("r", encoding="utf-8") as handle:
        prefreeze = json.load(handle)
    require(prefreeze["falsification"]["seed"] == SEED, "seed differs from prefreeze")

    gates: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("exhaustive_small_grid", exhaustive_small_grid),
        ("randomized_viable_schedules", randomized_viable_schedules),
        ("constructive_upward_boundary_failures", constructive_failures),
        ("realized_prefix_equivalence", realized_prefix_equivalence),
        ("minimum_budget_exactness", minimum_budget_exactness),
        ("upper_cost_feasibility_invariance", upper_cost_invariance),
        ("constant_schedule_exact_recovery", constant_schedule_recovery),
        ("invalid_contracts_fail_closed", invalid_contracts_fail_closed),
        ("existing_unit_suite", existing_unit_suite),
    ]
    results: list[dict[str, Any]] = []
    for gate_id, gate in gates:
        try:
            evidence = gate()
        except Exception as error:  # report every frozen failure before rejecting
            results.append(
                {
                    "gate": gate_id,
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        else:
            results.append({"gate": gate_id, "status": "PASS", "evidence": evidence})

    passed = all(item["status"] == "PASS" for item in results)
    report = {
        "version": "v341",
        "report_date": prefreeze["date"],
        "seed": SEED,
        "tolerance": TOLERANCE,
        "prefreeze": relative(PREFREEZE),
        "prefreeze_sha256": sha256(PREFREEZE),
        "implementation": relative(ROOT / "prefix_causal_harm_projection_v321.py"),
        "implementation_sha256": sha256(ROOT / "prefix_causal_harm_projection_v321.py"),
        "unit_tests": relative(ROOT / "test_prefix_causal_harm_projection_v321.py"),
        "unit_tests_sha256": sha256(ROOT / "test_prefix_causal_harm_projection_v321.py"),
        "decision": "RETAIN" if passed else "REJECT",
        "status": "PASS" if passed else "FAIL",
        "gates": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
