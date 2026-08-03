"""Prefix-causal monotone projection with deterministic loss-harm caps.

The operator consumes a causal non-increasing protected baseline ``b_t``, an
arbitrary causal candidate ``c_t``, and a non-negative harm budget ``delta``.
At each time it projects the current candidate onto

    [max(y_min, b_t-delta), min(y_max, b_t+delta, p_{t-1})].

For a constant budget and a non-increasing baseline, this interval is always
non-empty.  The update is prefix causal, non-increasing, and never moves more
than ``delta`` from the protected baseline.  Consequently, under absolute loss
its worst-case per-record loss increase relative to that baseline is at most
``delta`` for every possible outcome.

The accompanying one-sided causal smoother converts arbitrary raw baseline
predictions into a non-increasing protected state without future access:

    b_t = b_{t-1} + alpha * min(r_t - b_{t-1}, 0).

``alpha=1`` gives a running minimum; smaller positive values attenuate isolated
downward shocks while retaining prefix causality and monotonicity.

The generalized operator supports asymmetric absolute decision loss

    ell(p, y) = c_under * max(y-p, 0) + c_over * max(p-y, 0).

Its exact worst-case loss increase relative to a protected value ``b`` is
``c_over*(p-b)`` for ``p >= b`` and ``c_under*(b-p)`` for ``p < b``.  Hence a
decision-loss budget ``eta`` yields the necessary-and-sufficient tube
``[b-eta/c_under, b+eta/c_over]``.  Equal unit costs recover the original
absolute-loss PCHP exactly.

Budgets and directional costs may also vary by record.  In that case, let
``L_t=max(y_min, b_t-eta_t/c_under_t)`` be the standalone lower tube endpoint.
The current recursive interval is non-empty exactly when ``L_t <= p_{t-1}``.
It is non-empty for every possible candidate sequence exactly when the lower
endpoint sequence is non-increasing.  The implementation checks realized
feasibility and fails closed rather than silently enlarging a declared budget.
"""

from __future__ import annotations

import numpy as np


def _validated_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validated_positive_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _validated_schedule(
    values: float | np.ndarray,
    length: int,
    name: str,
    *,
    strictly_positive: bool,
) -> np.ndarray:
    """Return a finite scalar-broadcast or aligned one-dimensional schedule."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        array = np.full(length, scalar, dtype=float)
    elif array.ndim == 1 and array.size == length:
        array = array.astype(float, copy=True)
    else:
        raise ValueError(f"{name} must be scalar or an aligned one-dimensional schedule")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if strictly_positive:
        if (array <= 0.0).any():
            raise ValueError(f"{name} must be positive at every record")
    elif (array < 0.0).any():
        raise ValueError(f"{name} must be non-negative at every record")
    return array


def causal_nonincreasing_state(
    raw_prediction: np.ndarray,
    *,
    assimilation: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.3,
) -> np.ndarray:
    """Return a prefix-causal non-increasing baseline state.

    Parameters
    ----------
    raw_prediction:
        Predictions in temporal order for one physical unit.
    assimilation:
        Fraction of each downward innovation incorporated into the state.
        Must lie in ``(0, 1]``.  Upward innovations are ignored.
    y_min, y_max:
        Closed physical prediction range.
    """

    raw = _validated_vector(raw_prediction, "raw_prediction")
    if not np.isfinite(assimilation) or not 0.0 < assimilation <= 1.0:
        raise ValueError("assimilation must be finite and lie in (0, 1]")
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min > y_max:
        raise ValueError("prediction bounds must be finite and ordered")
    if raw.size == 0:
        return raw.copy()

    output = np.empty_like(raw)
    output[0] = np.clip(raw[0], y_min, y_max)
    for index in range(1, raw.size):
        innovation = min(raw[index] - output[index - 1], 0.0)
        output[index] = np.clip(
            output[index - 1] + assimilation * innovation,
            y_min,
            y_max,
        )
    return output


def worst_case_asymmetric_absolute_loss_increase(
    protected_baseline: np.ndarray,
    prediction: np.ndarray,
    *,
    underestimation_cost: float,
    overestimation_cost: float,
) -> np.ndarray:
    """Return the exact outcome-uniform asymmetric loss increase.

    ``underestimation_cost`` applies when a prediction is below the outcome;
    ``overestimation_cost`` applies when it is above the outcome.  The returned
    value is the supremum over every real-valued outcome of the prediction loss
    minus the protected-baseline loss.
    """

    baseline = _validated_vector(protected_baseline, "protected_baseline")
    values = _validated_vector(prediction, "prediction")
    if baseline.shape != values.shape:
        raise ValueError("protected baseline and prediction must have equal length")
    under = _validated_positive_scalar(
        underestimation_cost, "underestimation_cost"
    )
    over = _validated_positive_scalar(overestimation_cost, "overestimation_cost")
    displacement = values - baseline
    return np.where(displacement >= 0.0, over * displacement, under * -displacement)


def time_varying_asymmetric_harm_tube_bounds(
    protected_baseline: np.ndarray,
    budget: float | np.ndarray,
    *,
    underestimation_cost: float | np.ndarray,
    overestimation_cost: float | np.ndarray,
    y_min: float = 0.0,
    y_max: float = 1.3,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standalone exact tube endpoints for a time-varying contract.

    The returned upper endpoint does not yet include the previous published
    output.  The lower endpoint sequence is non-increasing if and only if the
    resulting recursive intervals are feasible for every candidate sequence.
    """

    baseline = _validated_vector(protected_baseline, "protected_baseline")
    budgets = _validated_schedule(
        budget,
        baseline.size,
        "budget",
        strictly_positive=False,
    )
    under = _validated_schedule(
        underestimation_cost,
        baseline.size,
        "underestimation_cost",
        strictly_positive=True,
    )
    over = _validated_schedule(
        overestimation_cost,
        baseline.size,
        "overestimation_cost",
        strictly_positive=True,
    )
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min > y_max:
        raise ValueError("prediction bounds must be finite and ordered")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if baseline.size == 0:
        return baseline.copy(), baseline.copy()
    if (baseline < y_min - tolerance).any() or (baseline > y_max + tolerance).any():
        raise ValueError("protected baseline lies outside prediction bounds")
    if (np.diff(baseline) > tolerance).any():
        raise ValueError("protected baseline must be non-increasing")
    lower = np.maximum(y_min, baseline - budgets / under)
    upper = np.minimum(y_max, baseline + budgets / over)
    return lower, upper


def minimum_viable_asymmetric_budget(
    protected_baseline: np.ndarray,
    previous_output: np.ndarray,
    *,
    underestimation_cost: float | np.ndarray,
) -> np.ndarray:
    """Return the exact per-record budget floor for realized feasibility.

    Each element is ``c_under * max(b - p_previous, 0)``.  The caller is
    responsible for aligning each protected value with the output published
    immediately before it.
    """

    baseline = _validated_vector(protected_baseline, "protected_baseline")
    previous = _validated_vector(previous_output, "previous_output")
    if baseline.shape != previous.shape:
        raise ValueError("protected baseline and previous output must have equal length")
    under = _validated_schedule(
        underestimation_cost,
        baseline.size,
        "underestimation_cost",
        strictly_positive=True,
    )
    return under * np.maximum(baseline - previous, 0.0)


def prefix_causal_time_varying_asymmetric_harm_projection(
    protected_baseline: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float | np.ndarray,
    *,
    underestimation_cost: float | np.ndarray,
    overestimation_cost: float | np.ndarray,
    y_min: float = 0.0,
    y_max: float = 1.3,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project under time-varying asymmetric budgets and directional costs.

    Schedule values at record ``t`` affect only record ``t``.  If a new lower
    tube endpoint exceeds the previous published output, the recursive
    interval is empty and the function raises ``RuntimeError``.  It never
    changes the requested budget to rescue feasibility.
    """

    baseline = _validated_vector(protected_baseline, "protected_baseline")
    candidate = _validated_vector(candidate_prediction, "candidate_prediction")
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate predictions must have equal length")
    lower, standalone_upper = time_varying_asymmetric_harm_tube_bounds(
        baseline,
        budget,
        underestimation_cost=underestimation_cost,
        overestimation_cost=overestimation_cost,
        y_min=y_min,
        y_max=y_max,
        tolerance=tolerance,
    )
    output = np.empty_like(baseline)
    previous = y_max
    for index, candidate_value in enumerate(candidate):
        upper = min(standalone_upper[index], previous)
        if lower[index] > upper + tolerance:
            raise RuntimeError(
                f"empty recursive feasible interval at index {index}: "
                f"lower={lower[index]:.17g}, previous_output={previous:.17g}"
            )
        current_lower = lower[index]
        if current_lower > upper:
            current_lower = upper
        output[index] = np.clip(candidate_value, current_lower, upper)
        previous = output[index]
    return output


def prefix_causal_asymmetric_harm_projection(
    protected_baseline: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float,
    *,
    underestimation_cost: float,
    overestimation_cost: float,
    y_min: float = 0.0,
    y_max: float = 1.3,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project a trajectory under an asymmetric absolute-loss harm budget.

    The returned value at time ``t`` depends only on inputs up to ``t``.  The
    protected baseline must already be non-increasing.  A scalar constant
    budget and constant positive costs are deliberate: they make both tube
    boundaries non-increasing with the protected state and therefore preserve
    recursive feasibility without an extra temporal condition.
    """

    return prefix_causal_time_varying_asymmetric_harm_projection(
        protected_baseline,
        candidate_prediction,
        budget,
        underestimation_cost=underestimation_cost,
        overestimation_cost=overestimation_cost,
        y_min=y_min,
        y_max=y_max,
        tolerance=tolerance,
    )


def prefix_causal_harm_projection(
    protected_baseline: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float,
    *,
    y_min: float = 0.0,
    y_max: float = 1.3,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project a trajectory under the original symmetric absolute-loss budget."""

    return prefix_causal_asymmetric_harm_projection(
        protected_baseline,
        candidate_prediction,
        budget,
        underestimation_cost=1.0,
        overestimation_cost=1.0,
        y_min=y_min,
        y_max=y_max,
        tolerance=tolerance,
    )


def prefix_causal_cellwise_projection(
    cell_id: np.ndarray,
    cycle: np.ndarray,
    raw_baseline: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float,
    *,
    assimilation: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the causal baseline and harm projection independently per cell."""

    identifiers = np.asarray(cell_id)
    cycles = _validated_vector(cycle, "cycle")
    raw = _validated_vector(raw_baseline, "raw_baseline")
    candidate = _validated_vector(candidate_prediction, "candidate_prediction")
    if not (identifiers.ndim == 1 and identifiers.size == cycles.size == raw.size == candidate.size):
        raise ValueError("cell ids, cycles, and predictions must be aligned vectors")

    baseline_output = np.empty_like(raw)
    projected_output = np.empty_like(raw)
    groups: dict[object, list[int]] = {}
    for position, identifier in enumerate(identifiers.tolist()):
        groups.setdefault(identifier, []).append(position)
    for positions_list in groups.values():
        positions = np.asarray(positions_list, dtype=int)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        baseline = causal_nonincreasing_state(
            raw[order],
            assimilation=assimilation,
            y_min=y_min,
            y_max=y_max,
        )
        projected = prefix_causal_harm_projection(
            baseline,
            candidate[order],
            budget,
            y_min=y_min,
            y_max=y_max,
        )
        baseline_output[order] = baseline
        projected_output[order] = projected
    return baseline_output, projected_output
