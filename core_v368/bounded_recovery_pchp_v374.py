"""Prefix-causal harm-budget projection with a bounded recovery envelope.

The strict PCHP operator is recovered exactly when ``recovery_allowance=0``.
For a non-negative per-record schedule ``rho_t``, the protected state and the
issued output satisfy

    b_t - b_{t-1} <= rho_t,
    p_t - p_{t-1} <= rho_t,
    |p_t - b_t| <= delta.

The recovery allowance is an output-contract quantity.  It is not interpreted
as an electrochemical recovery rate by this module.
"""

from __future__ import annotations

import numpy as np


DEFAULT_LOWER = 0.0
DEFAULT_UPPER = 1.3
DEFAULT_TOLERANCE = 1e-12


def _validated_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_assimilation(value: float) -> float:
    assimilation = float(value)
    if not np.isfinite(assimilation) or not (0.0 < assimilation <= 1.0):
        raise ValueError("assimilation must be finite and lie in (0, 1]")
    return assimilation


def _validated_nonnegative_schedule(
    value: float | np.ndarray,
    length: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=float)
    if array.ndim != 1 or array.size != length:
        raise ValueError(f"{name} must be scalar or have one entry per record")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _validated_range(lower: float, upper: float) -> tuple[float, float]:
    lower_value = float(lower)
    upper_value = float(upper)
    if not np.isfinite(lower_value) or not np.isfinite(upper_value):
        raise ValueError("physical range endpoints must be finite")
    if not lower_value < upper_value:
        raise ValueError("physical lower endpoint must be below upper endpoint")
    return lower_value, upper_value


def causal_bounded_recovery_state(
    raw_prediction: np.ndarray,
    *,
    assimilation: float,
    recovery_allowance: float | np.ndarray,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> np.ndarray:
    """Construct the protected state with bounded positive innovations."""

    raw = _validated_vector(raw_prediction, "raw_prediction")
    alpha = _validated_assimilation(assimilation)
    recovery = _validated_nonnegative_schedule(
        recovery_allowance, raw.size, "recovery_allowance"
    )
    lower_value, upper_value = _validated_range(lower, upper)

    state = np.empty_like(raw)
    state[0] = np.clip(raw[0], lower_value, upper_value)
    for index in range(1, raw.size):
        innovation = raw[index] - state[index - 1]
        if innovation <= 0.0:
            step = alpha * innovation
        else:
            step = min(innovation, recovery[index])
        state[index] = np.clip(
            state[index - 1] + step,
            lower_value,
            upper_value,
        )
    return state


def prefix_causal_bounded_recovery_projection(
    raw_baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float | np.ndarray,
    *,
    assimilation: float,
    recovery_allowance: float | np.ndarray,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bounded-recovery protected state and projected output."""

    raw = _validated_vector(raw_baseline_prediction, "raw_baseline_prediction")
    candidate = _validated_vector(candidate_prediction, "candidate_prediction")
    if raw.shape != candidate.shape:
        raise ValueError("baseline and candidate predictions must have equal length")
    alpha = _validated_assimilation(assimilation)
    budgets = _validated_nonnegative_schedule(budget, raw.size, "budget")
    recovery = _validated_nonnegative_schedule(
        recovery_allowance, raw.size, "recovery_allowance"
    )
    lower_value, upper_value = _validated_range(lower, upper)
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    state = causal_bounded_recovery_state(
        raw,
        assimilation=alpha,
        recovery_allowance=recovery,
        lower=lower_value,
        upper=upper_value,
    )
    output = np.empty_like(candidate)
    previous = upper_value
    for index, candidate_value in enumerate(candidate):
        current_lower = max(lower_value, state[index] - budgets[index])
        current_upper = min(
            upper_value,
            state[index] + budgets[index],
            previous + recovery[index],
        )
        if current_lower > current_upper + tolerance_value:
            raise ValueError(
                "empty bounded-recovery interval; the state or contract schedule "
                "violates the declared recovery envelope"
            )
        if current_lower > current_upper:
            current_upper = current_lower
        output[index] = np.clip(candidate_value, current_lower, current_upper)
        previous = output[index]
    return state, output


def prefix_causal_bounded_recovery_cellwise_projection(
    cell_id: np.ndarray,
    cycle: np.ndarray,
    raw_baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float | np.ndarray,
    *,
    assimilation: float,
    recovery_allowance: float | np.ndarray,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the bounded-recovery operator independently to each cell."""

    identifiers = np.asarray(cell_id)
    cycles = _validated_vector(cycle, "cycle")
    raw = _validated_vector(raw_baseline_prediction, "raw_baseline_prediction")
    candidate = _validated_vector(candidate_prediction, "candidate_prediction")
    if identifiers.ndim != 1:
        raise ValueError("cell_id must be one-dimensional")
    if not (
        identifiers.size == cycles.size == raw.size == candidate.size
    ):
        raise ValueError("all record-level arrays must have equal length")
    budgets = _validated_nonnegative_schedule(budget, raw.size, "budget")
    recovery = _validated_nonnegative_schedule(
        recovery_allowance, raw.size, "recovery_allowance"
    )

    protected = np.empty_like(raw)
    projected = np.empty_like(candidate)
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        cell_state, cell_output = prefix_causal_bounded_recovery_projection(
            raw[order],
            candidate[order],
            budgets[order],
            assimilation=assimilation,
            recovery_allowance=recovery[order],
            lower=lower,
            upper=upper,
            tolerance=tolerance,
        )
        protected[order] = cell_state
        projected[order] = cell_output
    return protected, projected
