"""Deterministic regret-capped prediction updates.

For a baseline prediction b, candidate prediction c, and non-negative budget
delta, the projected prediction is b + clip(c - b, -delta, delta).  Under
absolute loss, its per-case loss increase relative to b is at most delta for
every possible outcome.  Any non-negative weighted average of cases inherits
the corresponding weighted budget bound without exchangeability assumptions.

The construction is also the exact Euclidean projection of the candidate onto
the set of predictions whose worst-case absolute-loss regret relative to the
baseline is at most ``delta``.  Indeed, the worst-case regret equals the
absolute displacement from the baseline.
"""

from __future__ import annotations

import numpy as np


def worst_case_absolute_loss_regret(
    baseline_prediction: np.ndarray,
    proposed_prediction: np.ndarray,
) -> np.ndarray:
    """Return the exact worst-case absolute-loss regret over all real outcomes.

    For scalar predictions ``b`` and ``p``,

    ``sup_y |p-y| - |b-y| = |p-b|``.

    The equality follows from the reverse triangle inequality for the upper
    bound and is attained by taking an outcome beyond both predictions in the
    direction opposite to the update.
    """

    baseline = np.asarray(baseline_prediction, dtype=float)
    proposed = np.asarray(proposed_prediction, dtype=float)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline and proposed predictions must have the same shape")
    if not np.isfinite(baseline).all() or not np.isfinite(proposed).all():
        raise ValueError("predictions must be finite")
    return np.abs(proposed - baseline)


def regret_capped_projection(
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float | np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline_prediction, dtype=float)
    candidate = np.asarray(candidate_prediction, dtype=float)
    budgets = np.asarray(budget, dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate predictions must have the same shape")
    try:
        budgets = np.broadcast_to(budgets, baseline.shape)
    except ValueError as error:
        raise ValueError("budget is not broadcast-compatible with predictions") from error
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or not np.isfinite(budgets).all()
    ):
        raise ValueError("predictions and budgets must be finite")
    if (budgets < 0.0).any():
        raise ValueError("budgets must be non-negative")
    update = np.clip(candidate - baseline, -budgets, budgets)
    return baseline + update


def absolute_loss_regret(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    projected_prediction: np.ndarray,
) -> np.ndarray:
    truth_array = np.asarray(truth, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    projected = np.asarray(projected_prediction, dtype=float)
    if truth_array.shape != baseline.shape or baseline.shape != projected.shape:
        raise ValueError("truth and prediction arrays must have the same shape")
    if (
        not np.isfinite(truth_array).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(projected).all()
    ):
        raise ValueError("truth and predictions must be finite")
    return np.abs(projected - truth_array) - np.abs(baseline - truth_array)


def verify_absolute_loss_budget(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    budget: float | np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> bool:
    projected = regret_capped_projection(
        baseline_prediction, candidate_prediction, budget
    )
    budgets = np.broadcast_to(np.asarray(budget, dtype=float), projected.shape)
    regret = absolute_loss_regret(truth, baseline_prediction, projected)
    return bool(np.all(regret <= budgets + tolerance))
