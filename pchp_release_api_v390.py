"""Validated public API for the frozen PCHP operator.

The scientific implementation used for the reported results remains unchanged
in ``updates_v389/prefix_causal_harm_projection_v321.py``.  This thin release
wrapper adds identifier validation before delegating to that hash-pinned code.
Valid inputs therefore produce exactly the frozen V321 outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from updates_v389.prefix_causal_harm_projection_v321 import (
    prefix_causal_cellwise_projection as _frozen_cellwise_projection,
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
    """Validate cell identities and apply the frozen V321 implementation."""

    identifiers = np.asarray(cell_id)
    if identifiers.ndim != 1:
        raise ValueError("cell_id must be a one-dimensional vector")
    missing = pd.isna(identifiers)
    empty = np.asarray(
        [isinstance(value, str) and not value.strip() for value in identifiers],
        dtype=bool,
    )
    if bool(np.any(missing)) or bool(np.any(empty)):
        raise ValueError("cell_id must not contain missing or empty identifiers")
    return _frozen_cellwise_projection(
        identifiers,
        cycle,
        raw_baseline,
        candidate_prediction,
        budget,
        assimilation=assimilation,
        y_min=y_min,
        y_max=y_max,
    )

