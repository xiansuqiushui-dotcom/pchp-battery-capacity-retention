"""Compatibility import for the immutable V368 test suite.

The V368 snapshot records a historical package-qualified import path that was
not present in the review-lite archive.  This module restores that path without
changing the hash-pinned implementation in ``core_v368``.
"""

from core_v368.prefix_causal_harm_projection_v321 import *  # noqa: F401,F403

