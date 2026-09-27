"""Measured diagnostics computed off passes the benchmark already pays for.

Each module here answers one question about a model that its *score* cannot: the score is a
scalar at the end of a stack, and a stack can be sick in ways a scalar hides. They are
observational — they report, they never gate a run — with one exception: a layer sweep never
picks its best layer from a numerically empty tap (:func:`~.depth_health.usable_layers`).
"""

from __future__ import annotations

from .depth_health import (
    DEGENERATE_RMS_ABS,
    DEGENERATE_RMS_REL,
    DepthHealth,
    depth_health,
    unit_roundoff,
)

__all__ = [
    "DEGENERATE_RMS_ABS",
    "DEGENERATE_RMS_REL",
    "DepthHealth",
    "depth_health",
    "unit_roundoff",
]
