"""Thin wrapper over the vendored upstream RNAGym ``performance_fitness.py``.

This module is the *only* place gLMBench computes a benchmark metric, and it does so
by **calling** the byte-identical upstream functions in
``glmbench.tasks._vendor.rnagym.performance_fitness`` — it contains **no statistical
math of its own**. That is the design rule: the single biggest correctness
risk in the suite is a subtle bug in the metric reduction (tie handling, the
median-split boundary, AUC symmetry, the macro aggregation) silently corrupting every
leaderboard cell while all tests stay green. We remove that risk by deferring entirely
to the benchmark's own published definition and only adapting I/O here:

- :func:`assay_metrics` — numpy arrays in, a flat lowercase ``dict`` out — delegates to
  upstream ``calculate_metrics`` (abs-Spearman, median-split symmetric AUC, abs-MCC).
- :func:`macro` — applies upstream's **type-mean** reduction (mean within each RNA type,
  then mean across types) via ``calculate_RNA_types_averages_with_se``. For the v1
  prok-coding set this degenerates to a plain mean (all 11 assays share the single RNA
  type ``mRNA-coding``), but it is obtained through the upstream type-mean path so it
  stays correct the moment a second RNA type is added.

Core carries ``scipy`` + ``scikit-learn`` *only* because upstream imports them; they are
upstream's deps, not ours. Still no torch in core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from glmbench.tasks._vendor.rnagym import performance_fitness as _upstream

# Upstream's metric keys (the order it reports them in).
_UPSTREAM_KEYS = ("Spearman", "AUC", "MCC")
# The lowercase keys gLMBench exposes (I/O adaptation only).
_OUT_KEYS = ("spearman", "auc", "mcc")
# A fixed model-column name for the single-model dataframe handed to upstream's
# type-mean reducer; arbitrary (upstream builds ``{metric}_{model}`` column names
# from it) but kept stable for readability.
_MODEL_COL = "model"


def assay_metrics(
    dms_score: Sequence[float] | np.ndarray,
    model_score: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Per-assay metrics for one assay, via upstream ``calculate_metrics``.

    Returns ``{"spearman", "auc", "mcc"}`` — upstream's abs-Spearman, median-split
    symmetric AUC (``max(auc, 1-auc)``), and abs-MCC. The math is entirely upstream's;
    this only lowercases the keys and coerces inputs to ``float`` arrays.
    """
    dms = np.asarray(dms_score, dtype=float)
    model = np.asarray(model_score, dtype=float)
    res = _upstream.calculate_metrics(dms, model)
    return {out: float(res[up]) for out, up in zip(_OUT_KEYS, _UPSTREAM_KEYS, strict=True)}


def macro(
    per_assay: Sequence[dict[str, Any]],
    *,
    rna_types: Sequence[str] | None = None,
) -> dict[str, float]:
    """Type-mean macro over per-assay metrics, via upstream's reducer.

    Args:
        per_assay: one dict per assay, each with ``rna_type`` and the three per-assay
            metrics ``spearman`` / ``auc`` / ``mcc`` (as produced by :func:`assay_metrics`).
        rna_types: the RNA types to aggregate over; defaults to the sorted set present
            in ``per_assay``. (For the v1 prok set this is just ``["mRNA-coding"]``.)

    Returns:
        ``{"macro_spearman", "macro_auc", "macro_mcc"}`` — each the mean of per-RNA-type
        means (upstream's ``*_All_Mean``). A type whose assays are all-NaN for a metric
        propagates NaN into that macro, matching upstream.
    """
    if not per_assay:
        return {f"macro_{k}": float("nan") for k in _OUT_KEYS}

    # Build the single-model dataframe upstream's reducer expects: one row per assay,
    # an ``RNA_TYPE`` column, and ``{Metric}_{model}`` metric columns.
    df = pd.DataFrame(
        {
            "RNA_TYPE": [a["rna_type"] for a in per_assay],
            f"Spearman_{_MODEL_COL}": [a["spearman"] for a in per_assay],
            f"AUC_{_MODEL_COL}": [a["auc"] for a in per_assay],
            f"MCC_{_MODEL_COL}": [a["mcc"] for a in per_assay],
        }
    )
    types = list(rna_types) if rna_types is not None else sorted(df["RNA_TYPE"].unique())

    result = _upstream.calculate_RNA_types_averages_with_se(
        df, types, [_MODEL_COL], calculate_se=False
    )
    row = result.iloc[0]
    return {
        "macro_spearman": float(row["Spearman_All_Mean"]),
        "macro_auc": float(row["AUC_All_Mean"]),
        "macro_mcc": float(row["MCC_All_Mean"]),
    }
