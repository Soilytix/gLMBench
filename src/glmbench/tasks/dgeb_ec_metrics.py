"""Thin wrapper over the vendored DGEB ``logRegClassificationEvaluator``.

This is the *only* place the EC classification task computes its score, and it does so by
**calling** the byte-identical upstream evaluator in
``glmbench.tasks._vendor.dgeb.evaluators`` — it contains **no statistical math of its
own**. The benchmark's definition (the sklearn ``LogisticRegression`` probe and the
``f1_score(average="macro")`` / ``accuracy_score`` reduction) lives entirely upstream; we
only adapt I/O (feature matrices + labels in, a flat ``dict[str, float]`` out).

Standardization (the optional ``StandardScaler``) and the per-layer/window aggregation are
**task logic**, applied *before* the arrays reach this wrapper — so the vendored evaluator
stays a pure, untouched copy of upstream.

Core (torch-free): numpy + scikit-learn only (already core deps via the RNAGym wrapper).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from glmbench.tasks._vendor.dgeb.evaluators import logRegClassificationEvaluator


def classification_scores(
    embeds_train: np.ndarray,
    y_train: Sequence[str],
    embeds_test: np.ndarray,
    y_test: Sequence[str],
    *,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Fit + evaluate the DGEB logistic-regression probe; return ``{accuracy, f1[, ap]}``.

    Delegates entirely to upstream ``logRegClassificationEvaluator``: macro-F1 (the
    benchmark's primary metric) + accuracy, plus ``ap`` only when the train set is binary
    (never for the 128-class EC set). The math is upstream's; this only coerces the result
    values to ``float``.
    """
    # The vendored DGEB evaluator is copied verbatim (untyped) — calling it from typed
    # core trips strict mypy's no-untyped-call; the verbatim copy is the metric authority.
    evaluator = logRegClassificationEvaluator(  # type: ignore[no-untyped-call]
        embeds_train,
        list(y_train),
        embeds_test,
        list(y_test),
        max_iter=max_iter,
        seed=seed,
    )
    raw = evaluator()
    return {k: float(v) for k, v in raw.items()}
