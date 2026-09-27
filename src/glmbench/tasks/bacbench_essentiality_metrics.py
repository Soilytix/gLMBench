"""BacBench gene-essentiality metric reduction — a documented sklearn RE-DERIVATION.

⚠️  RE-DERIVATION RISK. Unlike :mod:`glmbench.tasks.metrics`
(a thin wrapper that *calls* the byte-identical RNAGym ``performance_fitness.py``), there
is **no torch-free vendorable metric file** in BacBench: upstream's metric is entangled
with a PyTorch-Lightning training loop and ``torchmetrics`` (see the verbatim reference at
``_vendor/bacbench_essentiality/run_train_cls.py`` + its ``NOTICE``). So this module
*re-implements* the protocol with ``sklearn`` / ``numpy``:

- :func:`genome_ranking_metrics` — per-genome **AUROC** + **AUPRC** via
  ``sklearn.metrics.roc_auc_score`` / ``average_precision_score``. These are numerically
  equivalent (rank-based) to upstream's ``torchmetrics.functional.auroc /
  average_precision(task="binary")``; a dev-gated oracle test asserts agreement to 1e-6.
  Mirrors upstream ``_safe_ranking_metrics``: NaN for a single-class genome.
- :func:`macro` — mean **and** median across genomes (NaN-skipping), matching upstream's
  ``calculate_metrics_per_genome`` which prints both. The paper's reported tables use the
  **mean**; the task sets ``primary_metric`` to the macro-mean AUROC.
- :func:`confusion_counts` — summed (micro) TP/FP/TN/FN at a 0.5 probability threshold
  across all scored test genes, plus the derived precision/recall/specificity/F1/accuracy.
  (BacBench's F1 uses ``torchmetrics`` binary default threshold 0.5, i.e. logit>0 ⇔
  prob>0.5 — the same operating point.)

This module is core (torch-free): numpy + scikit-learn only (already core deps).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# The fixed operating point for the confusion matrix (matches torchmetrics binary F1).
DEFAULT_THRESHOLD = 0.5

_NAN = float("nan")


def genome_ranking_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Per-genome ranking metrics — ``{"auroc", "auprc"}`` via sklearn.

    Returns NaN for both when the genome lacks ≥2 classes (undefined AUROC/AUPRC) —
    mirroring upstream ``_has_both_binary_classes`` so such genomes drop out of the macro.
    ``scores`` may be probabilities or logits (both rank-monotone → identical AUROC/AUPRC).
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return {"auroc": _NAN, "auprc": _NAN}
    if np.isnan(s).any():
        raise ValueError("genome_ranking_metrics: NaN score — refusing to report a poisoned metric.")
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def macro(per_genome: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Macro AUROC/AUPRC across genomes — mean **and** median, NaN-skipping (upstream).

    Args:
        per_genome: one dict per scored genome, each with ``auroc`` / ``auprc`` (as from
            :func:`genome_ranking_metrics`; single-class genomes carry NaN and are skipped).

    Returns:
        ``{"macro_mean_auroc", "macro_median_auroc", "macro_mean_auprc",
        "macro_median_auprc"}``. All NaN if no genome had both classes.
    """
    auroc = np.asarray([g["auroc"] for g in per_genome], dtype=float)
    auprc = np.asarray([g["auprc"] for g in per_genome], dtype=float)

    def _mean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else _NAN

    def _median(a: np.ndarray) -> float:
        return float(np.nanmedian(a)) if np.isfinite(a).any() else _NAN

    return {
        "macro_mean_auroc": _mean(auroc),
        "macro_median_auroc": _median(auroc),
        "macro_mean_auprc": _mean(auprc),
        "macro_median_auprc": _median(auprc),
    }


def confusion_counts(
    labels: Sequence[int] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Summed (micro) confusion matrix + derived rates at a probability ``threshold``.

    Counts TP/FP/TN/FN over **all** input genes (the test set pooled across genomes), a
    prediction being positive iff ``prob >= threshold``. Returns flat floats so they drop
    straight into a ``TaskResult.metrics`` dict; derived rates use safe (0-guarded)
    division. ``threshold`` defaults to 0.5 (the torchmetrics binary-F1 operating point).
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    if y.size != p.size:
        raise ValueError(f"confusion_counts: {y.size} labels vs {p.size} probs.")
    if y.size and np.isnan(p).any():
        raise ValueError("confusion_counts: NaN probability — refusing to report a poisoned metric.")
    pred = p >= threshold
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    fn = int(np.sum(~pred & pos))

    def _safe(num: int, den: int) -> float:
        return float(num) / float(den) if den else _NAN

    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)  # sensitivity / TPR
    specificity = _safe(tn, tn + fp)  # TNR
    f1 = _safe(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe(tp + tn, tp + tn + fp + fn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": float(threshold),
    }
