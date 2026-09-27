"""RNAGym prokaryotic protein-coding DMS — zero-shot variant-effect scoring.

Score the **11 ``mRNA-coding ∩ PROK``** RNAGym assays as each variant's causal sequence
log-likelihood and correlate against experimental fitness. The task is
**model-agnostic**: it orchestrates + reduces only, and the scoring itself happens
inside each adapter's runner (the core never imports a model).

Lifecycle:

- :meth:`prepare` loads the 11 assays via the data loader (sha256-pinned, WT
  dropped, ``U→T`` normalized — every model sees identical DNA inputs).
- :meth:`run` collects **all** variant sequences across **all** assays and scores them
  in **one** batched ``adapter.score_sequences`` call (batch-first), then scatters the
  scores back per assay.
- :meth:`score` reduces to per-assay abs-Spearman / symmetric-AUC / abs-MCC **and the
  macro** through :mod:`glmbench.tasks.metrics` — i.e. via the vendored upstream
  ``performance_fitness.py``, never a hand-rolled reduction. The headline
  ``primary_metric`` is the upstream **type-mean** ``macro_spearman``; for the
  all-``mRNA-coding`` prok set this equals a plain mean over the 11 assays, but it is
  obtained through the upstream type-mean path so it generalizes. BLAT is broken out as
  its own line.

The scoring convention (prepend BOS, trim BOS, mean per-token log-prob, substitutions
only) is implemented in each adapter's runner, not here. The task only requests scores
and reduces them.

This module is core (torch-free): pure stdlib + the dependency-light core deps.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from glmbench import registry
from glmbench.adapters.base import Capability, ModelAdapter

from . import metrics
from .base import ResultStatus, Task, TaskResult
from .rnagym_data import DEFAULT_MANIFEST as _DEFAULT_MANIFEST
from .rnagym_data import Assay, assay_reference, load_assays, parse_mutant

logger = logging.getLogger(__name__)

# Vendored-data manifest (the 11 prok assays + sha256 + rna_type).
DEFAULT_MANIFEST = _DEFAULT_MANIFEST

# TEM-1 beta-lactamase, a widely used DMS benchmark — broken out as its own report line.
BLAT_PREFIX = "BLAT_ECOLX"


@registry.register("task", "rnagym-dms")
class RnagymDmsTask(Task):
    """Score the RNAGym prokaryotic protein-coding DMS assays (zero-shot)."""

    name = "rnagym-dms"
    version = "1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})
    higher_is_better = True  # macro abs-Spearman
    # Variant-effect fallback: an MLM encoder with no causal likelihood
    # (gLM2 and any future masked-LM) scores the assays via masked-marginal LLR
    # instead. AR models (LOAM, Evo, Evo2, GenomeOcean) satisfy the primary and never
    # take this path. The metric is method-agnostic abs-Spearman.
    capability_fallbacks = (frozenset({Capability.MASKED_MARGINAL_LLR}),)

    def __init__(
        self,
        manifest_path: Path | str = DEFAULT_MANIFEST,
        *,
        data_dir: Path | str | None = None,
        reduction: str = "mean",
        verify_hashes: bool = True,
        max_context: int | None = None,
    ) -> None:
        """Construct the task.

        Args:
            manifest_path: the sha256-pinned manifest declaring the assays to load.
            data_dir: where the ``processed_DMS_files/`` live (defaults to the
                manifest's own directory).
            reduction: per-sequence log-likelihood reduction passed to the adapter
                (``mean``, the per-token average, is what the reported numbers use).
            verify_hashes: enforce the manifest integrity gate in ``prepare``.
            max_context: optional per-sequence length cap in **nucleotides**. Variants
                longer than this are skipped-and-counted, never silently truncated. If
                ``None``, the adapter's own ``max_context`` attribute is consulted; if that
                is also absent, no cap is applied.
        """
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir) if data_dir is not None else self.manifest_path.parent
        self.reduction = reduction
        self.verify_hashes = verify_hashes
        self.max_context = max_context

        self.assays: list[Assay] = []
        # Per assay: the within-assay indices of variants actually scored (the rest
        # were over-context or unscoreable and skipped). Aligned with ``self._model_scores``.
        self._kept_idx: list[list[int]] = []
        self._model_scores: list[list[float]] = []
        self._n_truncated = 0  # skipped: reference longer than max_context (nt)
        self._n_unscoreable = 0  # skipped: indel / non-substitution (LLR path only)
        # The scoring path negotiated by ``evaluate`` (set on the instance before ``run``);
        # ``"sequence_loglikelihood"`` (AR) or ``"masked_marginal_llr"`` (MLM fallback).
        self._scoring_method = "sequence_loglikelihood"

    # --- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        self.assays = load_assays(
            self.manifest_path, data_dir=self.data_dir, verify_hashes=self.verify_hashes
        )

    def run(self, adapter: ModelAdapter) -> None:
        # Which negotiated path? The primary (AR causal log-likelihood) wins whenever the
        # adapter offers it; an MLM-only adapter falls back to masked-marginal LLR.
        chosen = self._chosen_capabilities or self.required_capabilities
        use_llr = (
            Capability.MASKED_MARGINAL_LLR in chosen
            and Capability.SEQUENCE_LOGLIKELIHOOD not in chosen
        )
        self._scoring_method = "masked_marginal_llr" if use_llr else "sequence_loglikelihood"

        # The length cap (nucleotides) — explicit task knob wins, else the adapter's
        # own declaration; ``None`` means "no cap".
        max_context = self.max_context
        if max_context is None:
            max_context = getattr(adapter, "max_context", None)

        if use_llr:
            self._run_llr(adapter, max_context)
        else:
            self._run_loglik(adapter, max_context)

    def _run_loglik(self, adapter: ModelAdapter, max_context: int | None) -> None:
        """Primary AR path — unchanged: one batched ``score_sequences`` over all variants."""
        # Partition every variant into kept (in-context) vs skipped (over-context),
        # building ONE flat sequence list for ONE batched adapter call.
        flat_seqs: list[str] = []
        self._kept_idx = []
        self._n_truncated = 0
        self._n_unscoreable = 0
        for assay in self.assays:
            kept: list[int] = []
            for wi, seq in enumerate(assay.sequences):
                if max_context is not None and len(seq) > max_context:
                    self._n_truncated += 1
                    continue
                kept.append(wi)
                flat_seqs.append(seq)
            self._kept_idx.append(kept)

        if self._n_truncated:
            # Loud, never silent.
            logger.warning(
                "rnagym-dms: %d variant(s) exceed max_context=%s nt and were skipped "
                "(counted in n_missing_predictions, never silently truncated).",
                self._n_truncated,
                max_context,
            )

        # One batched call for every in-context variant across all assays.
        flat_scores = adapter.score_sequences(flat_seqs, reduction=self.reduction)
        if len(flat_scores) != len(flat_seqs):
            raise ValueError(
                f"adapter returned {len(flat_scores)} scores for {len(flat_seqs)} "
                "sequences — batched score_sequences must be 1:1 with its inputs."
            )

        # Scatter the flat scores back into per-assay lists (kept order is preserved).
        self._scatter(flat_scores)

    def _run_llr(self, adapter: ModelAdapter, max_context: int | None) -> None:
        """MLM fallback path — masked-marginal LLR over all variant items, one batched call.

        Per assay: parse each mutant string, derive-and-verify the reference (WT) from the
        scoreable variants, then keep the substitution-only, in-context variants
        as ``{"reference", "mutations"}`` items. Indels / non-substitutions and
        over-``max_context`` variants are skipped-and-counted (never silently scored). One
        flat batched ``score_variant_llr`` call across all assays.
        """
        flat_items: list[dict[str, Any]] = []
        self._kept_idx = []
        self._n_truncated = 0
        self._n_unscoreable = 0
        for assay in self.assays:
            parsed = [parse_mutant(m) for m in assay.mutants]
            # Derive + verify the single reference (WT) for this assay (loud on divergence).
            reference = assay_reference(assay, parsed)
            kept: list[int] = []
            for wi, muts in enumerate(parsed):
                if muts is None:
                    self._n_unscoreable += 1
                    continue
                if max_context is not None and len(reference) > max_context:
                    self._n_truncated += 1
                    continue
                kept.append(wi)
                flat_items.append(
                    {"reference": reference, "mutations": [list(m) for m in muts]}
                )
            self._kept_idx.append(kept)

        if self._n_unscoreable:
            logger.warning(
                "rnagym-dms: %d variant(s) are not clean single-nt substitutions "
                "(indel/non-substitution) and were skipped under the masked-marginal LLR "
                "path (counted in n_missing_predictions, never silently scored).",
                self._n_unscoreable,
            )
        if self._n_truncated:
            logger.warning(
                "rnagym-dms: %d variant(s) exceed max_context=%s nt and were skipped "
                "(counted in n_missing_predictions, never silently truncated).",
                self._n_truncated,
                max_context,
            )

        flat_scores = adapter.score_variant_llr(flat_items)
        if len(flat_scores) != len(flat_items):
            raise ValueError(
                f"adapter returned {len(flat_scores)} LLR scores for {len(flat_items)} "
                "variant items — batched score_variant_llr must be 1:1 with its inputs."
            )
        self._scatter(flat_scores)

    def _scatter(self, flat_scores: list[float]) -> None:
        """Scatter a flat per-variant score list back into per-assay lists (kept order)."""
        self._model_scores = []
        cursor = 0
        for kept in self._kept_idx:
            self._model_scores.append(flat_scores[cursor : cursor + len(kept)])
            cursor += len(kept)

    def score(self) -> TaskResult:
        if not self.assays:
            raise RuntimeError("RnagymDmsTask.prepare must be called before score")
        if len(self._model_scores) != len(self.assays):
            raise RuntimeError("RnagymDmsTask.run must be called before score")

        per_assay: list[dict[str, Any]] = []
        for assay, kept, model_scores in zip(
            self.assays, self._kept_idx, self._model_scores, strict=True
        ):
            dms = np.asarray([assay.dms_scores[wi] for wi in kept], dtype=float)
            model = np.asarray(model_scores, dtype=float)
            n_missing = len(assay) - len(kept)

            if model.size == 0:
                # Whole assay was over-context: NaN metrics, excluded from the macro
                # (upstream's type-mean uses pandas skipna), full assay counted missing.
                m = {"spearman": float("nan"), "auc": float("nan"), "mcc": float("nan")}
            else:
                if np.isnan(model).any():
                    # Guard before a NaN can poison a Spearman.
                    raise ValueError(
                        f"assay {assay.dms_id}: adapter returned a NaN score — "
                        "refusing to report a poisoned metric."
                    )
                m = metrics.assay_metrics(dms, model)

            per_assay.append(
                {
                    "dms_id": assay.dms_id,
                    "rna_type": assay.rna_type,
                    "n_variants": len(kept),
                    "n_missing": n_missing,
                    **m,
                }
            )

        # Macro via the vendored upstream type-mean reducer — never re-derived.
        macro = metrics.macro(per_assay)

        n_missing_total = self._n_truncated + self._n_unscoreable
        metrics_out: dict[str, float] = {
            "macro_spearman": macro["macro_spearman"],
            "macro_auc": macro["macro_auc"],
            "macro_mcc": macro["macro_mcc"],
            "n_assays": float(len(self.assays)),
            "n_missing_predictions": float(n_missing_total),
        }
        # The 11 per-assay headline keys (one abs-Spearman per assay).
        for p in per_assay:
            metrics_out[f"{p['dms_id']}_spearman"] = float(p["spearman"])

        # BLAT broken out as its own line — first assay matching the prefix.
        blat = next((p for p in per_assay if p["dms_id"].startswith(BLAT_PREFIX)), None)
        if blat is not None:
            metrics_out["blat_ecolx_spearman"] = float(blat["spearman"])

        n_scored = sum(p["n_variants"] for p in per_assay)
        metadata: dict[str, Any] = {
            "n_scored": n_scored,
            "n_missing": n_missing_total,
            "n_over_context": self._n_truncated,
            "n_unscoreable": self._n_unscoreable,
            "reduction": self.reduction,
            "scoring_method": self._scoring_method,
            "task_version": self.version,
            "max_context": self.max_context,
            "per_assay": per_assay,
        }
        if blat is not None:
            metadata["blat_dms_id"] = blat["dms_id"]

        # An entirely unscored benchmark (every variant over-context) is an ERROR, not a
        # silent all-NaN row — surfaced via the macro being NaN.
        if n_scored == 0 or math.isnan(metrics_out["macro_spearman"]):
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {
                    "error": "rnagym-dms scored 0 in-context variants (all over max_context)",
                    **metadata,
                },
            )

        return TaskResult(
            self.name,
            ResultStatus.OK,
            "macro_spearman",
            metrics_out,
            metadata,
        )
