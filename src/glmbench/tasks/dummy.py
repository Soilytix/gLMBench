"""Tiny model-free tasks that exercise the lifecycle + leaderboard end to end.

``DummyScoreTask`` needs ``SEQUENCE_LOGLIKELIHOOD`` and reports a deterministic metric
from a handful of toy sequences; ``DummyEmbedTask`` needs ``EMBEDDING`` so that, run
against a score-only adapter, it produces an ``N/A`` cell (capability gating end to end).
Both are model-free (they ride the EchoAdapter / any adapter) and torch-free.
"""

from __future__ import annotations

import numpy as np

from glmbench import registry
from glmbench.adapters.base import Capability, ModelAdapter

from .base import ResultStatus, Task, TaskResult

# Fixed toy sequences — deterministic inputs make deterministic metrics.
_TOY_SEQS = ["ACGTACGT", "TTTTAAAA", "GGGGCCCC", "ACACACAC", "TGCATGCA"]


@registry.register("task", "dummy")
class DummyScoreTask(Task):
    """Mean sequence log-likelihood over five toy sequences (needs SEQUENCE_LOGLIKELIHOOD)."""

    name = "dummy"
    version = "0.1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def __init__(self) -> None:
        self._seqs: list[str] = []
        self._scores: list[float] = []

    def prepare(self) -> None:
        self._seqs = list(_TOY_SEQS)

    def run(self, adapter: ModelAdapter) -> None:
        # batch-first: one call for all sequences
        self._scores = adapter.score_sequences(self._seqs, reduction="mean")

    def score(self) -> TaskResult:
        scores = np.asarray(self._scores, dtype=float)
        if scores.size == 0:
            raise ValueError("DummyScoreTask scored 0 sequences.")
        if np.isnan(scores).any():
            raise ValueError("DummyScoreTask got a NaN score — refusing to report it.")
        metrics = {
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
        }
        return TaskResult(
            self.name,
            ResultStatus.OK,
            "mean_score",
            metrics,
            {"n_scored": int(scores.size), "n_missing": 0, "task_version": self.version},
        )


@registry.register("task", "dummy-embed")
class DummyEmbedTask(Task):
    """Mean embedding-norm over two toy sequences (needs EMBEDDING).

    Exists so a score-only adapter yields an ``N/A`` cell.
    """

    name = "dummy-embed"
    version = "0.1.0"
    required_capabilities = frozenset({Capability.EMBEDDING})

    def __init__(self) -> None:
        self._seqs: list[str] = []
        self._norm: float = float("nan")

    def prepare(self) -> None:
        self._seqs = list(_TOY_SEQS[:2])

    def run(self, adapter: ModelAdapter) -> None:
        result = adapter.embed(self._seqs, layers="last", pool="mean")
        # one [len(layers), embedding_dim] array per sequence; reduce to a scalar norm
        norms = [float(np.linalg.norm(np.asarray(a))) for a in result.arrays]
        self._norm = float(np.mean(norms)) if norms else float("nan")

    def score(self) -> TaskResult:
        if np.isnan(self._norm):
            raise ValueError("DummyEmbedTask produced no embeddings.")
        return TaskResult(
            self.name,
            ResultStatus.OK,
            "mean_embed_norm",
            {"mean_embed_norm": self._norm},
            {"n_scored": len(self._seqs), "n_missing": 0, "task_version": self.version},
        )
