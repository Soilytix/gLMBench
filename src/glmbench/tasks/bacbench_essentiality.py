"""BacBench gene-essentiality prediction — the first EMBEDDING-consuming task.

Probe the frozen gene embeddings of a genomic LM for gene essentiality (a binary,
gene-level classification at the *gene scale* of BacBench). The task is **model-agnostic**:
it requests one pooled embedding per gene from the adapter (the heavy forward pass runs in
the adapter's own runner), then fits + evaluates a shallow **CPU sklearn** probe (the probe
is task logic, never behind the adapter).

Lifecycle:

- :meth:`prepare` loads the train+test genomes via the sha256-pinned loader
  (:mod:`bacbench_essentiality_data`); each gene's sequence is the BacBench DNA-LM input
  (CDS + 128 bp upstream promoter, strand-aware, U→T-normalized).
- :meth:`run` windows every gene to the adapter's context (window-chunk + mean-across-chunks,
  the BacBench recipe — never a silent truncation), flattens **all** chunks across
  **all** genes into ONE batched ``adapter.embed(..., pool="mean")`` call (batch-first),
  and means the chunk vectors back into one frozen embedding per gene.
- :meth:`score` standardizes + fits an ``sklearn`` ``LogisticRegression`` on the **train**
  genomes and evaluates **per test genome**: AUROC / AUPRC (sklearn), macro-aggregated
  across genomes (mean **and** median), plus a summed confusion matrix at a 0.5
  probability threshold. ``primary_metric`` is the macro-**mean** AUROC (the paper's
  reported table; the upstream code also logs the median — both are exposed).

**Re-derivation:** BacBench's upstream probe is a torch+Lightning net and its metric is
``torchmetrics`` — neither is importable by the torch-free core. This task re-implements the
protocol with sklearn (probe) + sklearn (AUROC/AUPRC); the byte-identical upstream reference
is vendored for provenance at ``_vendor/bacbench_essentiality/run_train_cls.py`` (see its
``NOTICE``). Reproduction is a SANITY BAND, not an exact match.

This module is core (torch-free): stdlib + numpy + scikit-learn (core deps).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glmbench import registry
from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter

from . import bacbench_essentiality_metrics as bb_metrics
from .bacbench_essentiality_data import DEFAULT_MANIFEST as _DEFAULT_MANIFEST
from .bacbench_essentiality_data import Genome, load_genomes
from .base import ResultStatus, Task, TaskResult
from ._depth import choose_best_layer, measure_depth_health, sweep_diagnostics

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = _DEFAULT_MANIFEST

# Optional env override of the manifest path — handy for a timing/smoke subset run
# (e.g. ``scripts/make_essentiality_subset.py``) without a custom benchmark YAML. The
# benchmark CLI builds the task with default args, so this is the only seam to point it at
# a subset manifest from the command line.
_MANIFEST_ENV = "GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST"

# Window-chunk parameters recycled from BacBench ``embed_dna.py`` (overlap 32, min 32).
DEFAULT_WINDOW_OVERLAP = 32
DEFAULT_MIN_SEQ_LEN = 32

# The splits the probe needs: fit on ``train``, evaluate on ``test``. ``validation`` (used
# upstream only for LR tuning + early stopping, which the fixed sklearn probe does not need)
# is intentionally not embedded — saving ~20% of the forward passes.
_TRAIN_SPLIT = "train"
_TEST_SPLIT = "test"
_SPLITS = (_TRAIN_SPLIT, _TEST_SPLIT)


def chunk_dna_sequence(dna_seq: str, max_seq_len: int, overlap: int) -> list[str]:
    """Overlapping windows of length ``max_seq_len`` (stride ``max_seq_len - overlap``).

    Byte-for-byte the BacBench ``embed_dna.chunk_dna_sequence`` recipe: keep emitting
    windows until the end is reached. A sequence ``<= max_seq_len`` yields a single chunk.
    """
    chunks: list[str] = []
    n = len(dna_seq)
    start = 0
    while start < n:
        end = min(start + max_seq_len, n)
        chunks.append(dna_seq[start:end])
        if end == n:
            break
        start += max_seq_len - overlap
    return chunks


# --- the shared probe: fit on train genomes, evaluate per test genome -----------
#
# Extracted so both essentiality tasks (the last-layer embedding and the per-layer sweep)
# fit + evaluate with the SAME proven sklearn protocol — one source of truth, no metric
# divergence. Pure CPU; the feature *source* differs per task, the probe does not.


class _ProbeError(RuntimeError):
    """The probe could not be fit/evaluated (e.g. a single-class train split)."""


@dataclass
class _ProbeOutcome:
    """The probe's verdict for one feature matrix (one model signal / one layer)."""

    metrics: dict[str, float]  # macro + confusion + accounting (the metrics_out keys)
    per_genome: list[dict[str, Any]]
    n_train_genomes: int
    n_train_genes: int
    n_test_genomes: int
    n_test_genomes_scored: int
    n_excluded_single_class: int


def _fit_and_evaluate_probe(
    genomes: list[Genome],
    embeddings: list[np.ndarray],
    *,
    threshold: float,
    seed: int,
    probe_C: float,
    probe_max_iter: int,
    task_name: str = "bacbench-essentiality",
) -> _ProbeOutcome:
    """Standardize + fit LogisticRegression on the train genomes, evaluate per test genome.

    ``embeddings`` is one ``[n_genes, n_features]`` matrix per genome, aligned with
    ``genomes``. Raises :class:`_ProbeError` when the probe is undefined (no train/test
    genomes, or a single-class train split). Otherwise returns the macro AUROC/AUPRC,
    summed confusion matrix at ``threshold``, the per-genome breakdown, and the accounting.
    """
    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    test_genomes: list[tuple[Genome, np.ndarray]] = []
    n_train_genomes = 0
    for genome, emb in zip(genomes, embeddings, strict=True):
        y = np.asarray(genome.labels, dtype=int)
        if genome.split == _TRAIN_SPLIT:
            train_X.append(emb)
            train_y.append(y)
            n_train_genomes += 1
        elif genome.split == _TEST_SPLIT:
            test_genomes.append((genome, emb))

    if not train_X:
        raise _ProbeError(f"{task_name}: no train genomes to fit the probe.")
    if not test_genomes:
        raise _ProbeError(f"{task_name}: no test genomes to evaluate.")

    Xtr = np.concatenate(train_X, axis=0)
    ytr = np.concatenate(train_y, axis=0)
    if int(np.unique(ytr).size) < 2:
        raise _ProbeError(
            f"{task_name}: train split has a single class — cannot fit a binary probe."
        )

    # CPU sklearn probe (a re-derivation of upstream's torch probe): standardize then
    # LogisticRegression.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(
        C=probe_C,
        max_iter=probe_max_iter,
        class_weight=None,  # upstream uses unweighted BCE (matches the 0.5-threshold F1)
        random_state=seed,
    )
    clf.fit(scaler.transform(Xtr), ytr)

    # Evaluate per test genome; pool probs+labels for the summed confusion matrix.
    per_genome: list[dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n_excluded_single_class = 0
    for genome, emb in test_genomes:
        y = np.asarray(genome.labels, dtype=int)
        probs = clf.predict_proba(scaler.transform(emb))[:, 1]
        all_probs.append(probs)
        all_labels.append(y)
        rk = bb_metrics.genome_ranking_metrics(y, probs)
        scored = genome.has_both_classes
        if not scored:
            n_excluded_single_class += 1
        per_genome.append(
            {
                "genome_name": genome.genome_name,
                "genus": genome.genus,
                "n_genes": int(y.size),
                "n_essential": int(y.sum()),
                "scored": scored,
                "auroc": rk["auroc"],
                "auprc": rk["auprc"],
            }
        )

    macro = bb_metrics.macro(per_genome)
    confusion = bb_metrics.confusion_counts(
        np.concatenate(all_labels), np.concatenate(all_probs), threshold=threshold
    )
    n_scored = sum(1 for p in per_genome if p["scored"])
    n_test_genes = int(sum(p["n_genes"] for p in per_genome))

    metrics: dict[str, float] = {
        "macro_mean_auroc": macro["macro_mean_auroc"],
        "macro_median_auroc": macro["macro_median_auroc"],
        "macro_mean_auprc": macro["macro_mean_auprc"],
        "macro_median_auprc": macro["macro_median_auprc"],
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "tn": confusion["tn"],
        "fn": confusion["fn"],
        "precision": confusion["precision"],
        "recall": confusion["recall"],
        "specificity": confusion["specificity"],
        "f1": confusion["f1"],
        "accuracy": confusion["accuracy"],
        "n_test_genomes_scored": float(n_scored),
        "n_test_genes": float(n_test_genes),
    }
    return _ProbeOutcome(
        metrics=metrics,
        per_genome=per_genome,
        n_train_genomes=n_train_genomes,
        n_train_genes=int(ytr.size),
        n_test_genomes=len(test_genomes),
        n_test_genomes_scored=n_scored,
        n_excluded_single_class=n_excluded_single_class,
    )


@registry.register("task", "bacbench-essentiality")
class BacBenchEssentialityTask(Task):
    """Probe frozen gene embeddings for gene essentiality (AUROC/AUPRC + confusion)."""

    name = "bacbench-essentiality"
    # Shares one forward pass with the other tasks over this corpus (adapters/reuse.py).
    fusion_group = "bacbench-essentiality-windows"
    version = "1.0"
    required_capabilities = frozenset({Capability.EMBEDDING})
    higher_is_better = True  # macro-mean AUROC
    # Human-readable provenance for which model signal the probe features came from
    # (echoed into the result metadata). Overridden by the layer sweep.
    _feature_source = "embedding (hidden-state, layers='last', pool='mean')"

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        data_dir: Path | str | None = None,
        verify_hashes: bool = True,
        max_context: int | None = None,
        threshold: float = bb_metrics.DEFAULT_THRESHOLD,
        seed: int = 1,
        probe_C: float = 1.0,
        probe_max_iter: int = 1000,
        window_overlap: int = DEFAULT_WINDOW_OVERLAP,
        min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
    ) -> None:
        """Construct the task.

        Args:
            manifest_path: the sha256-pinned manifest declaring the per-split gene files.
                ``None`` → the ``GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST`` env var if set
                (a subset manifest for a timing/smoke run), else the vendored default.
            data_dir: where the processed split CSVs live (defaults to the manifest dir).
            verify_hashes: enforce the manifest integrity gate in ``prepare``.
            max_context: per-gene length cap in **nucleotides**. Longer genes are
                window-chunked (never truncated). ``None`` → consult the adapter's own
                ``max_context``; if that is also absent, each gene is embedded whole.
            threshold: probability cut for the confusion matrix (default 0.5).
            seed: probe random_state (determinism).
            probe_C / probe_max_iter: sklearn ``LogisticRegression`` knobs.
            window_overlap / min_seq_len: window-chunk parameters (BacBench recipe).
        """
        if manifest_path is None:
            manifest_path = os.environ.get(_MANIFEST_ENV) or DEFAULT_MANIFEST
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir) if data_dir is not None else self.manifest_path.parent
        self.verify_hashes = verify_hashes
        self.max_context = max_context
        self.threshold = threshold
        self.seed = seed
        self.probe_C = probe_C
        self.probe_max_iter = probe_max_iter
        self.window_overlap = window_overlap
        self.min_seq_len = min_seq_len

        self.genomes: list[Genome] = []
        # Per genome, in load order: the [n_genes, H] embedding matrix (filled by run()).
        self._embeddings: list[np.ndarray] = []
        self._n_windows = 0
        self._embedding_dim = 0
        self._embed_seconds = 0.0

    # --- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        self.genomes = load_genomes(
            self.manifest_path,
            data_dir=self.data_dir,
            verify_hashes=self.verify_hashes,
            splits=_SPLITS,
        )

    def _windows(self, seq: str, max_context: int | None) -> list[str]:
        """Window one gene to the context; ``None`` → the whole gene as one chunk."""
        if max_context is None:
            return [seq]
        chunks = [c for c in chunk_dna_sequence(seq, max_context, self.window_overlap)
                  if len(c) >= self.min_seq_len]
        # Never drop a labeled gene: if every window fell under min_seq_len, keep the gene
        # whole (a single short chunk) rather than silently losing its label.
        return chunks if chunks else [seq]

    def _build_flat_windows(
        self, adapter: ModelAdapter
    ) -> tuple[list[str], list[tuple[int, int]], list[int]]:
        """Flatten every gene of every genome into ONE window list (batch-first).

        Returns ``(flat_chunks, per_gene_span, gene_counts)`` where ``per_gene_span[g]`` is
        the half-open ``(start, stop)`` slice of ``flat_chunks`` belonging to gene ``g`` (in
        flattened gene order), and ``gene_counts[i]`` is the gene count of genome ``i`` (load
        order) so results can be regrouped per genome. Used by the layer sweep; the
        last-layer :meth:`run` builds the same list inline.
        """
        max_context = self.max_context
        if max_context is None:
            max_context = getattr(adapter, "max_context", None)
        flat_chunks: list[str] = []
        per_gene_span: list[tuple[int, int]] = []
        gene_counts: list[int] = []
        for genome in self.genomes:
            gene_counts.append(len(genome))
            for seq in genome.sequences:
                start = len(flat_chunks)
                flat_chunks.extend(self._windows(seq, max_context))
                per_gene_span.append((start, len(flat_chunks)))
        return flat_chunks, per_gene_span, gene_counts

    def run(self, adapter: ModelAdapter) -> None:
        max_context = self.max_context
        if max_context is None:
            max_context = getattr(adapter, "max_context", None)

        # Build ONE flat chunk list across ALL genes of ALL genomes; remember, per gene,
        # the (start, stop) slice of chunk rows so we can mean them back afterwards.
        flat_chunks: list[str] = []
        per_gene_span: list[tuple[int, int]] = []  # aligned with the flattened gene order
        gene_counts: list[int] = []  # n_genes per genome (load order)
        for genome in self.genomes:
            gene_counts.append(len(genome))
            for seq in genome.sequences:
                start = len(flat_chunks)
                flat_chunks.extend(self._windows(seq, max_context))
                per_gene_span.append((start, len(flat_chunks)))

        self._n_windows = len(flat_chunks)
        if not flat_chunks:
            raise RuntimeError("bacbench-essentiality: no gene windows to embed (empty data).")

        # ONE batched embedding call for every window across every gene.
        # Wall-time it so a small subset run extrapolates to the full-corpus GPU cost.
        _t0 = time.perf_counter()
        result: EmbeddingResult = adapter.embed(flat_chunks, layers="last", pool="mean")
        self._embed_seconds = time.perf_counter() - _t0
        logger.info(
            "bacbench-essentiality: embedded %d windows in %.1fs (%.4fs/window).",
            self._n_windows,
            self._embed_seconds,
            self._embed_seconds / max(1, self._n_windows),
        )
        if len(result.arrays) != len(flat_chunks):
            raise ValueError(
                f"adapter returned {len(result.arrays)} embeddings for {len(flat_chunks)} "
                "windows — batched embed must be 1:1 with its inputs."
            )

        # Each array is [len(layers), H]; layers='last' ⇒ one layer ⇒ take row 0.
        win_vecs = np.stack([np.asarray(a, dtype=np.float64)[0] for a in result.arrays], axis=0)
        if np.isnan(win_vecs).any():
            raise ValueError("bacbench-essentiality: adapter returned a NaN embedding.")
        self._embedding_dim = int(win_vecs.shape[1])

        # Mean the window vectors back into one embedding per gene, regrouped per genome.
        gene_embs = np.stack(
            [win_vecs[a:b].mean(axis=0) for (a, b) in per_gene_span], axis=0
        )
        self._embeddings = []
        cursor = 0
        for n in gene_counts:
            self._embeddings.append(gene_embs[cursor : cursor + n])
            cursor += n

    def _probe_metadata(self, outcome: _ProbeOutcome) -> dict[str, Any]:
        """The task-level metadata around one probe outcome (timings, dims, provenance)."""
        return {
            "n_train_genomes": outcome.n_train_genomes,
            "n_train_genes": outcome.n_train_genes,
            "n_test_genomes": outcome.n_test_genomes,
            "n_test_genomes_scored": outcome.n_test_genomes_scored,
            "n_excluded_single_class": outcome.n_excluded_single_class,
            "n_windows_embedded": self._n_windows,
            "embedding_dim": self._embedding_dim,
            "embed_seconds": round(self._embed_seconds, 3),
            "embed_seconds_per_window": round(
                self._embed_seconds / max(1, self._n_windows), 6
            ),
            "threshold": self.threshold,
            "probe": "sklearn.LogisticRegression(StandardScaler)+re-derivation",
            "probe_C": self.probe_C,
            "probe_seed": self.seed,
            "aggregation": "macro mean (primary) + median across test genomes",
            "max_context": self.max_context,
            "feature_source": self._feature_source,
            "task_version": self.version,
            "per_genome": outcome.per_genome,
        }

    def score(self) -> TaskResult:
        if not self.genomes:
            raise RuntimeError("BacBenchEssentialityTask.prepare must be called before score")
        if len(self._embeddings) != len(self.genomes):
            raise RuntimeError("BacBenchEssentialityTask.run must be called before score")

        try:
            outcome = _fit_and_evaluate_probe(
                self.genomes,
                self._embeddings,
                threshold=self.threshold,
                seed=self.seed,
                probe_C=self.probe_C,
                probe_max_iter=self.probe_max_iter,
                task_name=self.name,
            )
        except _ProbeError as e:
            return TaskResult(
                self.name, ResultStatus.ERROR, None, {}, {"error": str(e), "task_version": self.version}
            )

        metadata = self._probe_metadata(outcome)
        # No test genome had both classes → AUROC/AUPRC are all-NaN → ERROR, not a poisoned OK.
        if outcome.n_test_genomes_scored == 0 or math.isnan(outcome.metrics["macro_mean_auroc"]):
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {
                    "error": f"{self.name}: 0 test genomes had both classes "
                    "(macro AUROC undefined).",
                    **metadata,
                },
            )
        return TaskResult(
            self.name, ResultStatus.OK, "macro_mean_auroc", outcome.metrics, metadata
        )


# Env-var seam to pick which layers the sweep covers (the CLI builds tasks with no args, so
# this mirrors the manifest env override). Unset ⇒ sweep EVERY layer the model returns.
_SWEEP_LAYERS_ENV = "GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS"
_SWEEP_POOL_ENV = "GLMBENCH_BACBENCH_LAYER_SWEEP_POOL"


def parse_layer_list(value: str | None) -> list[int] | None:
    """Parse a layer selection string → ``list[int]`` or ``None`` (= every layer).

    Accepts a comma list (``"6,7,8,9,10,11,12"``), inclusive ranges (``"6-12"``), a mix
    (``"0,4-6,12"``), negatives (``"-1,-2"``), or ``"all"``/``""`` → ``None``. The layer
    numbers are **not** hardcoded anywhere — a model with a different depth just gets a
    different selection (or ``None`` to sweep all of its layers).
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("", "all", "*"):
        return None
    out: list[int] = []
    for tok in (t.strip() for t in v.split(",")):
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):  # an inclusive range "lo-hi"
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out or None


@registry.register("task", "bacbench-essentiality-layer-sweep")
class BacBenchEssentialityLayerSweepTask(BacBenchEssentialityTask):
    """Find the best embedding **layer** for essentiality — all layers in ONE forward pass.

    Captures every requested layer's pooled embedding in a **single** batched
    ``adapter.embed(layers=…)`` call (the expensive GPU work happens once), then fits a
    **separate** sklearn probe per layer on CPU and reports each layer's macro AUROC plus
    the winner. Same data, windowing, probe and metrics as :class:`BacBenchEssentialityTask`;
    only the feature is swept over the layer axis.

    **Layer selection is not hardcoded.** By default it sweeps *every* layer the model returns
    (``layers='all'`` → ``0..n_layers``), so it works on any model regardless of depth. Pass
    ``sweep_layers=[…]`` or set ``GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS`` (e.g. ``"6-12"``) to
    restrict it. The reported layer ids are whatever the adapter echoes back, so they always
    match that model's numbering.

    Requires ``EMBEDDING``, and is meant for adapters that serve every layer; on an adapter
    that serves only its final layer, every non-final layer would error in its runner.
    """

    name = "bacbench-essentiality-layer-sweep"
    # Shares one forward pass with the other tasks over this corpus (adapters/reuse.py).
    fusion_group = "bacbench-essentiality-windows"
    fusion_wants_all_layers = True
    version = "1.0"
    required_capabilities = frozenset({Capability.EMBEDDING})

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        sweep_layers: list[int] | None = None,
        pool: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(manifest_path, **kwargs)
        if sweep_layers is None:
            sweep_layers = parse_layer_list(os.environ.get(_SWEEP_LAYERS_ENV))
        # None ⇒ ask the adapter for "all" layers; a list ⇒ exactly those.
        self.sweep_layers = sweep_layers
        self.pool = pool or os.environ.get(_SWEEP_POOL_ENV) or "mean"
        self._feature_source = (
            f"embedding hidden states (per-layer sweep, pool='{self.pool}')"
        )
        # layer id -> per-genome list of [n_genes, H] feature matrices (filled by run()).
        self._per_layer: dict[int, list[np.ndarray]] = {}
        self._layer_ids: list[int] = []
        self._depth_health = None  # set in run(), read in score()
        self._layer_selection: dict = {}

    def run(self, adapter: ModelAdapter) -> None:
        flat_chunks, per_gene_span, gene_counts = self._build_flat_windows(adapter)
        self._n_windows = len(flat_chunks)
        if not flat_chunks:
            raise RuntimeError(
                "bacbench-essentiality-layer-sweep: no gene windows to embed (empty data)."
            )

        layers_arg: list[int] | str = (
            list(self.sweep_layers) if self.sweep_layers is not None else "all"
        )
        # ONE batched, multi-layer embed call: [len(layers), H] per window. On the
        # any-layer adapters this is a SINGLE forward pass over all windows.
        _t0 = time.perf_counter()
        result: EmbeddingResult = adapter.embed(flat_chunks, layers=layers_arg, pool=self.pool)
        self._embed_seconds = time.perf_counter() - _t0
        self._layer_ids = [int(x) for x in result.layers]
        logger.info(
            "bacbench-essentiality-layer-sweep: embedded %d windows × %d layers in %.1fs.",
            self._n_windows,
            len(self._layer_ids),
            self._embed_seconds,
        )
        if len(result.arrays) != len(flat_chunks):
            raise ValueError(
                f"adapter returned {len(result.arrays)} embeddings for {len(flat_chunks)} "
                "windows — batched embed must be 1:1 with its inputs."
            )
        win = np.stack([np.asarray(a, dtype=np.float64) for a in result.arrays], axis=0)
        if win.ndim != 3 or win.shape[1] != len(self._layer_ids):
            raise ValueError(
                f"expected per-window arrays of shape [n_layers={len(self._layer_ids)}, H]; "
                f"got stacked shape {win.shape}."
            )
        if np.isnan(win).any():
            raise ValueError("bacbench-essentiality-layer-sweep: adapter returned a NaN embedding.")
        self._embedding_dim = int(win.shape[2])

        # Mean window vectors back into one [n_layers, H] embedding per gene, keeping the layer
        # axis; then regroup per genome, per layer, into the per-layer feature matrices.
        gene_stack = np.stack(
            [win[a:b].mean(axis=0) for (a, b) in per_gene_span], axis=0
        )  # [total_genes, n_layers, H]
        # Measure depth health HERE, where the [items, layers, H] array is in
        # hand. Below it is split per layer per genome and the layer axis is gone.
        self._depth_health = (
            measure_depth_health(gene_stack, self._layer_ids)
            if len(self._layer_ids) > 1
            else None
        )
        self._per_layer = {}
        for li, layer_id in enumerate(self._layer_ids):
            feats = gene_stack[:, li, :]  # [total_genes, H] for this layer
            per_genome: list[np.ndarray] = []
            cursor = 0
            for n in gene_counts:
                per_genome.append(feats[cursor : cursor + n])
                cursor += n
            self._per_layer[layer_id] = per_genome

    def score(self) -> TaskResult:
        if not self.genomes:
            raise RuntimeError("BacBenchEssentialityLayerSweepTask.prepare must be called before score")
        if not self._per_layer:
            raise RuntimeError("BacBenchEssentialityLayerSweepTask.run must be called before score")

        # Fit + evaluate the SAME probe once per layer (the GPU work was already done once).
        outcomes: dict[int, _ProbeOutcome] = {}
        for layer_id in self._layer_ids:
            try:
                outcomes[layer_id] = _fit_and_evaluate_probe(
                    self.genomes,
                    self._per_layer[layer_id],
                    threshold=self.threshold,
                    seed=self.seed,
                    probe_C=self.probe_C,
                    probe_max_iter=self.probe_max_iter,
                    task_name=self.name,
                )
            except _ProbeError as e:
                # train single-class / no train|test is layer-independent → the task errors.
                return TaskResult(
                    self.name, ResultStatus.ERROR, None, {}, {"error": str(e), "task_version": self.version}
                )

        # Whether any test genome has both classes is layer-independent (it's the labels).
        any_scored = next(iter(outcomes.values())).n_test_genomes_scored
        if any_scored == 0:
            md = self._sweep_metadata(outcomes, best_layer=None)
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {
                    "error": f"{self.name}: 0 test genomes had both classes (macro AUROC undefined).",
                    **md,
                },
            )

        def _auroc(layer_id: int) -> float:
            v = outcomes[layer_id].metrics["macro_mean_auroc"]
            return v if not math.isnan(v) else float("-inf")

        best_layer, self._layer_selection = choose_best_layer(
            list(self._layer_ids), _auroc, self._depth_health
        )
        best = outcomes[best_layer]

        # Headline metrics = the BEST layer's full metric set (so the row is directly
        # comparable to the single-layer task), plus every layer's AUROC/AUPRC + best_layer.
        metrics: dict[str, float] = dict(best.metrics)
        for layer_id in self._layer_ids:
            metrics[f"layer_{layer_id}_macro_mean_auroc"] = outcomes[layer_id].metrics["macro_mean_auroc"]
            metrics[f"layer_{layer_id}_macro_mean_auprc"] = outcomes[layer_id].metrics["macro_mean_auprc"]
        metrics["best_layer"] = float(best_layer)

        metadata = self._sweep_metadata(outcomes, best_layer=best_layer)
        return TaskResult(self.name, ResultStatus.OK, "macro_mean_auroc", metrics, metadata)

    def _sweep_metadata(
        self, outcomes: dict[int, _ProbeOutcome], *, best_layer: int | None
    ) -> dict[str, Any]:
        """Per-layer table + the best layer's full breakdown (timings, per-genome, dims)."""
        ref = outcomes[best_layer] if best_layer is not None else next(iter(outcomes.values()))
        md = self._probe_metadata(ref)  # best layer's per_genome + the shared timings/dims
        md["swept_layers"] = [int(layer_id) for layer_id in self._layer_ids]
        md["best_layer"] = int(best_layer) if best_layer is not None else None
        md.update(sweep_diagnostics(self._depth_health, self._layer_selection))
        md["per_layer"] = [
            {
                "layer": int(layer_id),
                "macro_mean_auroc": outcomes[layer_id].metrics["macro_mean_auroc"],
                "macro_median_auroc": outcomes[layer_id].metrics["macro_median_auroc"],
                "macro_mean_auprc": outcomes[layer_id].metrics["macro_mean_auprc"],
                "n_test_genomes_scored": outcomes[layer_id].n_test_genomes_scored,
            }
            for layer_id in self._layer_ids
        ]
        return md
