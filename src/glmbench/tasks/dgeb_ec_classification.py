"""DGEB EC classification (DNA) — a multiclass EMBEDDING probe over coding sequences.

Probe the frozen sequence embeddings of a genomic LM for **Enzyme Commission (EC) class**
(single-label, 128-way classification at the gene scale). The task is **model-agnostic**:
it requests one pooled embedding per coding sequence from the adapter (the heavy forward
pass runs in the adapter's own runner), then fits + evaluates the DGEB logistic-regression
probe on the train/test split (the probe is task logic, never behind the adapter).

Lifecycle:

- :meth:`prepare` loads the train+test splits via the sha256-pinned loader
  (:mod:`dgeb_ec_data`); each record is a ``(entry, DNA sequence, EC label)``, U→T-normalized.
- :meth:`run` resolves over-context sequences (max 14.7 kb > every model's context) by one
  of two toggleable strategies — **window-and-mean** (default; chunk to the context, mean
  the chunk vectors, never drops an instance) or **truncate-central** (DGEB-style, keep the
  central context-length piece) — flattens **all** windows across train+test into ONE batched
  ``adapter.embed(layers=…, pool="mean")`` call (batch-first), and means windows back
  into one embedding per sequence.
- :meth:`score` optionally standardizes (off by default), then fits + evaluates the
  **vendored DGEB** ``logRegClassificationEvaluator`` (:mod:`dgeb_ec_metrics`). ``primary_metric``
  is **macro-F1** (the paper's reported EC metric); accuracy is also exposed.

**Metric authority:** unlike BacBench, DGEB's classification evaluator is torch-free
(sklearn LogisticRegression + ``f1_score``/``accuracy_score``), so it is vendored *and
called* char-for-char — see ``_vendor/dgeb/NOTICE``. This is a true vendor, not a
re-derivation.

**Layer choice** mirrors the gene-essentiality task: the primary task embeds at a single
configurable layer (default ``"last"``; set via arg or ``GLMBENCH_DGEB_EC_LAYER``), and a
sibling :class:`DGEBECClassificationLayerSweepTask` sweeps every layer in ONE forward pass
to find the best. An adapter that serves only its final layer uses ``"last"``.

This module is core (torch-free): stdlib + numpy + scikit-learn (core deps).
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from glmbench import registry
from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter

from .base import ResultStatus, Task, TaskResult
from ._depth import choose_best_layer, measure_depth_health, sweep_diagnostics
from .dgeb_ec_data import DEFAULT_MANIFEST as _DEFAULT_MANIFEST
from .dgeb_ec_data import ECSplit, load_ec_splits
from .dgeb_ec_metrics import classification_scores

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = _DEFAULT_MANIFEST

# Env seams (the benchmark CLI builds tasks with default args, so these are the only
# command-line knobs without a custom YAML — mirrors the bacbench essentiality task).
_MANIFEST_ENV = "GLMBENCH_DGEB_EC_MANIFEST"
_LAYER_ENV = "GLMBENCH_DGEB_EC_LAYER"  # "last" | "<int>"
_OVERFLOW_ENV = "GLMBENCH_DGEB_EC_OVERFLOW"  # "window" | "truncate"
_SCALE_ENV = "GLMBENCH_DGEB_EC_SCALE"  # truthy → StandardScaler on
_SWEEP_LAYERS_ENV = "GLMBENCH_DGEB_EC_SWEEP_LAYERS"  # "all" | "6-12" | "0,4,8"
_SWEEP_POOL_ENV = "GLMBENCH_DGEB_EC_SWEEP_POOL"

# Window-chunk parameters (BacBench ``embed_dna.py`` recipe: overlap 32, min 32).
DEFAULT_WINDOW_OVERLAP = 32
DEFAULT_MIN_SEQ_LEN = 32

# How an over-context sequence is reduced to one embedding.
OVERFLOW_WINDOW = "window"  # chunk to context, mean the chunk vectors (default; no drop)
OVERFLOW_TRUNCATE = "truncate"  # keep the central context-length piece (DGEB-style)
_OVERFLOW_STRATEGIES = (OVERFLOW_WINDOW, OVERFLOW_TRUNCATE)

_TRAIN = "train"
_TEST = "test"
_SPLITS = (_TRAIN, _TEST)

# Probe default (DGEB ``logRegClassificationEvaluator`` defaults: max_iter 1000, seed 42).
DEFAULT_MAX_ITER = 1000
DEFAULT_SEED = 42


def chunk_dna_sequence(dna_seq: str, max_seq_len: int, overlap: int) -> list[str]:
    """Overlapping windows of length ``max_seq_len`` (stride ``max_seq_len - overlap``).

    Byte-for-byte the BacBench ``embed_dna.chunk_dna_sequence`` recipe: emit windows until
    the end is reached; a sequence ``<= max_seq_len`` yields a single chunk.
    """
    chunks: list[str] = []
    n = len(dna_seq)
    # Guard against a non-positive stride (overlap >= window) that would loop forever;
    # the BacBench recipe assumes overlap < window, which holds for any real model context.
    stride = max(1, max_seq_len - overlap)
    start = 0
    while start < n:
        end = min(start + max_seq_len, n)
        chunks.append(dna_seq[start:end])
        if end == n:
            break
        start += stride
    return chunks


def central_window(seq: str, max_context: int) -> str:
    """The central ``max_context``-nt slice of ``seq`` (the truncate-central strategy).

    DGEB truncates over-context sequences; gLMBench never truncates *silently*, so this is
    offered as an explicit, counted strategy. Taking the centre (vs the head) keeps signal
    from both ends of the gene rather than dropping the 3' half.
    """
    n = len(seq)
    if n <= max_context:
        return seq
    start = (n - max_context) // 2
    return seq[start : start + max_context]


def parse_layer_list(value: str | None) -> list[int] | None:
    """Parse a layer-selection string → ``list[int]`` or ``None`` (= every layer).

    Accepts a comma list (``"0,4,8"``), inclusive ranges (``"6-12"``), a mix
    (``"0,4-6,12"``), negatives (``"-1,-2"``), or ``"all"``/``""`` → ``None``. Layer
    numbers are not hardcoded — a model of any depth gets the matching selection.
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
        if "-" in tok and not tok.startswith("-"):
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out or None


def _truthy(value: str | None) -> bool | None:
    """Parse an env flag to bool; ``None`` if unset (so a code/arg default can win)."""
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_layer_arg(layer: int | str) -> list[int] | str:
    """Map the single-layer selector to an ``adapter.embed`` ``layers`` argument."""
    if isinstance(layer, str):
        if layer.strip().lower() == "last":
            return "last"
        return [int(layer)]
    return [int(layer)]


class _ECProbeData:
    """Per-layer feature matrices + labels + accounting, produced by the embed pass."""

    def __init__(
        self,
        layer_ids: list[int],
        train_stack: np.ndarray,  # [n_train, n_layers, H]
        test_stack: np.ndarray,  # [n_test, n_layers, H]
        y_train: list[str],
        y_test: list[str],
        accounting: dict[str, Any],
    ) -> None:
        self.layer_ids = layer_ids
        self.train_stack = train_stack
        self.test_stack = test_stack
        self.y_train = y_train
        self.y_test = y_test
        self.accounting = accounting


class DGEBECClassificationBase(Task):
    """Shared lifecycle for the EC classification probe (single-layer + layer-sweep)."""

    required_capabilities = frozenset({Capability.EMBEDDING})
    higher_is_better = True  # macro-F1

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        data_dir: Path | str | None = None,
        verify_hashes: bool = True,
        pool: str = "mean",
        overflow_strategy: str | None = None,
        max_context: int | None = None,
        window_overlap: int = DEFAULT_WINDOW_OVERLAP,
        min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
        standard_scale: bool | None = None,
        max_iter: int = DEFAULT_MAX_ITER,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if manifest_path is None:
            manifest_path = os.environ.get(_MANIFEST_ENV) or DEFAULT_MANIFEST
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir) if data_dir is not None else self.manifest_path.parent
        self.verify_hashes = verify_hashes
        self.pool = pool

        if overflow_strategy is None:
            overflow_strategy = os.environ.get(_OVERFLOW_ENV) or OVERFLOW_WINDOW
        if overflow_strategy not in _OVERFLOW_STRATEGIES:
            raise ValueError(
                f"overflow_strategy must be one of {_OVERFLOW_STRATEGIES}, got "
                f"{overflow_strategy!r}."
            )
        self.overflow_strategy = overflow_strategy

        self.max_context = max_context
        self.window_overlap = window_overlap
        self.min_seq_len = min_seq_len

        if standard_scale is None:
            env_scale = _truthy(os.environ.get(_SCALE_ENV))
            standard_scale = env_scale if env_scale is not None else False
        self.standard_scale = standard_scale

        self.max_iter = max_iter
        self.seed = seed

        self.splits: dict[str, ECSplit] = {}
        self._probe: _ECProbeData | None = None

    # --- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        self.splits = load_ec_splits(
            self.manifest_path,
            data_dir=self.data_dir,
            verify_hashes=self.verify_hashes,
            splits=_SPLITS,
        )

    def _instance_windows(self, seq: str, max_context: int | None) -> tuple[list[str], bool]:
        """Reduce one sequence to its embed windows + whether it overflowed the context.

        ``window`` → overlapping chunks (mean-pooled later); ``truncate`` → the central
        context-length slice. A sequence within context (or ``max_context is None``) is a
        single whole window and does not count as overflow.
        """
        if max_context is None or len(seq) <= max_context:
            return [seq], False
        if self.overflow_strategy == OVERFLOW_TRUNCATE:
            return [central_window(seq, max_context)], True
        chunks = [
            c
            for c in chunk_dna_sequence(seq, max_context, self.window_overlap)
            if len(c) >= self.min_seq_len
        ]
        # Never drop a labeled instance: if every window fell under min_seq_len, keep the
        # central slice rather than losing the row.
        return (chunks if chunks else [central_window(seq, max_context)]), True

    def _embed(self, adapter: ModelAdapter, layers_arg: list[int] | str) -> _ECProbeData:
        """ONE batched embed over all windows of train+test; mean windows → per-instance.

        Returns per-instance feature matrices keeping the layer axis ([n, n_layers, H]) so
        the single-layer task squeezes it and the sweep iterates it. Asserts the 1:1
        contract and guards NaN loudly.
        """
        max_context = self.max_context
        if max_context is None:
            max_context = getattr(adapter, "max_context", None)
            if max_context in (0, None):  # adapters may report 0 = "unknown"
                max_context = None

        flat_chunks: list[str] = []
        per_instance_span: list[tuple[int, int]] = []  # train then test, in order
        n_overflow = {_TRAIN: 0, _TEST: 0}
        counts: dict[str, int] = {}
        for split_name in _SPLITS:
            split = self.splits[split_name]
            counts[split_name] = len(split)
            for seq in split.sequences:
                start = len(flat_chunks)
                windows, overflowed = self._instance_windows(seq, max_context)
                flat_chunks.extend(windows)
                per_instance_span.append((start, len(flat_chunks)))
                if overflowed:
                    n_overflow[split_name] += 1

        if not flat_chunks:
            raise RuntimeError("dgeb-ec-classification-dna: no sequences to embed (empty data).")

        _t0 = time.perf_counter()
        result: EmbeddingResult = adapter.embed(flat_chunks, layers=layers_arg, pool=self.pool)
        embed_seconds = time.perf_counter() - _t0
        logger.info(
            "dgeb-ec-classification-dna: embedded %d windows in %.1fs (%.4fs/window).",
            len(flat_chunks),
            embed_seconds,
            embed_seconds / max(1, len(flat_chunks)),
        )
        if len(result.arrays) != len(flat_chunks):
            raise ValueError(
                f"adapter returned {len(result.arrays)} embeddings for {len(flat_chunks)} "
                "windows — batched embed must be 1:1 with its inputs."
            )

        layer_ids = [int(x) for x in result.layers]
        # Normalize each window array to [n_layers, H] (a single-layer adapter may return
        # [H] or [1, H]); stack to [n_windows, n_layers, H].
        win = np.stack(
            [np.atleast_2d(np.asarray(a, dtype=np.float64)) for a in result.arrays], axis=0
        )
        if win.shape[1] != len(layer_ids):
            raise ValueError(
                f"expected per-window arrays with layer axis = {len(layer_ids)}; got {win.shape}."
            )
        if np.isnan(win).any():
            raise ValueError("dgeb-ec-classification-dna: adapter returned a NaN embedding.")
        embedding_dim = int(win.shape[2])

        # Mean windows back into one [n_layers, H] vector per instance (train then test).
        per_instance = np.stack(
            [win[a:b].mean(axis=0) for (a, b) in per_instance_span], axis=0
        )  # [n_train + n_test, n_layers, H]
        n_train = counts[_TRAIN]
        train_stack = per_instance[:n_train]
        test_stack = per_instance[n_train:]

        accounting = {
            "n_train": n_train,
            "n_test": counts[_TEST],
            "n_classes_train": self.splits[_TRAIN].n_classes,
            "n_classes_test": self.splits[_TEST].n_classes,
            "n_windows_embedded": len(flat_chunks),
            "n_train_overflow": n_overflow[_TRAIN],
            "n_test_overflow": n_overflow[_TEST],
            "embedding_dim": embedding_dim,
            "embed_seconds": round(embed_seconds, 3),
        }
        if n_overflow[_TRAIN] or n_overflow[_TEST]:
            logger.warning(
                "dgeb-ec-classification-dna: %d train + %d test sequences exceeded the "
                "context and were handled by '%s' (never silently dropped).",
                n_overflow[_TRAIN],
                n_overflow[_TEST],
                self.overflow_strategy,
            )
        return _ECProbeData(
            layer_ids=layer_ids,
            train_stack=train_stack,
            test_stack=test_stack,
            y_train=list(self.splits[_TRAIN].labels),
            y_test=list(self.splits[_TEST].labels),
            accounting=accounting,
        )

    def _score_one_layer(
        self, train_X: np.ndarray, test_X: np.ndarray, y_train: list[str], y_test: list[str]
    ) -> dict[str, float]:
        """Optionally standardize, then run the vendored DGEB probe on one layer's features."""
        if self.standard_scale:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(train_X)
            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)
        return classification_scores(
            train_X, y_train, test_X, y_test, max_iter=self.max_iter, seed=self.seed
        )

    def _base_metadata(self, probe: _ECProbeData) -> dict[str, Any]:
        md: dict[str, Any] = dict(probe.accounting)
        md.update(
            {
                "overflow_strategy": self.overflow_strategy,
                "pool": self.pool,
                "standard_scale": self.standard_scale,
                "max_context": self.max_context,
                "max_iter": self.max_iter,
                "probe_seed": self.seed,
                "probe": "vendored DGEB logRegClassificationEvaluator (sklearn)",
                "primary_metric_note": "macro-F1 (f1_score average='macro'), DGEB EC metric",
                "task_version": self.version,
            }
        )
        return md


@registry.register("task", "dgeb-ec-classification-dna")
class DGEBECClassificationTask(DGEBECClassificationBase):
    """EC class probe on a single embedding layer (macro-F1). Full real-fleet coverage."""

    name = "dgeb-ec-classification-dna"
    # Shares one forward pass with the other tasks over this corpus (adapters/reuse.py).
    fusion_group = "dgeb-ec-windows"
    version = "1.0"

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        layer: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(manifest_path, **kwargs)
        if layer is None:
            env_layer = os.environ.get(_LAYER_ENV)
            layer = env_layer if env_layer else "last"
        # Keep "last" symbolic; coerce a numeric string to int.
        if isinstance(layer, str) and layer.strip().lower() != "last":
            layer = int(layer)
        self.layer: int | str = layer
        self._layer_id: int | None = None

    def run(self, adapter: ModelAdapter) -> None:
        self._probe = self._embed(adapter, _resolve_layer_arg(self.layer))
        self._layer_id = self._probe.layer_ids[0]

    def score(self) -> TaskResult:
        if not self.splits:
            raise RuntimeError("DGEBECClassificationTask.prepare must be called before score")
        if self._probe is None:
            raise RuntimeError("DGEBECClassificationTask.run must be called before score")

        probe = self._probe
        train_X = probe.train_stack[:, 0, :]
        test_X = probe.test_stack[:, 0, :]

        if len(set(probe.y_train)) < 2:
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {"error": f"{self.name}: train split has <2 EC classes — probe undefined.",
                 **self._base_metadata(probe)},
            )

        raw = self._score_one_layer(train_X, test_X, probe.y_train, probe.y_test)
        if math.isnan(raw.get("f1", float("nan"))):
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {"error": f"{self.name}: macro-F1 is NaN (poisoned probe).",
                 **self._base_metadata(probe)},
            )

        metrics: dict[str, float] = {
            "f1": raw["f1"],
            "accuracy": raw["accuracy"],
            "n_train": float(probe.accounting["n_train"]),
            "n_test": float(probe.accounting["n_test"]),
            "n_classes": float(probe.accounting["n_classes_train"]),
        }
        metadata = self._base_metadata(probe)
        metadata["layer"] = self._layer_id
        metadata["feature_source"] = (
            f"embedding (hidden-state, layer={self._layer_id}, pool='{self.pool}')"
        )
        return TaskResult(self.name, ResultStatus.OK, "f1", metrics, metadata)


@registry.register("task", "dgeb-ec-classification-dna-layer-sweep")
class DGEBECClassificationLayerSweepTask(DGEBECClassificationBase):
    """Find the best EC-classification layer — all layers in ONE forward pass.

    Captures every requested layer's pooled embedding in a single batched
    ``adapter.embed(layers=…)`` call (the GPU work happens once — the point of the
    any-layer adapters), then fits a **separate** DGEB probe per layer on CPU and reports
    each layer's macro-F1 plus the winner. Same data/windowing/probe as the single-layer
    task; only the feature is swept over the layer axis.

    **Layer selection is not hardcoded** — by default it sweeps every layer the model
    returns (``layers='all'`` → ``0..n_layers``). Pass ``sweep_layers=[…]`` or set
    ``GLMBENCH_DGEB_EC_SWEEP_LAYERS`` (e.g. ``"6-12"``) to restrict it. An adapter that
    serves only its final layer errors on non-final layers — use the single-layer task
    (``layer='last'``) for it.
    """

    name = "dgeb-ec-classification-dna-layer-sweep"
    # Shares one forward pass with the other tasks over this corpus (adapters/reuse.py).
    fusion_group = "dgeb-ec-windows"
    fusion_wants_all_layers = True
    version = "1.0"

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        *,
        sweep_layers: list[int] | None = None,
        pool: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_pool = pool or os.environ.get(_SWEEP_POOL_ENV) or "mean"
        super().__init__(manifest_path, pool=resolved_pool, **kwargs)
        if sweep_layers is None:
            sweep_layers = parse_layer_list(os.environ.get(_SWEEP_LAYERS_ENV))
        self.sweep_layers = sweep_layers  # None ⇒ ask the adapter for "all"

    def run(self, adapter: ModelAdapter) -> None:
        layers_arg: list[int] | str = (
            list(self.sweep_layers) if self.sweep_layers is not None else "all"
        )
        self._probe = self._embed(adapter, layers_arg)

    def score(self) -> TaskResult:
        if not self.splits:
            raise RuntimeError("DGEBECClassificationLayerSweepTask.prepare must run before score")
        if self._probe is None:
            raise RuntimeError("DGEBECClassificationLayerSweepTask.run must run before score")

        probe = self._probe
        if len(set(probe.y_train)) < 2:
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {"error": f"{self.name}: train split has <2 EC classes — probe undefined.",
                 **self._base_metadata(probe)},
            )

        per_layer: dict[int, dict[str, float]] = {}
        for li, layer_id in enumerate(probe.layer_ids):
            per_layer[layer_id] = self._score_one_layer(
                probe.train_stack[:, li, :],
                probe.test_stack[:, li, :],
                probe.y_train,
                probe.y_test,
            )

        def _f1(layer_id: int) -> float:
            v = per_layer[layer_id]["f1"]
            return v if not math.isnan(v) else float("-inf")

        # Depth health off the array this sweep already built, and a `best_layer` that
        # cannot be a numerically empty tap. The train stack is the feature the probe is
        # actually fitted on, so it is the right one to measure.
        health = measure_depth_health(probe.train_stack, list(probe.layer_ids))
        best_layer, selection = choose_best_layer(list(probe.layer_ids), _f1, health)
        if math.isnan(per_layer[best_layer]["f1"]):
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {"error": f"{self.name}: macro-F1 is NaN on every layer (poisoned probe).",
                 **self._base_metadata(probe)},
            )

        best = per_layer[best_layer]
        metrics: dict[str, float] = {
            "f1": best["f1"],
            "accuracy": best["accuracy"],
            "best_layer": float(best_layer),
            "n_train": float(probe.accounting["n_train"]),
            "n_test": float(probe.accounting["n_test"]),
            "n_classes": float(probe.accounting["n_classes_train"]),
        }
        for layer_id in probe.layer_ids:
            metrics[f"layer_{layer_id}_f1"] = per_layer[layer_id]["f1"]
            metrics[f"layer_{layer_id}_accuracy"] = per_layer[layer_id]["accuracy"]

        metadata = self._base_metadata(probe)
        metadata.update(
            {
                "swept_layers": [int(x) for x in probe.layer_ids],
                **sweep_diagnostics(health, selection),
                "best_layer": int(best_layer),
                "feature_source": f"embedding hidden states (per-layer sweep, pool='{self.pool}')",
                "per_layer": [
                    {
                        "layer": int(layer_id),
                        "f1": per_layer[layer_id]["f1"],
                        "accuracy": per_layer[layer_id]["accuracy"],
                    }
                    for layer_id in probe.layer_ids
                ],
            }
        )
        return TaskResult(self.name, ResultStatus.OK, "f1", metrics, metadata)
