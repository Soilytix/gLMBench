"""BacBench gene-essentiality task (the first EMBEDDING-consuming task).

Gate: the ``bacbench-essentiality`` task yields a complete, well-formed ``TaskResult``
against an embedding adapter and a first-class ``N/A`` against a score-only adapter.
Stress tests cover: the macro + confusion metric keys; the per-genome breakdown with a
single-class genome excluded from the macro but counted; the batch-first invariant (ONE
``embed`` call); determinism; the data-integrity gate; the vendored-reference sha256 pin;
the gene+promoter extraction recipe; window-chunking; and a metric oracle (sklearn vs a
hand-computed AUROC/AP, plus an optional torchmetrics cross-check).

CPU-only and torch-free — rides the ``EchoAdapter`` (real wire boundary) and an in-process
composition adapter (label-separable embeddings so the probe actually learns), the
committed tiny fixture, and synthetic in-tmp manifests.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter
from glmbench.adapters.echo import EchoAdapter
from glmbench.tasks import bacbench_essentiality_metrics as bbm
from glmbench.tasks.bacbench_essentiality import (
    BacBenchEssentialityLayerSweepTask,
    BacBenchEssentialityTask,
    chunk_dna_sequence,
    parse_layer_list,
)
from glmbench.tasks.bacbench_essentiality_data import (
    ManifestHashError,
    get_dna_seq,
    load_genomes,
)
from glmbench.tasks.base import ResultStatus

REPO = Path(__file__).resolve().parents[2]
TINY_MANIFEST = REPO / "tests" / "data" / "bacbench_essentiality_tiny" / "tiny_manifest.yaml"
VENDOR = (
    REPO / "src" / "glmbench" / "tasks" / "_vendor" / "bacbench_essentiality"
)


# ---------------------------------------------------------------------------
# An in-process EMBEDDING adapter whose vectors encode base composition, so the
# fixture's A/C-rich (essential) vs T/G-rich (non-essential) genes are linearly
# separable → the probe learns and AUROC is high. Torch-free; no runner needed.
# ---------------------------------------------------------------------------
class _CompositionAdapter(ModelAdapter):
    name = "composition-embed"
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.EMBEDDING})

    def __init__(self) -> None:
        self.embed_calls: list[int] = []

    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return "glmb:composition-test"

    def describe(self) -> dict:
        return {"name": self.name, "embedding_dim": 5, "n_layers": 1}

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        self.embed_calls.append(len(sequences))
        arrays = []
        for s in sequences:
            n = max(1, len(s))
            vec = np.array(
                [
                    s.count("A") / n,
                    s.count("C") / n,
                    s.count("G") / n,
                    s.count("T") / n,
                    1.0,  # bias channel
                ],
                dtype=np.float64,
            )
            arrays.append(vec.reshape(1, 5))  # [len(layers)=1, H=5]
        return EmbeddingResult(
            arrays=arrays, layers=[-1], pool=pool, token_spans=None, embedding_dim=5
        )


class _CountingEmbed(ModelAdapter):
    """Wraps an adapter, counting embed calls (proves the batch-first invariant)."""

    name = "counting-embed"
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.EMBEDDING})

    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner
        self.calls: list[int] = []

    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return self._inner.model_hash()

    def describe(self) -> dict:
        return self._inner.describe()

    def embed(self, sequences, *, layers="last", pool="mean"):  # type: ignore[no-untyped-def]
        self.calls.append(len(sequences))
        return self._inner.embed(sequences, layers=layers, pool=pool)


# --- helpers to build synthetic in-tmp manifests ------------------------------
_EXPECTED_METRIC_KEYS = {
    "macro_mean_auroc",
    "macro_median_auroc",
    "macro_mean_auprc",
    "macro_median_auprc",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
    "n_test_genomes_scored",
    "n_test_genes",
}


def _write_manifest(tmp: Path, splits: dict[str, list[tuple[str, str, list[int]]]]) -> Path:
    """Build per-split CSVs + a (hashless) manifest in ``tmp``; return manifest path."""
    lines = ["promoter_len: 128", "splits:"]
    for split, genomes in splits.items():
        fname = f"{split}.csv"
        with open(tmp / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["genome_name", "genus", "essential", "sequence"])
            for gname, genus, labels in genomes:
                for i, lab in enumerate(labels):
                    seq = "ATG" + ("AAACAC" if lab else "TTTGTG") * (4 + i)
                    w.writerow([gname, genus, lab, seq])
        lines += [f"  - name: {split}", f"    file: {fname}"]
    mp = tmp / "manifest.yaml"
    mp.write_text("\n".join(lines) + "\n")
    return mp


# --- the gate: well-formed result vs an embedding adapter ---------------------


def test_gate_well_formed_result_against_embedding_adapter() -> None:
    task = BacBenchEssentialityTask(TINY_MANIFEST)
    result = task.evaluate(_CompositionAdapter())

    assert result.status is ResultStatus.OK
    assert result.task == "bacbench-essentiality"
    assert result.primary_metric == "macro_mean_auroc"

    m = result.metrics
    assert set(m) >= _EXPECTED_METRIC_KEYS
    # The fixture is linearly separable → the probe should learn it well.
    assert m["macro_mean_auroc"] > 0.9
    assert 0.0 <= m["macro_mean_auprc"] <= 1.0
    # Confusion counts are integers that sum to the number of scored test genes.
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["n_test_genes"]
    assert m["n_test_genes"] == 21.0  # tiny test split has 8 + 8 + 5 genes

    # Per-genome breakdown: 3 test genomes, the single-class one excluded from the macro.
    pg = result.metadata["per_genome"]
    assert len(pg) == 3
    assert result.metadata["n_excluded_single_class"] == 1
    assert result.metadata["n_test_genomes_scored"] == 2
    assert m["n_test_genomes_scored"] == 2.0
    # validation split is never embedded (probe doesn't need it).
    assert result.metadata["n_train_genomes"] == 3


def test_gate_na_against_score_only_adapter() -> None:
    adapter = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    result = BacBenchEssentialityTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.NA
    assert result.primary_metric is None
    assert result.metrics == {}
    assert result.metadata["missing_capabilities"] == ["embedding"]


def test_single_batched_embed_call() -> None:
    """All gene windows across all genomes are embedded in exactly ONE adapter call."""
    adapter = _CountingEmbed(_CompositionAdapter())
    result = BacBenchEssentialityTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert len(adapter.calls) == 1  # one batched embed
    # train(30) + test(21) genes, each a single window (short seqs) → 51 windows.
    assert adapter.calls[0] == 51
    assert result.metadata["n_windows_embedded"] == 51


def test_deterministic_metrics() -> None:
    r1 = BacBenchEssentialityTask(TINY_MANIFEST).evaluate(_CompositionAdapter())
    r2 = BacBenchEssentialityTask(TINY_MANIFEST).evaluate(_CompositionAdapter())
    assert r1.metrics == r2.metrics


def test_runs_end_to_end_against_echo_wire_boundary() -> None:
    """Against the real EchoAdapter (subprocess wire), the task still yields a clean OK.

    Echo embeddings carry no label signal, so AUROC ≈ chance — we assert structure/finite,
    not value: this exercises the actual embed wire round-trip end to end.
    """
    result = BacBenchEssentialityTask(TINY_MANIFEST).evaluate(EchoAdapter())
    assert result.status is ResultStatus.OK
    assert set(result.metrics) >= _EXPECTED_METRIC_KEYS
    assert result.metrics["macro_mean_auroc"] == result.metrics["macro_mean_auroc"]  # not NaN


# --- ERROR paths --------------------------------------------------------------


def test_all_test_genomes_single_class_is_error(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        {
            "train": [("TR", "Alpha", [1, 0, 1, 0, 1, 0])],
            "test": [("TE1", "Beta", [0, 0, 0]), ("TE2", "Gamma", [1, 1, 1])],
        },
    )
    task = BacBenchEssentialityTask(manifest, verify_hashes=False)
    result = task.evaluate(_CompositionAdapter())
    assert result.status is ResultStatus.ERROR
    assert "both classes" in result.metadata["error"]


def test_single_class_train_is_error(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        {
            "train": [("TR", "Alpha", [1, 1, 1, 1])],  # single class → probe undefined
            "test": [("TE", "Beta", [0, 1, 0, 1])],
        },
    )
    result = BacBenchEssentialityTask(manifest, verify_hashes=False).evaluate(_CompositionAdapter())
    assert result.status is ResultStatus.ERROR
    assert "single class" in result.metadata["error"]


# --- integrity gate bites through prepare() (outside evaluate's try) ----------


def test_verify_hashes_trips_on_tampered_fixture(tmp_path: Path) -> None:
    import shutil

    work = tmp_path / "fixture"
    shutil.copytree(TINY_MANIFEST.parent, work)
    target = work / "tiny_test.csv"
    target.write_text(target.read_text().replace("ATG", "ATC", 1))
    with pytest.raises(ManifestHashError, match="sha256 drift"):
        BacBenchEssentialityTask(work / "tiny_manifest.yaml").evaluate(_CompositionAdapter())


# --- vendored upstream reference is byte-identical (sha256 pin) ----------------


def test_vendored_reference_sha256_pin() -> None:
    sums = (VENDOR / "SHA256SUMS").read_text().split()[0]
    actual = hashlib.sha256((VENDOR / "run_train_cls.py").read_bytes()).hexdigest()
    assert actual == sums, "vendored run_train_cls.py drifted from its pinned sha256"


# --- gene + promoter extraction recipe (the BacBench get_dna_seq, no revcomp) --


def test_get_dna_seq_strand_and_promoter() -> None:
    dna = "".join("ACGT"[i % 4] for i in range(1000))
    # + strand: promoter upstream of start, end unchanged.
    plus = get_dna_seq(dna, start=200, end=260, strand="+", promoter_len=128)
    assert plus == dna[200 - 128 - 1 : 260]
    # - strand: promoter on the higher-coordinate (upstream for minus), start unchanged.
    minus = get_dna_seq(dna, start=200, end=260, strand="-", promoter_len=128)
    assert minus == dna[200 - 1 : 260 + 128]
    # None strand: promoter both sides.
    both = get_dna_seq(dna, start=200, end=260, strand=None, promoter_len=128)
    assert both == dna[200 - 128 - 1 : 260 + 128]
    # Clamping at the contig start (no negative slice).
    clamped = get_dna_seq(dna, start=10, end=60, strand="+", promoter_len=128)
    assert clamped == dna[0:60]
    # No reverse complement is ever applied (forward substring only).
    assert set(minus) <= set("ACGT")


def test_window_chunking_recipe() -> None:
    seq = "A" * 250
    # window 100, overlap 32 → stride 68: [0:100],[68:168],[136:236],[204:250]
    chunks = chunk_dna_sequence(seq, max_seq_len=100, overlap=32)
    assert [len(c) for c in chunks] == [100, 100, 100, 46]
    # A sequence shorter than the window is a single chunk.
    assert chunk_dna_sequence("ACGT", 100, 32) == ["ACGT"]


def test_long_gene_is_windowed_not_dropped() -> None:
    """A gene longer than max_context is window-chunked + mean-pooled, never skipped."""
    adapter = _CountingEmbed(_CompositionAdapter())
    # cap below the fixture's gene lengths so several genes split into >1 window.
    task = BacBenchEssentialityTask(TINY_MANIFEST, max_context=40)
    result = task.evaluate(adapter)
    assert result.status is ResultStatus.OK
    # more windows than genes (some genes split), but still ONE batched call.
    assert len(adapter.calls) == 1
    assert adapter.calls[0] > 51
    assert result.metrics["n_test_genes"] == 21.0  # genes preserved (none dropped)


# --- metric oracle: sklearn re-derivation vs hand-computed + optional torchmetrics ---


def test_metric_oracle_hand_computed() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    rk = bbm.genome_ranking_metrics(labels, scores)
    assert rk["auroc"] == pytest.approx(0.75)
    assert rk["auprc"] == pytest.approx(0.8333333, abs=1e-6)
    # Single-class genome → NaN (excluded from the macro).
    nan = bbm.genome_ranking_metrics([1, 1, 1], [0.2, 0.9, 0.5])
    assert nan["auroc"] != nan["auroc"]  # NaN


def test_confusion_counts_at_threshold() -> None:
    labels = [1, 1, 0, 0]
    probs = [0.9, 0.4, 0.6, 0.1]  # at 0.5: pred=[1,0,1,0]
    c = bbm.confusion_counts(labels, probs, threshold=0.5)
    assert (c["tp"], c["fp"], c["tn"], c["fn"]) == (1.0, 1.0, 1.0, 1.0)
    assert c["precision"] == pytest.approx(0.5)
    assert c["recall"] == pytest.approx(0.5)
    assert c["specificity"] == pytest.approx(0.5)
    assert c["f1"] == pytest.approx(0.5)


def test_macro_mean_and_median() -> None:
    per_genome = [
        {"auroc": 0.8, "auprc": 0.6},
        {"auroc": 0.6, "auprc": 0.4},
        {"auroc": float("nan"), "auprc": float("nan")},  # single-class → skipped
    ]
    macro = bbm.macro(per_genome)
    assert macro["macro_mean_auroc"] == pytest.approx(0.7)
    assert macro["macro_median_auroc"] == pytest.approx(0.7)
    assert macro["macro_mean_auprc"] == pytest.approx(0.5)


def test_metric_equivalence_torchmetrics_oracle() -> None:
    """sklearn AUROC/AUPRC == torchmetrics binary auroc/average_precision (a cross-check).

    GPU/torch-gated: skipped where torchmetrics is absent (core stays torch-free).
    """
    torch = pytest.importorskip("torch")
    tm = pytest.importorskip("torchmetrics.functional")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    if labels.min() == labels.max():
        labels[0] = 1 - labels[0]
    scores = rng.random(200)
    sk = bbm.genome_ranking_metrics(labels, scores)
    t_labels = torch.tensor(labels, dtype=torch.long)
    t_scores = torch.tensor(scores, dtype=torch.float32)
    tm_auroc = tm.auroc(t_scores, t_labels, task="binary").item()
    tm_auprc = tm.average_precision(t_scores, t_labels, task="binary").item()
    assert sk["auroc"] == pytest.approx(tm_auroc, abs=1e-6)
    assert sk["auprc"] == pytest.approx(tm_auprc, abs=1e-6)


# --- loader ignores non-requested splits --------------------------------------


def test_loader_loads_only_requested_splits() -> None:
    genomes = load_genomes(TINY_MANIFEST, splits=("train", "test"))
    splits = {g.split for g in genomes}
    assert splits == {"train", "test"}  # validation present in manifest but not loaded


# --- real-data path (skips on CI where the gitignored CSVs are absent) ---------

_REAL_TRAIN = (
    REPO / "src" / "glmbench" / "tasks" / "data" / "bacbench_essentiality"
    / "essential_genes_train.csv"
)


@pytest.mark.skipif(
    not _REAL_TRAIN.exists(),
    reason="vendored BacBench CSVs absent (gitignored cache); run scripts/fetch_bacbench_essentiality.py",
)
def test_real_data_loads() -> None:
    from glmbench.tasks.bacbench_essentiality_data import DEFAULT_MANIFEST

    genomes = load_genomes(DEFAULT_MANIFEST, splits=("train", "test"))
    assert len(genomes) > 0
    assert all(len(g) > 0 for g in genomes)


# ===========================================================================
# Layer-sweep variant — best embedding layer for essentiality (one forward pass)
# ===========================================================================


def _det_seed(s: str, layer: int) -> int:
    """A stable per-(sequence, layer) seed (FNV-1a; deterministic across runs)."""
    h = 1469598103934665603
    for ch in f"{s}|{layer}".encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


class _LayeredEmbedAdapter(ModelAdapter):
    """Multi-layer EMBEDDING adapter: ``good_layers`` carry the separable composition signal,
    every other layer carries deterministic label-independent noise (≈ chance AUROC). So the
    sweep's best layer must fall in ``good_layers`` — a real "layers differ" test. Returns all
    requested layers in ONE embed call (the any-layer route). Torch-free; no runner.
    """

    name = "layered-embed"
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.EMBEDDING})

    def __init__(self, n_layers: int = 12, good_layers=frozenset({11, 12})) -> None:
        self.n_layers = n_layers
        self.good_layers = frozenset(good_layers)
        self.embed_calls: list[int] = []

    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return "glmb:layered-test"

    def describe(self) -> dict:
        return {"name": self.name, "n_layers": self.n_layers, "embedding_dim": 5}

    def _resolve(self, layers) -> list[int]:
        all_layers = list(range(self.n_layers + 1))
        if layers == "all":
            return all_layers
        if layers == "last":
            return [self.n_layers]
        if isinstance(layers, int):
            return [all_layers[layers]]
        return [all_layers[int(i)] for i in layers]

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        self.embed_calls.append(len(sequences))
        sel = self._resolve(layers)
        arrays = []
        for s in sequences:
            n = max(1, len(s))
            comp = np.array(
                [s.count("A") / n, s.count("C") / n, s.count("G") / n, s.count("T") / n, 1.0],
                dtype=np.float64,
            )
            rows = []
            for layer in sel:
                if layer in self.good_layers:
                    rows.append(comp)  # label-separable (A/C-rich vs T/G-rich)
                else:
                    rng = np.random.default_rng(_det_seed(s, layer))
                    rows.append(rng.standard_normal(5))  # no label signal → ≈ chance
            arrays.append(np.stack(rows, axis=0))  # [len(sel), 5]
        return EmbeddingResult(
            arrays=arrays, layers=sel, pool=pool, token_spans=None, embedding_dim=5
        )


def test_parse_layer_list() -> None:
    assert parse_layer_list("6,7,8,9,10,11,12") == [6, 7, 8, 9, 10, 11, 12]
    assert parse_layer_list("6-12") == [6, 7, 8, 9, 10, 11, 12]
    assert parse_layer_list("0,4-6,12") == [0, 4, 5, 6, 12]
    assert parse_layer_list("-1,-2") == [-1, -2]
    assert parse_layer_list("all") is None
    assert parse_layer_list("") is None
    assert parse_layer_list(None) is None


def test_layer_sweep_default_sweeps_all_layers_one_call() -> None:
    """No layer config → sweep every layer the model returns, in ONE embed call."""
    adapter = _LayeredEmbedAdapter(n_layers=12, good_layers=frozenset({11, 12}))
    result = BacBenchEssentialityLayerSweepTask(TINY_MANIFEST).evaluate(adapter)

    assert result.status is ResultStatus.OK
    assert result.task == "bacbench-essentiality-layer-sweep"
    assert result.primary_metric == "macro_mean_auroc"
    assert len(adapter.embed_calls) == 1  # the GPU work happens ONCE
    assert adapter.embed_calls[0] == 51    # all gene windows in the one call

    # every layer 0..12 has a reported AUROC; the headline keys are the base metric set too
    assert set(result.metrics) >= _EXPECTED_METRIC_KEYS
    for layer in range(13):
        assert f"layer_{layer}_macro_mean_auroc" in result.metrics
    assert result.metadata["swept_layers"] == list(range(13))
    assert len(result.metadata["per_layer"]) == 13

    # the sweep finds a GOOD layer (separable) as the winner; headline == best layer's AUROC
    best = int(result.metrics["best_layer"])
    assert best in {11, 12}
    assert result.metadata["best_layer"] == best
    assert result.metrics["macro_mean_auroc"] == pytest.approx(
        result.metrics[f"layer_{best}_macro_mean_auroc"]
    )
    # a good layer clearly beats a noise layer
    assert result.metrics[f"layer_{best}_macro_mean_auroc"] > 0.9
    assert result.metrics["layer_3_macro_mean_auroc"] < result.metrics[f"layer_{best}_macro_mean_auroc"]


def test_layer_sweep_explicit_range_6_to_12() -> None:
    """An explicit list restricts the sweep to exactly those layers (the 6-12 use case)."""
    adapter = _LayeredEmbedAdapter(n_layers=12, good_layers=frozenset({9}))
    task = BacBenchEssentialityLayerSweepTask(TINY_MANIFEST, sweep_layers=[6, 7, 8, 9, 10, 11, 12])
    result = task.evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert result.metadata["swept_layers"] == [6, 7, 8, 9, 10, 11, 12]
    assert len(adapter.embed_calls) == 1
    assert "layer_5_macro_mean_auroc" not in result.metrics  # layer 5 not swept
    assert int(result.metrics["best_layer"]) == 9  # the only good layer in range


def test_layer_sweep_env_var_configures_layers(monkeypatch) -> None:
    """GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS configures the sweep without code/args (CLI seam)."""
    monkeypatch.setenv("GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS", "6-12")
    task = BacBenchEssentialityLayerSweepTask(TINY_MANIFEST)
    assert task.sweep_layers == [6, 7, 8, 9, 10, 11, 12]


def test_layer_sweep_na_against_score_only_adapter() -> None:
    adapter = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    result = BacBenchEssentialityLayerSweepTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.NA
    assert result.metadata["missing_capabilities"] == ["embedding"]
