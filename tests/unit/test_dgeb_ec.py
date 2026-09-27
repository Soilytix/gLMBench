"""DGEB EC classification (DNA) task — the multiclass EMBEDDING probe.

Gate: the ``dgeb-ec-classification-dna`` task yields a complete, well-formed ``TaskResult``
against an embedding adapter and a first-class ``N/A`` against a score-only adapter. Stress
tests cover: the metric keys (macro-F1 primary + accuracy); batch-first (ONE embed call);
the window-vs-truncate overflow toggle; the standard-scale toggle; the configurable layer +
env seam; the layer sweep; determinism; the data-integrity gate; the vendored-evaluator
sha256 pin; and a metric-equivalence test (wrapper == calling the vendored DGEB evaluator
directly).

CPU-only and torch-free — rides an in-process composition adapter (label-separable
embeddings so the probe actually learns), the real ``EchoAdapter`` wire boundary, the
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
from glmbench.tasks.base import ResultStatus
from glmbench.tasks.dgeb_ec_classification import (
    DGEBECClassificationLayerSweepTask,
    DGEBECClassificationTask,
    central_window,
    chunk_dna_sequence,
    parse_layer_list,
)
from glmbench.tasks.dgeb_ec_data import (
    ECClassificationContractError,
    ManifestHashError,
    load_ec_splits,
)
from glmbench.tasks.dgeb_ec_metrics import classification_scores

REPO = Path(__file__).resolve().parents[2]
TINY_MANIFEST = REPO / "tests" / "data" / "dgeb_ec_dna_tiny" / "tiny_manifest.yaml"
VENDOR = REPO / "src" / "glmbench" / "tasks" / "_vendor" / "dgeb"

_EXPECTED_METRIC_KEYS = {"f1", "accuracy", "n_train", "n_test", "n_classes"}


# ---------------------------------------------------------------------------
# In-process EMBEDDING adapter whose vectors encode base composition, so the
# fixture's A/C/G-rich classes are linearly separable → the probe learns and
# macro-F1 is high. Multi-layer aware: ``good_layers`` carry the composition signal,
# every other layer returns a CONSTANT (non-separable) vector → a degenerate probe →
# low F1, so the sweep's best layer is unambiguously a good one. Torch-free; no runner.
# ---------------------------------------------------------------------------
class _CompositionAdapter(ModelAdapter):
    name = "composition-embed"
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.EMBEDDING})

    def __init__(self, n_layers: int = 1, good_layers=None, max_context: int | None = None) -> None:
        self.n_layers = n_layers
        self.good_layers = frozenset(good_layers) if good_layers is not None else None
        self.max_context = max_context
        self.embed_calls: list[int] = []

    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return "glmb:composition-test"

    def describe(self) -> dict:
        return {"name": self.name, "embedding_dim": 5, "n_layers": self.n_layers}

    def _resolve(self, layers) -> list[int]:
        alll = list(range(self.n_layers + 1))
        if layers == "all":
            return alll
        if layers == "last":
            return [alll[-1]]
        return [alll[int(i)] for i in layers]

    def _comp(self, s: str) -> np.ndarray:
        n = max(1, len(s))
        return np.array(
            [s.count("A") / n, s.count("C") / n, s.count("G") / n, s.count("T") / n, 1.0],
            dtype=np.float64,
        )

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        self.embed_calls.append(len(sequences))
        sel = self._resolve(layers)
        arrays = []
        for s in sequences:
            comp = self._comp(s)
            rows = []
            for layer in sel:
                if self.good_layers is None or layer in self.good_layers:
                    rows.append(comp)
                else:
                    rows.append(np.zeros(5, dtype=np.float64))  # constant → non-separable
            arrays.append(np.stack(rows, axis=0))  # [len(sel), 5]
        return EmbeddingResult(arrays=arrays, layers=sel, pool=pool, token_spans=None, embedding_dim=5)


class _CountingEmbed(ModelAdapter):
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

    @property
    def max_context(self):  # type: ignore[no-untyped-def]
        return getattr(self._inner, "max_context", None)

    def embed(self, sequences, *, layers="last", pool="mean"):  # type: ignore[no-untyped-def]
        self.calls.append(len(sequences))
        return self._inner.embed(sequences, layers=layers, pool=pool)


def _write_manifest(tmp: Path, splits: dict[str, list[tuple[str, str]]]) -> Path:
    """Build per-split CSVs + a (hashless) manifest in ``tmp``; ``splits`` = {name: [(seq,label)]}."""
    lines = ["splits:"]
    for split, rows in splits.items():
        fname = f"{split}.csv"
        with open(tmp / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entry", "sequence", "label"])
            for i, (seq, label) in enumerate(rows):
                w.writerow([f"{split}{i}", seq, label])
        lines += [f"  - name: {split}", f"    file: {fname}", "    sha256: null"]
    mp = tmp / "manifest.yaml"
    mp.write_text("\n".join(lines) + "\n")
    return mp


# --- the gate: well-formed result vs an embedding adapter ---------------------


def test_gate_well_formed_result_against_embedding_adapter() -> None:
    task = DGEBECClassificationTask(TINY_MANIFEST)
    result = task.evaluate(_CompositionAdapter())

    assert result.status is ResultStatus.OK
    assert result.task == "dgeb-ec-classification-dna"
    assert result.primary_metric == "f1"
    assert set(result.metrics) >= _EXPECTED_METRIC_KEYS
    # composition-separable fixture → the probe learns it (perfect on 3 clean classes).
    assert result.metrics["f1"] > 0.9
    assert result.metrics["accuracy"] > 0.9
    assert result.metrics["n_train"] == 12.0
    assert result.metrics["n_test"] == 6.0
    assert result.metrics["n_classes"] == 3.0
    assert result.metadata["overflow_strategy"] == "window"
    assert result.metadata["standard_scale"] is False


def test_gate_na_against_score_only_adapter() -> None:
    adapter = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    result = DGEBECClassificationTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.NA
    assert result.primary_metric is None
    assert result.metrics == {}
    assert result.metadata["missing_capabilities"] == ["embedding"]


def test_single_batched_embed_call() -> None:
    """All sequences across both splits are embedded in exactly ONE adapter call."""
    adapter = _CountingEmbed(_CompositionAdapter())
    result = DGEBECClassificationTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert len(adapter.calls) == 1
    assert adapter.calls[0] == 18  # 12 train + 6 test, each one window (short seqs)
    assert result.metadata["n_windows_embedded"] == 18


def test_deterministic_metrics() -> None:
    r1 = DGEBECClassificationTask(TINY_MANIFEST).evaluate(_CompositionAdapter())
    r2 = DGEBECClassificationTask(TINY_MANIFEST).evaluate(_CompositionAdapter())
    assert r1.metrics == r2.metrics


def test_runs_end_to_end_against_echo_wire_boundary() -> None:
    """Against the real EchoAdapter (subprocess wire), the task yields a clean OK.

    Echo embeddings carry no label signal, so macro-F1 ≈ chance — we assert structure +
    finiteness, exercising the actual embed wire round-trip end to end.
    """
    result = DGEBECClassificationTask(TINY_MANIFEST).evaluate(EchoAdapter())
    assert result.status is ResultStatus.OK
    assert set(result.metrics) >= _EXPECTED_METRIC_KEYS
    assert result.metrics["f1"] == result.metrics["f1"]  # not NaN


# --- overflow toggle: window vs truncate -------------------------------------


def test_central_window() -> None:
    assert central_window("ABCDEFGHIJ", 4) == "DEFG"  # central 4 of 10: start=(10-4)//2=3
    assert central_window("ABCDE", 10) == "ABCDE"  # within context → unchanged


def test_window_strategy_chunks_long_sequences_one_call() -> None:
    adapter = _CountingEmbed(_CompositionAdapter())
    # cap below the fixture's ~45 nt genes so several split into >1 window.
    task = DGEBECClassificationTask(TINY_MANIFEST, max_context=40, overflow_strategy="window")
    result = task.evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert len(adapter.calls) == 1
    assert adapter.calls[0] > 18  # some sequences windowed into >1 chunk
    assert result.metrics["n_train"] == 12.0 and result.metrics["n_test"] == 6.0  # none dropped
    assert result.metadata["n_train_overflow"] + result.metadata["n_test_overflow"] > 0


def test_truncate_strategy_one_window_per_instance() -> None:
    adapter = _CountingEmbed(_CompositionAdapter())
    task = DGEBECClassificationTask(TINY_MANIFEST, max_context=40, overflow_strategy="truncate")
    result = task.evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert len(adapter.calls) == 1
    assert adapter.calls[0] == 18  # truncate → exactly one window per instance
    assert result.metadata["overflow_strategy"] == "truncate"
    assert result.metadata["n_train_overflow"] + result.metadata["n_test_overflow"] > 0


def test_invalid_overflow_strategy_raises() -> None:
    with pytest.raises(ValueError, match="overflow_strategy"):
        DGEBECClassificationTask(TINY_MANIFEST, overflow_strategy="nonsense")


# --- standard-scale toggle ----------------------------------------------------


def test_standard_scale_toggle_default_off_and_on() -> None:
    off = DGEBECClassificationTask(TINY_MANIFEST).evaluate(_CompositionAdapter())
    on = DGEBECClassificationTask(TINY_MANIFEST, standard_scale=True).evaluate(_CompositionAdapter())
    assert off.metadata["standard_scale"] is False
    assert on.metadata["standard_scale"] is True
    assert off.status is ResultStatus.OK and on.status is ResultStatus.OK


def test_scale_env_seam(monkeypatch) -> None:
    monkeypatch.setenv("GLMBENCH_DGEB_EC_SCALE", "1")
    task = DGEBECClassificationTask(TINY_MANIFEST)
    assert task.standard_scale is True


# --- configurable layer -------------------------------------------------------


def test_layer_default_is_last() -> None:
    task = DGEBECClassificationTask(TINY_MANIFEST)
    assert task.layer == "last"
    result = task.evaluate(_CompositionAdapter(n_layers=3))
    assert result.status is ResultStatus.OK
    assert result.metadata["layer"] == 3  # last layer of a 3-block model = id 3


def test_layer_explicit_int() -> None:
    adapter = _CompositionAdapter(n_layers=3)
    result = DGEBECClassificationTask(TINY_MANIFEST, layer=1).evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert result.metadata["layer"] == 1


def test_layer_env_seam(monkeypatch) -> None:
    monkeypatch.setenv("GLMBENCH_DGEB_EC_LAYER", "2")
    task = DGEBECClassificationTask(TINY_MANIFEST)
    assert task.layer == 2


# --- layer sweep --------------------------------------------------------------


def test_parse_layer_list() -> None:
    assert parse_layer_list("0,4,8") == [0, 4, 8]
    assert parse_layer_list("6-12") == [6, 7, 8, 9, 10, 11, 12]
    assert parse_layer_list("all") is None
    assert parse_layer_list(None) is None


def test_layer_sweep_finds_best_layer_one_call() -> None:
    adapter = _CompositionAdapter(n_layers=6, good_layers=frozenset({5, 6}))
    result = DGEBECClassificationLayerSweepTask(TINY_MANIFEST).evaluate(adapter)
    assert result.status is ResultStatus.OK
    assert result.task == "dgeb-ec-classification-dna-layer-sweep"
    assert result.primary_metric == "f1"
    assert len(adapter.embed_calls) == 1  # the GPU work happens ONCE
    # every layer 0..6 reported; headline f1 == best layer's f1
    for layer in range(7):
        assert f"layer_{layer}_f1" in result.metrics
    assert result.metadata["swept_layers"] == list(range(7))
    best = int(result.metrics["best_layer"])
    assert best in {5, 6}
    assert result.metrics["f1"] == pytest.approx(result.metrics[f"layer_{best}_f1"])
    assert result.metrics[f"layer_{best}_f1"] > 0.9


def test_layer_sweep_explicit_range(monkeypatch) -> None:
    monkeypatch.setenv("GLMBENCH_DGEB_EC_SWEEP_LAYERS", "2-4")
    task = DGEBECClassificationLayerSweepTask(TINY_MANIFEST)
    assert task.sweep_layers == [2, 3, 4]


# --- ERROR path: single-class train ------------------------------------------


def test_single_class_train_is_error(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        {
            "train": [("ATGAAAA", "X"), ("ATGCCCC", "X")],  # one class only → undefined probe
            "test": [("ATGAAAA", "X"), ("ATGGGGG", "Y")],
        },
    )
    result = DGEBECClassificationTask(manifest, verify_hashes=False).evaluate(_CompositionAdapter())
    assert result.status is ResultStatus.ERROR
    assert "2 EC classes" in result.metadata["error"]


# --- integrity gate bites through prepare() (outside evaluate's try) ----------


def test_verify_hashes_trips_on_tampered_fixture(tmp_path: Path) -> None:
    import shutil

    work = tmp_path / "fixture"
    shutil.copytree(TINY_MANIFEST.parent, work)
    target = work / "tiny_test.csv"
    target.write_text(target.read_text().replace("ATG", "ATC", 1))
    with pytest.raises(ManifestHashError, match="sha256 drift"):
        DGEBECClassificationTask(work / "tiny_manifest.yaml").evaluate(_CompositionAdapter())


# --- vendored DGEB evaluator is byte-pinned (sha256) --------------------------


def test_vendored_evaluator_sha256_pin() -> None:
    sums = (VENDOR / "SHA256SUMS").read_text().split()[0]
    actual = hashlib.sha256((VENDOR / "evaluators.py").read_bytes()).hexdigest()
    assert actual == sums, "vendored dgeb/evaluators.py drifted from its pinned sha256"


# --- metric equivalence: wrapper == calling the vendored evaluator directly ----


def test_metric_equivalence_with_vendored_evaluator() -> None:
    """The thin wrapper's output equals calling the vendored DGEB evaluator directly."""
    from glmbench.tasks._vendor.dgeb.evaluators import logRegClassificationEvaluator

    rng = np.random.default_rng(0)
    # 3-class, linearly separable blobs so the probe is well-defined.
    centers = {0: [3, 0], 1: [0, 3], 2: [-3, -3]}
    Xtr, ytr, Xte, yte = [], [], [], []
    for c, ctr in centers.items():
        for _ in range(8):
            Xtr.append(np.array(ctr) + rng.standard_normal(2) * 0.3)
            ytr.append(str(c))
        for _ in range(4):
            Xte.append(np.array(ctr) + rng.standard_normal(2) * 0.3)
            yte.append(str(c))
    Xtr, Xte = np.array(Xtr), np.array(Xte)

    wrapped = classification_scores(Xtr, ytr, Xte, yte, max_iter=1000, seed=42)
    direct = logRegClassificationEvaluator(Xtr, ytr, Xte, yte, max_iter=1000, seed=42)()
    assert wrapped["f1"] == pytest.approx(float(direct["f1"]))
    assert wrapped["accuracy"] == pytest.approx(float(direct["accuracy"]))
    # 128-class EC is never binary, so the wrapper never emits 'ap' for the real task;
    # here (3 classes) it also must not.
    assert "ap" not in wrapped


# --- chunking recipe + loader -------------------------------------------------


def test_chunk_dna_sequence() -> None:
    chunks = chunk_dna_sequence("A" * 250, max_seq_len=100, overlap=32)
    assert [len(c) for c in chunks] == [100, 100, 100, 46]
    assert chunk_dna_sequence("ACGT", 100, 32) == ["ACGT"]


def test_loader_contract_and_splits() -> None:
    splits = load_ec_splits(TINY_MANIFEST)
    assert set(splits) == {"train", "test"}
    assert len(splits["train"]) == 12
    assert splits["train"].n_classes == 3
    # U→T normalization applied in the loader
    assert all("U" not in s for s in splits["train"].sequences)


def test_loader_missing_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("entry,sequence\nE1,ATGAAA\n")
    manifest = tmp_path / "m.yaml"
    manifest.write_text("splits:\n  - name: train\n    file: bad.csv\n    sha256: null\n")
    with pytest.raises(ECClassificationContractError, match="missing required column"):
        load_ec_splits(manifest, splits=("train",))


# --- real-data path (skips on CI where the gitignored CSVs are absent) ---------

_REAL_TRAIN = (
    REPO / "src" / "glmbench" / "tasks" / "data" / "dgeb_ec_dna"
    / "ec_classification_dna_train.csv"
)


@pytest.mark.skipif(
    not _REAL_TRAIN.exists(),
    reason="vendored DGEB EC CSVs absent (gitignored cache); run scripts/fetch_dgeb_ec_dna.py",
)
def test_real_data_loads() -> None:
    from glmbench.tasks.dgeb_ec_data import DEFAULT_MANIFEST

    splits = load_ec_splits(DEFAULT_MANIFEST)
    assert len(splits["train"]) == 512
    assert len(splits["test"]) == 128
