"""The `loam_paper` benchmark end to end, and what makes a run's identity and record safe.

1. ``benchmarks/loam_paper.yaml`` is exactly the paper's five task rows, and every scoring
   task the package registers is one of them, so a task added later has to be placed on
   purpose rather than forgotten.
2. The five rows run end to end on CPU: the echo double over the committed tiny fixtures
   (``tests/data/*_tiny``). Every row is OK; each corpus costs ONE forward pass (the
   single-layer rows are read from their sweep's pass); the record carries each row's
   data-manifest hash and no host name; a re-run resumes instead of recomputing.
3. The optional ``display_name`` is **provenance only** — it is echoed into the record and
   rendered on the board, but does NOT enter the model hash, so renaming never changes identity.
4. The leaderboard keeps two same-named checkpoints with **different weights** on distinct
   rows (different content hash → different file, different row), and warns loudly when the
   weights digest is not content-addressed (the only way different weights could collide).
5. Results are written after every task, and a re-run resumes from them.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import glmbench.tasks  # noqa: F401  (registers the shipped tasks)
from glmbench import registry
from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter
from glmbench.adapters.echo import EchoAdapter
from glmbench.config.benchmark_spec import (
    BenchmarkSpec,
    builtin_benchmarks_dir,
    load_benchmark_spec,
    resolve_benchmark_spec,
)
from glmbench.config.model_spec import ModelSpec
from glmbench.leaderboard.render import aggregate_dataframe, render_markdown
from glmbench.leaderboard.store import read_result, read_results, result_filename
from glmbench.run import run_benchmark, warn_on_weak_weights_digest
from glmbench.tasks.base import ResultStatus, Task, TaskResult

PAPER_ROWS = [
    "bacbench-essentiality",
    "bacbench-essentiality-layer-sweep",
    "dgeb-ec-classification-dna",
    "dgeb-ec-classification-dna-layer-sweep",
    "rnagym-dms",
]
_DUMMY_TASKS = {"dummy", "dummy-embed"}

DATA = Path(__file__).resolve().parents[1] / "data"
_BACBENCH_TINY = DATA / "bacbench_essentiality_tiny" / "tiny_manifest.yaml"
_DGEB_EC_TINY = DATA / "dgeb_ec_dna_tiny" / "tiny_manifest.yaml"
_RNAGYM_TINY = DATA / "rnagym_tiny" / "tiny_manifest.yaml"
#: The manifest each paper row reads in the end-to-end run below.
TINY_MANIFESTS = {
    "bacbench-essentiality": _BACBENCH_TINY,
    "bacbench-essentiality-layer-sweep": _BACBENCH_TINY,
    "dgeb-ec-classification-dna": _DGEB_EC_TINY,
    "dgeb-ec-classification-dna-layer-sweep": _DGEB_EC_TINY,
    "rnagym-dms": _RNAGYM_TINY,
}


# --- 1. loam_paper is the paper's five rows, and nothing shipped is left out ---------------
def _shipped_tasks() -> set[str]:
    """Scoring tasks the package itself registers (test modules register throwaways too)."""
    return {
        name
        for name in registry.list("task")
        if getattr(registry.resolve("task", name), "__module__", "").startswith("glmbench.tasks.")
    } - _DUMMY_TASKS


def test_loam_paper_is_exactly_the_five_paper_rows() -> None:
    spec = resolve_benchmark_spec("loam_paper")
    assert spec.name == "loam_paper"
    assert spec.tasks == PAPER_ROWS
    for task_name in spec.tasks:
        assert registry.build("task", task_name) is not None


def test_every_shipped_task_is_a_paper_row() -> None:
    """A registered task that no paper row uses is either unfinished or unreported."""
    assert _shipped_tasks() == set(PAPER_ROWS)


def test_the_single_purpose_benchmarks_are_slices_of_loam_paper() -> None:
    for yaml_path in builtin_benchmarks_dir().glob("*.yaml"):
        spec = load_benchmark_spec(yaml_path)
        if spec.name == "dummy_bench":
            assert set(spec.tasks) == _DUMMY_TASKS
            continue
        assert set(spec.tasks) <= set(PAPER_ROWS), f"{yaml_path.name}: {spec.tasks}"
        assert len(spec.tasks) == len(set(spec.tasks)), f"duplicate task in {yaml_path.name}"


# --- 2. the five rows end to end, on the tiny fixtures --------------------------------------
_COUNT_ADAPTER = "fulltest-counting-echo"


@registry.register("adapter", _COUNT_ADAPTER)
class _CountingEcho(EchoAdapter):
    """The echo double, logging every forward pass a benchmark asks it for."""

    name = _COUNT_ADAPTER
    calls: list[tuple] = []

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        type(self).calls.append(("embed", len(sequences), layers))
        return super().embed(sequences, layers=layers, pool=pool)

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        type(self).calls.append(("score_sequences", len(sequences), reduction))
        return super().score_sequences(sequences, reduction=reduction)


def _count_spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {"adapter": _COUNT_ADAPTER, "adapter_version": EchoAdapter.adapter_version}
    )


@contextlib.contextmanager
def _tiny_paper_data() -> Iterator[None]:
    """Point the five rows at the committed tiny fixtures instead of the fetched corpora."""
    with pytest.MonkeyPatch.context() as mp:
        for var in list(os.environ):
            if var.startswith(("GLMBENCH_BACBENCH_", "GLMBENCH_DGEB_EC_")):
                mp.delenv(var)  # a stray layer/pool setting would change what the rows measure
        mp.setenv("GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST", str(_BACBENCH_TINY))
        mp.setenv("GLMBENCH_DGEB_EC_MANIFEST", str(_DGEB_EC_TINY))
        # rnagym-dms reads no manifest variable: bind its registered factory to the tiny one.
        real = registry.resolve("task", "rnagym-dms")
        registry.unregister("task", "rnagym-dms")
        registry.register("task", "rnagym-dms")(functools.partial(real, _RNAGYM_TINY))
        try:
            yield
        finally:
            registry.unregister("task", "rnagym-dms")
            registry.register("task", "rnagym-dms")(real)


@pytest.fixture(scope="module")
def paper_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, list[tuple], Path]:
    """ONE loam_paper run, shared by the tests below: (record, forward calls, results dir)."""
    results = tmp_path_factory.mktemp("loam_paper")
    _CountingEcho.calls.clear()
    with _tiny_paper_data():
        record = run_benchmark(
            _count_spec(), resolve_benchmark_spec("loam_paper"), results_dir=results
        )
    return record, list(_CountingEcho.calls), results


def test_every_paper_row_runs_ok(paper_run) -> None:
    record, _, _ = paper_run
    assert [t["task"] for t in record["tasks"]] == PAPER_ROWS
    for t in record["tasks"]:
        assert t["status"] == "ok", (t["task"], t["metadata"].get("error"))
        assert math.isfinite(t["metrics"][t["primary_metric"]]), t["task"]


def test_one_forward_pass_per_corpus(paper_run) -> None:
    """Five rows, three corpora, three forward passes: the single-layer rows cost nothing.

    Each single-layer row shares a fusion group with its sweep, so its corpus is embedded
    once, at every layer, and the row reads its layer from that pass.
    """
    _, calls, _ = paper_run
    assert sorted(c[0] for c in calls) == ["embed", "embed", "score_sequences"], calls
    assert all(c[2] == "all" for c in calls if c[0] == "embed"), calls


@pytest.mark.parametrize(
    ("row", "sweep", "metrics"),
    [
        ("bacbench-essentiality", "bacbench-essentiality-layer-sweep",
         ("macro_mean_auroc", "macro_mean_auprc")),
        ("dgeb-ec-classification-dna", "dgeb-ec-classification-dna-layer-sweep",
         ("f1", "accuracy")),
    ],
)
def test_a_single_layer_row_equals_its_sweep_at_the_last_layer(
    paper_run, row, sweep, metrics
) -> None:
    """Read from the same pass, the last-layer row and the sweep's last tap are one number."""
    record, _, _ = paper_run
    by_task = {t["task"]: t for t in record["tasks"]}
    last = max(by_task[sweep]["metadata"]["swept_layers"])
    for m in metrics:
        assert by_task[row]["metrics"][m] == by_task[sweep]["metrics"][f"layer_{last}_{m}"], m


def test_record_names_the_data_each_row_read_and_not_the_machine(paper_run) -> None:
    record, _, _ = paper_run
    provenance = record["provenance"]
    assert provenance["data_manifest_hashes"] == {
        task: hashlib.sha256(path.read_bytes()).hexdigest()
        for task, path in TINY_MANIFESTS.items()
    }
    assert "host" not in provenance


def test_the_board_renders_every_paper_row(paper_run) -> None:
    record, _, results = paper_run
    records = read_results(results, "loam_paper")
    assert [r["model_hash"] for r in records] == [record["model_hash"]]
    md = render_markdown(
        records, benchmark_name="loam_paper", benchmark_version="1.0", task_order=PAPER_ROWS
    )
    for task_name in PAPER_ROWS:
        assert f"## Task: {task_name}" in md


def test_a_rerun_resumes_every_row_without_a_forward_pass(
    paper_run, caplog: pytest.LogCaptureFixture
) -> None:
    record, _, results = paper_run
    _CountingEcho.calls.clear()
    with _tiny_paper_data(), caplog.at_level(logging.INFO):
        again = run_benchmark(
            _count_spec(), resolve_benchmark_spec("loam_paper"), results_dir=results
        )
    assert caplog.text.count("SKIP (resumed") == len(PAPER_ROWS)
    assert _CountingEcho.calls == []
    assert [t["metrics"] for t in again["tasks"]] == [t["metrics"] for t in record["tasks"]]


# --- 3. display_name is provenance only, not part of the hash ---------------
def _echo_spec(display_name: str | None, weights_value: str) -> ModelSpec:
    payload: dict = {
        "adapter": "echo",
        "adapter_version": EchoAdapter.adapter_version,
        "model": {"weights_value": weights_value},
    }
    if display_name is not None:
        payload["display_name"] = display_name
    return ModelSpec.model_validate(payload)


def test_display_name_does_not_change_model_hash() -> None:
    """Renaming a model (or two models sharing a name) never moves its identity."""
    no_name = EchoAdapter.from_spec(_echo_spec(None, "weights-A"))
    named = EchoAdapter.from_spec(_echo_spec("My LOAM @ step50k", "weights-A"))
    renamed = EchoAdapter.from_spec(_echo_spec("totally different label", "weights-A"))
    assert no_name.model_hash() == named.model_hash() == renamed.model_hash()


def test_display_name_is_echoed_into_provenance() -> None:
    spec = _echo_spec("Friendly Name", "weights-A")
    assert spec.echo()["display_name"] == "Friendly Name"
    # And it is absent from the hashed model config (lives at the top level only).
    assert "display_name" not in spec.model


def test_different_weights_same_name_get_different_hashes() -> None:
    a = EchoAdapter.from_spec(_echo_spec("same name", "weights-A"))
    b = EchoAdapter.from_spec(_echo_spec("same name", "weights-B"))
    assert a.model_hash() != b.model_hash()


# --- 4. leaderboard identity: same name, different weights → distinct rows ---
_ID_ADAPTER = "fulltest-id-adapter"


@registry.register("adapter", _ID_ADAPTER)
class _IdAdapter(ModelAdapter):
    """Model-free score adapter whose hash tracks a `weights` knob (no torch, no subprocess)."""

    name = _ID_ADAPTER
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def __init__(self, weights: str) -> None:
        self._weights = weights

    @classmethod
    def from_spec(cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False):
        return cls(weights=str(spec.model.get("weights", "w")))

    def model_hash(self) -> str:
        import hashlib

        h = hashlib.sha256(f"{self._weights}|{self.name}@{self.adapter_version}".encode())
        return "glmb:" + h.hexdigest()[:24]

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "adapter_version": self.adapter_version}

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        return [-float(len(s)) for s in sequences]


_ID_TASK = "fulltest-id-task"


@registry.register("task", _ID_TASK)
class _IdTask(Task):
    name = _ID_TASK
    version = "0.1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def prepare(self) -> None:
        self._seqs = ["ACGT", "TTGGCC"]

    def run(self, adapter: ModelAdapter) -> None:
        self._scores = adapter.score_sequences(self._seqs)

    def score(self) -> TaskResult:
        m = float(np.mean(self._scores))
        return TaskResult(self.name, ResultStatus.OK, "score", {"score": m}, {})


def _id_spec(display_name: str, weights: str) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "adapter": _ID_ADAPTER,
            "adapter_version": "0.1.0",
            "display_name": display_name,
            "model": {"weights": weights},
        }
    )


def test_board_keeps_same_named_different_weight_models_distinct(tmp_path: Path) -> None:
    bench = BenchmarkSpec(name="idprobe", version="1.0", tasks=[_ID_TASK])
    # Two models, IDENTICAL display name, DIFFERENT weights.
    rec_a = run_benchmark(_id_spec("Loam-100M", "ckpt-step-1000"), bench, results_dir=tmp_path)
    rec_b = run_benchmark(_id_spec("Loam-100M", "ckpt-step-9000"), bench, results_dir=tmp_path)
    assert rec_a["model_hash"] != rec_b["model_hash"]

    records = read_results(tmp_path, "idprobe")
    assert len(records) == 2, "different weights must NOT overwrite each other"

    # Tidy CSV: two distinct hashes, both carrying the shared display_name.
    df = aggregate_dataframe(records)
    assert set(df["model_hash"]) == {rec_a["model_hash"], rec_b["model_hash"]}
    assert set(df["display_name"]) == {"Loam-100M"}

    md = render_markdown(
        records, benchmark_name="idprobe", benchmark_version="1.0", task_order=[_ID_TASK]
    )
    assert "Model ID (hash)" in md
    # Both FULL hashes are present (unambiguous identity) and the friendly name renders.
    assert rec_a["model_hash"] in md and rec_b["model_hash"] in md
    assert "Loam-100M" in md


# --- 4b. weak-weights-digest warning ----------------------------------------
def test_file_sha256_digest_is_not_flagged() -> None:
    spec = ModelSpec.model_validate(
        {"adapter": "echo", "adapter_version": "0.1.0", "weights_digest": {"strategy": "file_sha256"}}
    )
    assert warn_on_weak_weights_digest(spec) is None


def test_declared_digest_warns(caplog: pytest.LogCaptureFixture) -> None:
    spec = ModelSpec.model_validate(
        {
            "adapter": "echo",
            "adapter_version": "0.1.0",
            "weights_digest": {"strategy": "declared", "value": "hand-typed"},
        }
    )
    with caplog.at_level(logging.WARNING):
        msg = warn_on_weak_weights_digest(spec)
    assert msg is not None and "declared" in msg
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("ref", ["main", "master", "HEAD", "latest", ""])
def test_mutable_hf_revision_warns(ref: str) -> None:
    spec = ModelSpec.model_validate(
        {
            "adapter": "echo",
            "adapter_version": "0.1.0",
            "weights_digest": {"strategy": "hf_revision", "value": ref},
        }
    )
    assert warn_on_weak_weights_digest(spec) is not None


def test_immutable_hf_revision_commit_is_not_flagged() -> None:
    spec = ModelSpec.model_validate(
        {
            "adapter": "echo",
            "adapter_version": "0.1.0",
            "weights_digest": {"strategy": "hf_revision", "value": "a1b2c3d4e5f6"},
        }
    )
    assert warn_on_weak_weights_digest(spec) is None


# --- 5. incremental writes (crash survival) + resume ------------------------
_CRASH_TASK = "fulltest-crash-on-build-task"


@registry.register("task", _CRASH_TASK)
class _CrashOnBuildTask(Task):
    """A task that explodes at construction — simulates a hard, uncaught mid-run crash."""

    name = _CRASH_TASK
    version = "0.1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def __init__(self) -> None:
        raise RuntimeError("boom: task could not be built")

    def prepare(self) -> None: ...  # pragma: no cover - never reached
    def run(self, adapter: ModelAdapter) -> None: ...  # pragma: no cover
    def score(self) -> TaskResult:  # pragma: no cover
        raise NotImplementedError


def test_incremental_write_persists_completed_tasks_before_a_crash(tmp_path: Path) -> None:
    """A hard crash on task 2 must NOT discard the OK result already written for task 1."""
    bench = BenchmarkSpec(name="crashprobe", version="1.0", tasks=[_ID_TASK, _CRASH_TASK])
    spec = _id_spec("Crashy", "ckpt-1")
    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark(spec, bench, results_dir=tmp_path)

    # The partial record on disk holds the completed first task.
    rec = read_result(tmp_path, "crashprobe", _IdAdapter(weights="ckpt-1").model_hash())
    assert rec is not None
    assert [t["task"] for t in rec["tasks"]] == [_ID_TASK]
    assert rec["tasks"][0]["status"] == "ok"


def test_resume_skips_completed_ok_tasks(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bench = BenchmarkSpec(name="resumeprobe", version="1.0", tasks=[_ID_TASK, "dummy"])
    spec = _id_spec("Resumable", "ckpt-1")
    first = run_benchmark(spec, bench, results_dir=tmp_path)
    assert [t["status"] for t in first["tasks"]] == ["ok", "ok"]

    with caplog.at_level(logging.INFO):
        again = run_benchmark(spec, bench, results_dir=tmp_path)
    # Both OK tasks are skipped via resume; the record is still complete + identical.
    assert caplog.text.count("SKIP (resumed") == 2
    assert [t["task"] for t in again["tasks"]] == [_ID_TASK, "dummy"]
    assert again["model_hash"] == first["model_hash"]


def test_resume_persists_trailing_ok_tasks_after_a_middle_pending_task(tmp_path: Path) -> None:
    """REGRESSION: a resume whose only pending task sits BEFORE already-OK tasks must not drop
    those trailing OK tasks from the ON-DISK record.

    The in-loop write fires only for computed tasks — resumed SKIPs hit `continue` and never
    write. So if the last *pending* task is not the last task in the benchmark, the final
    in-loop write predates the trailing SKIPs, and without a final full write the persisted
    record silently loses them: the already-OK tasks after the re-run one vanish from disk, and
    the board rebuilds them as N/A. `test_resume_skips_completed_ok_tasks` cannot see it because
    it checks run()'s RETURN (always complete); the corruption lives only on DISK.
    """
    spec = _id_spec("MidResume", "ckpt-1")
    # First: a 2-task board, both OK, persisted to disk.
    run_benchmark(
        spec, BenchmarkSpec(name="midprobe", version="1.0", tasks=[_ID_TASK, "dummy"]),
        results_dir=tmp_path,
    )
    # Then a NEW middle task appears (a benchmark gaining a task) — same board name +
    # model hash, so resume seeds the two prior OK tasks and only the middle one is pending.
    run_benchmark(
        spec,
        BenchmarkSpec(name="midprobe", version="1.0", tasks=[_ID_TASK, "dummy-embed", "dummy"]),
        results_dir=tmp_path,
    )

    rec = read_result(tmp_path, "midprobe", _IdAdapter(weights="ckpt-1").model_hash())
    assert rec is not None
    # The trailing already-OK task ("dummy") MUST survive on disk — not just in the return value.
    assert [t["task"] for t in rec["tasks"]] == [_ID_TASK, "dummy-embed", "dummy"]
    statuses = {t["task"]: t["status"] for t in rec["tasks"]}
    assert statuses[_ID_TASK] == "ok" and statuses["dummy"] == "ok"


def test_resume_reruns_non_ok_tasks(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # dummy-embed needs EMBEDDING, which _IdAdapter lacks → N/A; N/A must NOT be resumed away.
    bench = BenchmarkSpec(name="resumena", version="1.0", tasks=[_ID_TASK, "dummy-embed"])
    spec = _id_spec("Partial", "ckpt-1")
    run_benchmark(spec, bench, results_dir=tmp_path)

    with caplog.at_level(logging.INFO):
        again = run_benchmark(spec, bench, results_dir=tmp_path)
    statuses = {t["task"]: t["status"] for t in again["tasks"]}
    assert statuses[_ID_TASK] == "ok" and statuses["dummy-embed"] == "na"
    # Only the OK task is skipped; the N/A task is recomputed (no skip line for it).
    assert caplog.text.count("SKIP (resumed") == 1
    assert "dummy-embed: running" in caplog.text


def test_no_resume_recomputes_everything(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    bench = BenchmarkSpec(name="noresume", version="1.0", tasks=[_ID_TASK, "dummy"])
    spec = _id_spec("Fresh", "ckpt-1")
    run_benchmark(spec, bench, results_dir=tmp_path)
    with caplog.at_level(logging.INFO):
        run_benchmark(spec, bench, results_dir=tmp_path, resume=False)
    assert "SKIP (resumed" not in caplog.text


def test_empty_prior_result_is_ignored_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0-byte / corrupt prior result JSON (disk-full truncation) must warn, not crash.

    Regression: a bare ``json.loads`` on an empty result file aborted the whole suite in the
    resume path before any task ran. It must be treated as "no usable prior" (loud warning) so
    a re-run recomputes and overwrites it.
    """
    model_hash = _IdAdapter(weights="ckpt-1").model_hash()
    bench_dir = tmp_path / "corruptprobe"
    bench_dir.mkdir()
    (bench_dir / result_filename(model_hash)).write_text("")  # 0-byte, as a disk-full crash leaves
    (bench_dir / result_filename("glmb:other")).write_text("{ truncated")  # partial flush

    with caplog.at_level(logging.WARNING):
        assert read_result(tmp_path, "corruptprobe", model_hash) is None  # not-fatal → None
        assert read_results(tmp_path, "corruptprobe") == []  # both skipped, no crash
    assert caplog.text.count("ignoring unreadable result JSON") >= 2

    # And the full run proceeds over the empty prior, overwriting it with a valid record.
    spec = _id_spec("Recovered", "ckpt-1")
    rec = run_benchmark(spec, BenchmarkSpec(name="corruptprobe", version="1.0", tasks=[_ID_TASK]),
                        results_dir=tmp_path)
    assert rec["tasks"][0]["status"] == "ok"
    assert read_result(tmp_path, "corruptprobe", model_hash) is not None
