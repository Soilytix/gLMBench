"""End-to-end leaderboard + the extensibility stress test.

The gate has two halves:

1. **Combined leaderboard renders correctly.** Exercised here with three synthetic
   score-only models plus an ``N/A`` cell, asserting the Overall + per-task Markdown
   tables come out well-formed and sorted. (A real board comes from running real specs
   on a GPU; this test pins the rendering contract with no GPU.)

2. **The extensibility stress test (the headline gate):** adding a no-op throwaway task
   and a no-op throwaway adapter — each via **only its own module + one registry line** —
   makes them appear in ``glmbench models/tasks list`` and land a row on the board
   **through the unchanged runner** (:func:`glmbench.run.run_benchmark`). Nothing in
   ``run.py``, the adapters, the tasks, or the leaderboard is touched: composition over
   inheritance, proven mechanically.

This whole module is core (torch-free): the no-op adapter is model-free, so the proof
needs no subprocess/container.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glmbench import cli, registry
from glmbench.adapters.base import Capability, ModelAdapter
from glmbench.config.benchmark_spec import BenchmarkSpec
from glmbench.config.model_spec import ModelSpec
from glmbench.leaderboard.render import aggregate_dataframe, render_markdown
from glmbench.leaderboard.store import read_results
from glmbench.run import run_benchmark
from glmbench.tasks.base import ResultStatus, Task, TaskResult

# ---------------------------------------------------------------------------
# A no-op THROWAWAY adapter — "drop one module + one registry line", nothing else.
# Model-free (deterministic scores from a hash of the sequence) so the extensibility
# proof stays torch-free and needs no runner subprocess.
# ---------------------------------------------------------------------------
_NOOP_ADAPTER = "probe-noop-adapter"
_NOOP_TASK = "probe-noop-task"


@registry.register("adapter", _NOOP_ADAPTER)
class _NoOpAdapter(ModelAdapter):
    name = _NOOP_ADAPTER
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def __init__(self, digest: str = "noop") -> None:
        self._digest = digest

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> _NoOpAdapter:
        return cls(digest=str(spec.model.get("digest", "noop")))

    def model_hash(self) -> str:
        # Stable + weight-sensitive, computable with no model.
        import hashlib

        h = hashlib.sha256(f"{self._digest}|{self.name}@{self.adapter_version}".encode())
        return "glmb:" + h.hexdigest()[:24]

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "adapter_version": self.adapter_version, "digest": self._digest}

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        # Deterministic pseudo-scores: length + a stable per-string hash. No model.
        out = []
        for s in sequences:
            hv = (hash((self._digest, s)) % 1000) / 1000.0
            out.append(-float(len(s)) - hv)
        return out


# ---------------------------------------------------------------------------
# A no-op THROWAWAY task — same "one module + one registry line" deal.
# ---------------------------------------------------------------------------
@registry.register("task", _NOOP_TASK)
class _NoOpTask(Task):
    name = _NOOP_TASK
    version = "0.1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    _SEQS = ["ACGTACGT", "TTTTAAAA", "GGGGCCCCGG"]

    def __init__(self) -> None:
        self._scores: list[float] = []

    def prepare(self) -> None:
        self._seqs = list(self._SEQS)

    def run(self, adapter: ModelAdapter) -> None:
        self._scores = adapter.score_sequences(self._seqs, reduction="mean")  # one batched call

    def score(self) -> TaskResult:
        scores = np.asarray(self._scores, dtype=float)
        if scores.size == 0 or np.isnan(scores).any():
            raise ValueError("no-op task got no/NaN scores")
        return TaskResult(
            self.name,
            ResultStatus.OK,
            "noop_metric",
            {"noop_metric": float(scores.mean()), "n_scored": float(scores.size)},
            {"n_scored": int(scores.size)},
        )


def _model_spec(adapter: str, digest: str) -> ModelSpec:
    return ModelSpec.model_validate(
        {"adapter": adapter, "adapter_version": "0.1.0", "model": {"digest": digest}}
    )


# ---------------------------------------------------------------------------
# Extensibility gate (the headline gate)
# ---------------------------------------------------------------------------
def test_noop_adapter_and_task_are_registered() -> None:
    """The throwaway components are discoverable purely via their registry line."""
    assert _NOOP_ADAPTER in registry.list("adapter")
    assert _NOOP_TASK in registry.list("task")


def test_glmbench_lists_show_the_noop_components(capsys: pytest.CaptureFixture[str]) -> None:
    """`glmbench models list` / `glmbench tasks list` surface the new components."""
    assert cli.main(["models", "list"]) == 0
    models_out = capsys.readouterr().out
    assert _NOOP_ADAPTER in models_out
    assert "sequence_loglikelihood" in models_out  # its declared capability

    assert cli.main(["tasks", "list"]) == 0
    tasks_out = capsys.readouterr().out
    assert _NOOP_TASK in tasks_out


def test_noop_components_land_on_the_board_through_the_unchanged_runner(tmp_path: Path) -> None:
    """The gate: a brand-new adapter + task run through the existing run_benchmark and
    render onto the board — no edits to run.py, the adapters, the tasks, or the board."""
    bench = BenchmarkSpec(name="probe_bench", version="1.0", tasks=[_NOOP_TASK])

    record = run_benchmark(
        _model_spec(_NOOP_ADAPTER, "alpha"),
        bench,
        results_dir=tmp_path,
    )
    assert record["adapter_name"] == _NOOP_ADAPTER
    assert [t["status"] for t in record["tasks"]] == ["ok"]

    # Re-read from disk (idempotent store) + render exactly as the CLI leaderboard does.
    records = read_results(tmp_path, "probe_bench")
    md = render_markdown(
        records, benchmark_name="probe_bench", benchmark_version="1.0", task_order=[_NOOP_TASK]
    )
    assert _NOOP_TASK in md  # the new task is a rendered section
    assert "noop_metric" in md  # its metric column
    assert record["model_hash"][: len("glmb:")] == "glmb:"
    # The model's short-hash label appears in the Overall table.
    assert record["model_hash"].replace("glmb:", "")[:10] in md


def test_combined_multi_model_board_renders_sorted_with_na_cell(tmp_path: Path) -> None:
    """Half 1 of the gate: a combined multi-model board (3 synthetic models) renders
    Overall + per-task, sorted, with an N/A."""
    bench = BenchmarkSpec(name="probe_combo", version="1.0", tasks=[_NOOP_TASK, "dummy-embed"])

    # Three score-capable models (distinct digests → distinct hashes/rows).
    for digest in ("model-a", "model-b", "model-c"):
        run_benchmark(_model_spec(_NOOP_ADAPTER, digest), bench, results_dir=tmp_path)

    records = read_results(tmp_path, "probe_combo")
    assert len(records) == 3

    df = aggregate_dataframe(records)
    # Tidy CSV: one row per model × task.
    assert len(df) == 3 * 2

    md = render_markdown(
        records,
        benchmark_name="probe_combo",
        benchmark_version="1.0",
        task_order=[_NOOP_TASK, "dummy-embed"],
    )
    # The no-op task scores OK for all three; the embedding task is N/A (score-only adapter).
    assert "## Overall" in md
    assert _NOOP_TASK in md and "dummy-embed" in md
    assert "N/A" in md  # the embedding task against a score-only adapter
    # Overall table lists all three models.
    for digest in ("model-a", "model-b", "model-c"):
        h = _NoOpAdapter(digest=digest).model_hash().replace("glmb:", "")[:10]
        assert h in md
