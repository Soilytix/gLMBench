"""Task lifecycle, capability negotiation, leaderboard store/render, CLI.

Gate: ``glmbench run`` over a 2-task dummy benchmark with one capable + one incapable
adapter produces a correct Markdown leaderboard with an ``N/A`` cell.
All model-free (the EchoAdapter shells out to the deterministic echo runner) — no torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from glmbench import registry
from glmbench.adapters.base import Capability, ModelAdapter
from glmbench.adapters.echo import EchoAdapter
from glmbench.config.benchmark_spec import (
    BenchmarkSpec,
    load_benchmark_spec,
    resolve_benchmark_spec,
)
from glmbench.config.model_spec import ModelSpec, load_model_spec
from glmbench.leaderboard.render import (
    aggregate_dataframe,
    render_markdown,
    write_leaderboard,
)
from glmbench.leaderboard.store import read_results, result_filename
from glmbench.run import build_adapter, run_benchmark
from glmbench.tasks import dummy as _dummy  # noqa: F401  (registers dummy tasks)
from glmbench.tasks.base import ResultStatus, Task, TaskResult

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "specs"


@pytest.fixture
def full_adapter(tmp_path) -> EchoAdapter:
    return EchoAdapter(python_exe=sys.executable, scratch_dir=str(tmp_path / "full"))


@pytest.fixture
def score_only_adapter(tmp_path) -> EchoAdapter:
    return EchoAdapter(
        capabilities={Capability.SEQUENCE_LOGLIKELIHOOD},
        python_exe=sys.executable,
        scratch_dir=str(tmp_path / "scoreonly"),
        weights_value="score-only",
    )


# --- task lifecycle + capability negotiation -------------------------------


def test_dummy_task_against_capable_adapter_is_ok(full_adapter):
    task = registry.build("task", "dummy")
    result = task.evaluate(full_adapter)
    assert result.status is ResultStatus.OK
    assert result.primary_metric == "mean_score"
    assert "mean_score" in result.metrics
    assert result.metadata["n_scored"] == 5


def test_dummy_task_is_deterministic(full_adapter):
    r1 = registry.build("task", "dummy").evaluate(full_adapter)
    r2 = registry.build("task", "dummy").evaluate(full_adapter)
    assert r1.metrics == r2.metrics


def test_embed_task_against_score_only_is_na_not_crash(score_only_adapter):
    task = registry.build("task", "dummy-embed")
    result = task.evaluate(score_only_adapter)
    assert result.status is ResultStatus.NA
    assert result.primary_metric is None
    assert result.metadata["missing_capabilities"] == ["embedding"]


def test_task_raising_in_run_is_captured_as_error(full_adapter):
    class BoomTask(Task):
        name = "boom"
        version = "0.0.0"
        required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

        def prepare(self) -> None:
            pass

        def run(self, adapter: ModelAdapter) -> None:
            raise RuntimeError("kaboom")

        def score(self) -> TaskResult:  # pragma: no cover - never reached
            raise AssertionError("score should not run after run() raised")

    result = BoomTask().evaluate(full_adapter)
    assert result.status is ResultStatus.ERROR
    assert "kaboom" in result.metadata["error"]


# --- config schemas --------------------------------------------------------


def test_load_model_spec_and_from_spec(full_adapter):
    spec = load_model_spec(SPECS / "echo-full.yaml")
    assert isinstance(spec, ModelSpec)
    assert spec.adapter == "echo"
    adapter = build_adapter(spec)
    assert isinstance(adapter, EchoAdapter)
    assert adapter.capabilities() == frozenset(Capability)


def test_score_only_spec_restricts_capabilities():
    spec = load_model_spec(SPECS / "echo-score-only.yaml")
    adapter = build_adapter(spec)
    assert adapter.capabilities() == frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})


def test_model_spec_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("adapter: echo\nadapter_version: '0.1.0'\nbogus_key: 1\n")
    with pytest.raises(ValidationError):  # extra=forbid rejects unknown keys
        load_model_spec(bad)


def test_resolve_bundled_benchmark():
    spec = resolve_benchmark_spec("dummy_bench")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.tasks == ["dummy", "dummy-embed"]


def test_resolve_unknown_benchmark_is_loud():
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_benchmark_spec("no_such_benchmark_xyz")


def test_load_benchmark_spec_requires_tasks(tmp_path):
    bad = tmp_path / "b.yaml"
    bad.write_text("name: x\nversion: '1.0'\ntasks: []\n")
    with pytest.raises(ValidationError):  # tasks min_length=1
        load_benchmark_spec(bad)


# --- store idempotency -----------------------------------------------------


def test_store_is_idempotent_by_model_hash(tmp_path, full_adapter):
    spec = load_model_spec(SPECS / "echo-full.yaml")
    bench = resolve_benchmark_spec("dummy_bench")
    run_benchmark(spec, bench, results_dir=tmp_path)
    run_benchmark(spec, bench, results_dir=tmp_path)  # same hash → overwrite, not dup
    records = read_results(tmp_path, "dummy_bench")
    assert len(records) == 1


def test_result_filename_is_fs_safe():
    assert result_filename("glmb:abc123") == "glmb_abc123.json"


# --- the gate: end-to-end board with an N/A cell ---------------------------


def test_gate_two_model_board_has_na_cell(tmp_path):
    """glmbench run (full + score-only) over the 2-task dummy benchmark → board, N/A cell."""
    bench = resolve_benchmark_spec("dummy_bench")
    full = load_model_spec(SPECS / "echo-full.yaml")
    scoreonly = load_model_spec(SPECS / "echo-score-only.yaml")

    rec_full = run_benchmark(full, bench, results_dir=tmp_path)
    rec_score = run_benchmark(scoreonly, bench, results_dir=tmp_path)
    assert rec_full["model_hash"] != rec_score["model_hash"]

    records = read_results(tmp_path, "dummy_bench")
    assert len(records) == 2

    # full model: both tasks OK; score-only: dummy OK, dummy-embed NA
    by_hash = {r["model_hash"]: r for r in records}
    score_tasks = {t["task"]: t for t in by_hash[rec_score["model_hash"]]["tasks"]}
    assert score_tasks["dummy"]["status"] == "ok"
    assert score_tasks["dummy-embed"]["status"] == "na"

    md = render_markdown(
        records,
        benchmark_name=bench.name,
        benchmark_version=bench.version,
        task_order=bench.tasks,
    )
    assert "## Overall" in md
    assert "## Task: dummy" in md
    assert "## Task: dummy-embed" in md
    assert "N/A" in md  # the score-only model's embed cell
    # full model is on the board too
    assert "echo@0.1.0" in md


def test_aggregate_dataframe_one_row_per_model_task(tmp_path):
    bench = resolve_benchmark_spec("dummy_bench")
    for spec_name in ("echo-full.yaml", "echo-score-only.yaml"):
        run_benchmark(load_model_spec(SPECS / spec_name), bench, results_dir=tmp_path)
    records = read_results(tmp_path, "dummy_bench")
    df = aggregate_dataframe(records)
    # 2 models × 2 tasks = 4 rows
    assert len(df) == 4
    assert set(df["task"]) == {"dummy", "dummy-embed"}
    assert "metric.mean_score" in df.columns


def test_write_leaderboard_writes_md_and_csv(tmp_path):
    bench = resolve_benchmark_spec("dummy_bench")
    run_benchmark(load_model_spec(SPECS / "echo-full.yaml"), bench, results_dir=tmp_path)
    records = read_results(tmp_path, "dummy_bench")
    md_path = tmp_path / "out" / "LEADERBOARD.md"
    write_leaderboard(
        records,
        benchmark_name=bench.name,
        benchmark_version=bench.version,
        task_order=bench.tasks,
        md_path=md_path,
        csv_path=md_path.with_suffix(".csv"),
    )
    assert md_path.exists()
    assert md_path.with_suffix(".csv").exists()
    assert "# Leaderboard — dummy_bench" in md_path.read_text()


# --- error row keeps the suite alive ---------------------------------------


def test_error_task_does_not_abort_run(tmp_path, full_adapter):
    @registry.register("task", "boom-run")
    class BoomRun(Task):
        name = "boom-run"
        version = "0.0.0"
        required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

        def prepare(self) -> None:
            pass

        def run(self, adapter: ModelAdapter) -> None:
            raise RuntimeError("explode")

        def score(self) -> TaskResult:  # pragma: no cover
            raise AssertionError

    try:
        bench = BenchmarkSpec(
            name="boom_bench", version="1.0", tasks=["dummy", "boom-run"]
        )
        spec = load_model_spec(SPECS / "echo-full.yaml")
        record = run_benchmark(spec, bench, results_dir=tmp_path)
        statuses = {t["task"]: t["status"] for t in record["tasks"]}
        assert statuses == {"dummy": "ok", "boom-run": "error"}
    finally:
        registry.unregister("task", "boom-run")
