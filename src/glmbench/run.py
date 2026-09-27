"""The benchmark runner — one runner, all models.

The same :func:`run_benchmark` drives a small LOAM checkpoint on a laptop and a 7B Evo2
on a GPU: the difference is the adapter + its runner config in the
:class:`~glmbench.config.model_spec.ModelSpec`, never a code path. It builds the
adapter from the spec, evaluates each task in benchmark order (capability negotiation +
error capture live in :meth:`~glmbench.tasks.base.Task.evaluate`), and assembles the
provenance-bearing result record.

This module is core (torch-free): it only orchestrates the (torch-free) contracts and
shells out to runners via the adapter.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from glmbench import registry
from glmbench.adapters.base import ModelAdapter
from glmbench.adapters.reuse import ReusingAdapter, plan_fusion
from glmbench.config.benchmark_spec import BenchmarkSpec
from glmbench.config.model_spec import ModelSpec
from glmbench.leaderboard.store import build_result_record, read_result, write_result
from glmbench.tasks.base import ResultStatus, TaskResult

_STATUS_MARK = {"ok": "OK", "na": "N/A", "error": "ERROR"}


def _log_task_done(idx: int, n: int, result: TaskResult) -> None:
    """Emit a one-line per-task verdict (surfaced live, since the CLI logs at INFO)."""
    pm = result.primary_metric
    val = result.metrics.get(pm) if pm else None
    head = f"{pm}={val:.4f}" if isinstance(val, float | int) else "—"
    mark = _STATUS_MARK.get(result.status.value, result.status.value)
    logger.info("[%d/%d] %s: %s  %s", idx, n, result.task, mark, head)

logger = logging.getLogger(__name__)

# hf_revision values that name a *mutable* ref rather than an immutable commit. Two
# different weight sets pinned to the same mutable ref hash identically, so
# different weights could silently collide into one leaderboard row.
_MUTABLE_HF_REFS = frozenset({"main", "master", "head", "latest", "dev", ""})


def warn_on_weak_weights_digest(model_spec: ModelSpec) -> str | None:
    """Loudly flag a weights-digest strategy that is NOT content-addressed.

    The model hash keys every leaderboard row and result file; its weight-sensitivity is
    what keeps two same-named checkpoints with *different weights* on distinct rows. That
    guarantee holds only when the digest is content-addressed (``file_sha256``). A
    ``declared`` digest, or an ``hf_revision`` pinned to a mutable ref (``main``/``master``/
    …), can make different weights collide into one hash — silently overwriting results.
    Returns the warning string (also logged at WARNING) when weak, else ``None``.
    """
    strat = model_spec.weights_digest.strategy
    value = (model_spec.weights_digest.value or "").strip()
    msg: str | None = None
    if strat == "declared":
        msg = (
            f"weights_digest.strategy='declared' (value={value!r}): the model hash is NOT "
            "weight-verified. Two checkpoints with different weights but the same declared "
            "value will collide into one leaderboard row. Use 'file_sha256' for a "
            "content-addressed identity."
        )
    elif strat == "hf_revision" and value.lower() in _MUTABLE_HF_REFS:
        msg = (
            f"weights_digest.strategy='hf_revision' pinned to the MUTABLE ref {value!r}: "
            "the upstream weights behind this ref can change while the hash stays fixed, so "
            "a later run with different weights would silently overwrite this result. Pin an "
            "immutable commit SHA instead."
        )
    if msg is not None:
        logger.warning("LOUD: weak model identity — %s", msg)
    return msg


def check_adapter_version(model_spec: ModelSpec, adapter_cls: type[ModelAdapter]) -> None:
    """Refuse a spec whose ``adapter_version`` is not the version of the adapter it names.

    The model hash reads the version from the adapter class, never from the spec, so a
    mismatched spec would be scored and recorded as the class's version while claiming
    another. The spec field pins the version the spec was written against; a mismatch means
    the adapter changed since, and the spec must be checked and updated on purpose.
    """
    declared = model_spec.adapter_version
    actual = getattr(adapter_cls, "adapter_version", "")
    if declared and declared != actual:
        raise ValueError(
            f"Model spec {model_spec.source_path or '(in memory)'} pins adapter_version "
            f"{declared!r}, but the '{model_spec.adapter}' adapter is version {actual!r}. The "
            f"model hash uses the adapter's own version, so set `adapter_version: \"{actual}\"` "
            f"in the spec once you have checked that the adapter still does what the spec expects."
        )


def build_adapter(
    model_spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
) -> ModelAdapter:
    """Resolve the adapter class from the registry and build it from the spec."""
    adapter_cls = registry.resolve("adapter", model_spec.adapter)
    check_adapter_version(model_spec, adapter_cls)
    adapter = adapter_cls.from_spec(model_spec, keep_scratch=keep_scratch, dry_run=dry_run)
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(
            f"Adapter '{model_spec.adapter}'.from_spec returned {type(adapter).__name__}, "
            f"not a ModelAdapter."
        )
    return adapter


def manifest_sha256(task: Any) -> str | None:
    """sha256 of the data manifest *task* reads, or ``None`` if it declares none.

    A data-backed task names its manifest in ``manifest_path``; the manifest pins the
    sha256 of every data file, so its own hash identifies the exact data a score came from.
    """
    path = getattr(task, "manifest_path", None)
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_benchmark(
    model_spec: ModelSpec,
    benchmark_spec: BenchmarkSpec,
    *,
    results_dir: str | Path | None = None,
    keep_scratch: bool = False,
    dry_run: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run every task of *benchmark_spec* against *model_spec*; return the result record.

    Tasks run sequentially, each logged live (``[i/n] <task>: running…`` then a verdict).
    If *results_dir* is given, the record is written **after every task** (idempotently,
    keyed by model hash), so a crash partway through a long suite keeps the tasks that
    already finished instead of discarding everything. A task that raises out of
    ``evaluate`` is still captured as an ``ERROR`` row so one bad task never aborts the run
    (loud but contained).

    With *resume* (default) and a *results_dir*, a re-run reads the prior record for this
    model hash and skips the tasks that already completed ``OK`` — only the remaining (and
    any previously ``N/A``/``ERROR``) tasks are recomputed. Pass ``resume=False`` to force
    a full recompute.
    """
    warn_on_weak_weights_digest(model_spec)
    adapter = build_adapter(model_spec, keep_scratch=keep_scratch, dry_run=dry_run)

    tasks = benchmark_spec.tasks
    n = len(tasks)

    # Resume: pre-seed the OK tasks from a prior (partial or complete) record. NA/ERROR
    # tasks are intentionally NOT seeded — they get another chance (data may now be present).
    done: dict[str, TaskResult] = {}
    # task name -> sha256 of the data manifest it read; a resumed task keeps its prior hash.
    manifest_hashes: dict[str, str] = {}
    if results_dir is not None and resume:
        prior = read_result(results_dir, benchmark_spec.name, adapter.model_hash())
        if prior is not None:
            prior_hashes = (prior.get("provenance") or {}).get("data_manifest_hashes") or {}
            for t in prior.get("tasks", []):
                if t.get("status") == ResultStatus.OK.value:
                    done[t["task"]] = TaskResult.from_dict(t)
                    if t["task"] in prior_hashes:
                        manifest_hashes[t["task"]] = prior_hashes[t["task"]]
            if done:
                logger.info(
                    "resume: %d/%d task(s) already complete for %s — running only the rest",
                    len(done), n, adapter.model_hash(),
                )

    # Forward-pass reuse (adapters/reuse.py): tasks in one fusion group read the same corpus,
    # so ONE forward can serve all of them. The plan is built from the tasks ACTUALLY being run
    # (minus any resumed ones), which is what keeps a single-task run byte-for-byte unchanged:
    # with no second consumer in its group, nothing is promoted and nothing is cached.
    pending = [t for t in tasks if t not in done]
    plans = plan_fusion(pending, registry)
    if plans:
        adapter = ReusingAdapter(adapter, plans)
        for g, p in plans.items():
            if p.n_embed_consumers + p.n_logprob_consumers > 1:
                logger.info(
                    "reuse: group %r — %d task(s) share one corpus (all_layers=%s, logprob=%s); "
                    "one forward pass will serve them",
                    g, p.n_embed_consumers + p.n_logprob_consumers,
                    p.needs_all_layers, p.needs_logprob,
                )

    task_results: list[TaskResult] = []
    for idx, task_name in enumerate(tasks, start=1):
        if task_name in done:
            logger.info("[%d/%d] %s: SKIP (resumed, already OK)", idx, n, task_name)
            task_results.append(done[task_name])
            continue
        logger.info("[%d/%d] %s: running…", idx, n, task_name)
        task = registry.build("task", task_name)
        digest = manifest_sha256(task)
        if digest is not None:
            manifest_hashes[task_name] = digest
        if isinstance(adapter, ReusingAdapter):
            adapter.begin_task(getattr(task, "fusion_group", None))
        try:
            result = task.evaluate(adapter)
        except Exception as e:  # noqa: BLE001 - defense in depth around evaluate()
            result = TaskResult(task_name, ResultStatus.ERROR, None, {}, {"error": repr(e)})
        task_results.append(result)
        _log_task_done(idx, n, result)

        # Persist incrementally so a late crash keeps the completed tasks and a re-run
        # resumes from here. The final state == the last in-loop write (idempotent by hash).
        if results_dir is not None:
            partial = build_result_record(
                model_spec=model_spec,
                benchmark_spec=benchmark_spec,
                adapter=adapter,
                task_results=task_results,
                data_manifest_hashes=manifest_hashes,
            )
            write_result(results_dir, benchmark_spec.name, partial)

    # Persist the COMPLETE record one final time. The in-loop write above only fires for
    # computed tasks (resumed SKIPs hit `continue`), so if the last *pending* task is not the
    # last task in the benchmark, every already-OK task after it would be missing from the last
    # in-loop write. Without this final write the on-disk record silently loses those trailing
    # resumed tasks. Idempotent by hash; the caller uses the return only for the console summary.
    final = build_result_record(
        model_spec=model_spec,
        benchmark_spec=benchmark_spec,
        adapter=adapter,
        task_results=task_results,
        data_manifest_hashes=manifest_hashes,
    )
    if results_dir is not None:
        write_result(results_dir, benchmark_spec.name, final)
    return final
