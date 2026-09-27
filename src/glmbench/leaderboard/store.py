"""Leaderboard store — per-run result JSON, idempotent by model hash.

After a benchmark run we write ``results/<benchmark>/<model_hash>.json`` holding the
model hash, the model-spec echo, adapter + gLMBench + benchmark versions, the per-task
:class:`~glmbench.tasks.base.TaskResult`s, and provenance (timestamp, runner config,
package versions, data-manifest hashes). Writing is **idempotent**: the same model hash
overwrites its own file (re-running replaces, never duplicates).

This module is core (torch-free): stdlib + pydantic-echoed dicts only.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glmbench import __version__

if TYPE_CHECKING:
    from glmbench.adapters.base import ModelAdapter
    from glmbench.config.benchmark_spec import BenchmarkSpec
    from glmbench.config.model_spec import ModelSpec
    from glmbench.tasks.base import TaskResult

logger = logging.getLogger(__name__)


def _load_result_json(path: Path) -> dict[str, Any] | None:
    """Parse one result JSON, or warn loudly and return ``None`` if it is empty/corrupt.

    A result file truncated to 0 bytes (or partially flushed) is exactly what a disk-full
    crash mid-write leaves behind. A bare ``json.loads`` on it raises ``JSONDecodeError`` and
    aborts the whole suite before any task runs — the wrong failure mode for a stale, valueless
    artifact. We instead SCREAM and treat it as "no usable prior": the resume
    path recomputes from scratch, and a re-run overwrites the bad file with a good one.
    """
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(
            "LOUD: ignoring unreadable result JSON %s (%s) — treating as absent. A re-run "
            "will overwrite it; delete it if this persists. (Likely a disk-full truncation.)",
            path,
            e,
        )
        return None


def result_filename(model_hash: str) -> str:
    """Filesystem-safe filename for a model hash (``glmb:abc`` → ``glmb_abc.json``)."""
    safe = model_hash.replace(":", "_").replace("/", "_")
    return f"{safe}.json"


#: Distributions whose versions can move a metric's last digits: the probes (scikit-learn,
#: scipy, numpy) and the tables they read (pandas). Read from package metadata, so nothing is
#: imported. These are the versions of the process that fits the probes and computes the
#: metrics, not of the runner env that computed the embeddings.
_VERSIONED_PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas")


def package_versions() -> dict[str, str | None]:
    """``{package: version}`` for gLMBench, Python and :data:`_VERSIONED_PACKAGES`."""
    versions: dict[str, str | None] = {"glmbench": __version__, "python": platform.python_version()}
    for name in _VERSIONED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_result_record(
    *,
    model_spec: ModelSpec,
    benchmark_spec: BenchmarkSpec,
    adapter: ModelAdapter,
    task_results: list[TaskResult],
    data_manifest_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the full provenance-bearing result record for one model × benchmark.

    *data_manifest_hashes* maps a task name to the sha256 of the data manifest it read.
    """
    return {
        "model_hash": adapter.model_hash(),
        "adapter_name": adapter.name,
        "adapter_version": adapter.adapter_version,
        "glmbench_version": __version__,
        "benchmark": {"name": benchmark_spec.name, "version": benchmark_spec.version},
        "model_spec": model_spec.echo(),
        # WHICH TENSOR this run's embeddings are, at the top level of the record rather than
        # buried in `describe()`, so a consumer need not know each adapter's describe() shape
        # to tell whether two rows read the same kind of tensor.
        **adapter.readout_declaration(),
        "describe": adapter.describe(),
        "tasks": [tr.to_dict() for tr in task_results],
        "provenance": {
            "timestamp": datetime.now(UTC).isoformat(),
            "runner": model_spec.runner.model_dump(mode="json"),
            "package_versions": package_versions(),
            "data_manifest_hashes": data_manifest_hashes or {},
        },
    }


def write_result(
    results_dir: str | Path, benchmark_name: str, record: dict[str, Any]
) -> Path:
    """Write *record* to ``<results_dir>/<benchmark_name>/<model_hash>.json`` (idempotent)."""
    model_hash = record["model_hash"]
    out_dir = Path(results_dir) / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / result_filename(model_hash)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return out_path


def read_result(
    results_dir: str | Path, benchmark_name: str, model_hash: str
) -> dict[str, Any] | None:
    """Read the stored record for one model hash, or ``None`` if not yet written.

    Used by the resume path: a prior (possibly partial) record lets a re-run skip the
    tasks that already completed OK instead of recomputing the whole suite.
    """
    path = Path(results_dir) / benchmark_name / result_filename(model_hash)
    if not path.is_file():
        return None
    return _load_result_json(path)


def _is_result_record(record: dict[str, Any]) -> bool:
    """Does this parsed JSON actually look like a stored result record?

    The store's own files are ``glmb_<hash>.json``, but the benchmark directory is a place
    humans (and now the site build) drop other JSON next to them. A stray document that is
    valid JSON but not a record used to sail through and blow up downstream with a bare
    ``KeyError: 'task'`` — a nonsense error a long way from its cause. Check the shape here
    and SCREAM instead.
    """
    return isinstance(record.get("model_hash"), str) and isinstance(record.get("tasks"), list)


def read_results(results_dir: str | Path, benchmark_name: str) -> list[dict[str, Any]]:
    """Read every stored result JSON for a benchmark, sorted by filename (stable order)."""
    bench_dir = Path(results_dir) / benchmark_name
    if not bench_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(bench_dir.glob("*.json")):
        record = _load_result_json(path)
        if record is None:
            continue
        if not _is_result_record(record):
            logger.warning(
                "LOUD: %s is valid JSON but not a result record (no `model_hash` + `tasks`) — "
                "skipping it. Only the store's own `glmb_<hash>.json` files belong in %s; keep "
                "derived artifacts (leaderboard payloads, plots) out of the results store.",
                path,
                bench_dir,
            )
            continue
        records.append(record)
    return records
