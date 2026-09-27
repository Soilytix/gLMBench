"""BenchmarkSpec — the typed, ordered list of tasks that make up a benchmark.

A ``benchmarks/*.yaml`` file deserializes into a :class:`BenchmarkSpec`. Benchmarks
ship bundled inside the package (``glmbench/benchmarks/*.yaml``) and are resolvable by
name, or loaded from an arbitrary path. This module is core (torch-free).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkSpec(BaseModel):
    """An ordered list of task registry-keys plus identifying metadata."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    version: str
    description: str = ""
    tasks: list[str] = Field(min_length=1)

    source_path: str | None = Field(default=None, exclude=True)


def builtin_benchmarks_dir() -> Path:
    """The directory of benchmarks bundled inside the package (``glmbench/benchmarks``)."""
    return Path(__file__).resolve().parent.parent / "benchmarks"


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load + validate a ``BenchmarkSpec`` from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark spec not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark spec {p} must be a YAML mapping, got {type(data).__name__}.")
    spec = BenchmarkSpec.model_validate(data)
    spec.source_path = str(p)
    return spec


def resolve_benchmark_spec(name_or_path: str | Path) -> BenchmarkSpec:
    """Resolve a benchmark by filesystem path, else by bundled name.

    If ``name_or_path`` is an existing file it is loaded directly; otherwise it is
    treated as the name of a bundled benchmark (``glmbench/benchmarks/<name>.yaml``).
    """
    p = Path(name_or_path)
    if p.exists():
        return load_benchmark_spec(p)
    builtin = builtin_benchmarks_dir() / f"{name_or_path}.yaml"
    if builtin.exists():
        return load_benchmark_spec(builtin)
    available = sorted(f.stem for f in builtin_benchmarks_dir().glob("*.yaml"))
    raise FileNotFoundError(
        f"Benchmark {name_or_path!r} not found as a path or a bundled benchmark. "
        f"Available bundled benchmarks: {available if available else '(none)'}."
    )
