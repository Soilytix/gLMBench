"""Typed run specs — one spec, one run, one result.

A run is fully described by a :class:`ModelSpec` (which model + how to run it) and a
:class:`BenchmarkSpec` (which ordered tasks). Every knob lives in these Pydantic
schemas — no hidden defaults in code. This package is core (torch-free): pydantic +
pyyaml + stdlib only.
"""

from __future__ import annotations

from .benchmark_spec import (
    BenchmarkSpec,
    builtin_benchmarks_dir,
    load_benchmark_spec,
    resolve_benchmark_spec,
)
from .model_spec import (
    ModelSpec,
    RunnerSpec,
    WeightsDigestSpec,
    load_model_spec,
)

__all__ = [
    "BenchmarkSpec",
    "builtin_benchmarks_dir",
    "load_benchmark_spec",
    "resolve_benchmark_spec",
    "ModelSpec",
    "RunnerSpec",
    "WeightsDigestSpec",
    "load_model_spec",
]
