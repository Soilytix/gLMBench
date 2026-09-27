"""ModelSpec — the typed description of *which model* and *how to run it*.

A ``specs/*.yaml`` file deserializes into a :class:`ModelSpec`: the adapter to use,
the adapter version it was written against, adapter-specific model config, the
weights-digest strategy, and the runner backend config. Unknown keys are rejected loudly
(``extra="forbid"``) so a typo'd knob never silently no-ops — every knob must live in
the schema.

**Portability (``${VAR}`` expansion).** Specs must be able to name a checkpoint store and
a runner python without hardcoding one box's filesystem. Every string in the YAML is
expanded against the environment before validation, supporting ``${NAME}`` and
``${NAME:-default}``. An unset ``${NAME}`` with no default is a **loud** error naming the
variable — never a literal ``${NAME}`` that surfaces later as a baffling FileNotFoundError.
See ``docs/INSTALL.md`` for the conventional variables.

This is safe for leaderboard continuity precisely because a checkpoint's *path* never
enters the model hash: identity is ``weights_digest`` (content sha256 / HF commit)
+ the arch read from the config's *contents*. Relocating a checkpoint therefore preserves
its hash and its existing leaderboard row.

This module is core (torch-free): pydantic + pyyaml + stdlib only.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ${NAME} or ${NAME:-default}. Deliberately brace-only: a bare `$NAME` is left alone so
# shell-ish strings (docker args, gpus: '"device=0"') can never be mangled by accident.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class WeightsDigestSpec(BaseModel):
    """How to derive the weight-sensitive digest that keys the model hash."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    strategy: Literal["file_sha256", "hf_revision", "declared"] = "file_sha256"
    # required only for hf_revision (commit hash) / declared (user-supplied digest)
    value: str | None = None


class RunnerSpec(BaseModel):
    """The execution backend for the adapter's foreign env."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    backend: Literal["local", "docker"] = "local"
    # local backend: the adapter env's python; None → the adapter's own default
    python_exe: str | None = None
    # docker backend: the container image
    image: str | None = None
    # scratch dir for wire files; None → a temp dir
    scratch_dir: str | None = None
    # passed through to the runner (e.g. "0" or "all")
    gpus: str | None = "0"
    # adapter-specific knobs (tensor_parallel_size, micro_batch_size, …)
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    """One model + how to run it. Fully describes the model half of a run."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # registry key -> ModelAdapter subclass
    adapter: str
    # The adapter version this spec was written against. It does NOT enter the model hash
    # (the hash uses the adapter class's own version); building the adapter fails if the
    # two differ, so a spec cannot silently run under an adapter that has changed since.
    adapter_version: str
    # Optional human-readable label for the leaderboard (e.g. "LOAM-100M").
    # PROVENANCE ONLY: it is echoed into the record and rendered next to the hash, but it
    # does NOT participate in the model hash — so renaming a model never changes its
    # identity, and two checkpoints sharing a display_name still resolve to distinct hashes
    # (and distinct leaderboard rows) whenever their weights or config differ.
    display_name: str | None = None
    # adapter-specific; echoed into the model-hash config (checkpoint, arch config, …)
    model: dict[str, Any] = Field(default_factory=dict)
    weights_digest: WeightsDigestSpec = Field(default_factory=WeightsDigestSpec)
    runner: RunnerSpec = Field(default_factory=RunnerSpec)

    # Provenance only — the file this spec was loaded from (not part of the hash).
    source_path: str | None = Field(default=None, exclude=True)

    def echo(self) -> dict[str, Any]:
        """Plain-dict echo of the spec for the leaderboard provenance record."""
        return self.model_dump(mode="json")


def _expand_str(raw: str, env: dict[str, str], missing: list[str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        val = env.get(name)
        if val:  # set and non-empty
            return val
        if default is not None:
            return default
        missing.append(name)
        return m.group(0)

    return _VAR_RE.sub(_sub, raw)


def expand_env(obj: Any, env: dict[str, str], missing: list[str]) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in every string of a spec tree.

    Unset-and-defaultless variable names are appended to *missing* rather than raised on the
    spot, so one load reports **all** of them at once instead of one per re-run.
    """
    if isinstance(obj, str):
        return _expand_str(obj, env, missing)
    if isinstance(obj, dict):
        return {k: expand_env(v, env, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v, env, missing) for v in obj]
    return obj


def load_model_spec(path: str | Path, *, env: dict[str, str] | None = None) -> ModelSpec:
    """Load + validate a ``ModelSpec`` from a YAML file (loud on unknown keys).

    ``${VAR}`` / ``${VAR:-default}`` in any string are expanded against *env* (default
    ``os.environ``) before validation — see the module docstring.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model spec not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Model spec {p} must be a YAML mapping, got {type(data).__name__}.")

    missing: list[str] = []
    data = expand_env(data, dict(os.environ) if env is None else env, missing)
    if missing:
        names = sorted(set(missing))
        raise ValueError(
            f"Model spec {p} references undefined environment variable(s): "
            f"{', '.join(names)}. Either export them or give them a default in the spec "
            f"(${{NAME:-/some/path}}). See docs/INSTALL.md."
        )

    spec = ModelSpec.model_validate(data)
    spec.source_path = str(p)
    return spec
