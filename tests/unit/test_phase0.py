"""Scaffold and packaging: the torch-free import gate and the registry.

- ``import glmbench`` works (in a torch-free env).
- The generic registry: register / resolve / build / list / kind-isolation /
  fail-loud-on-duplicate.
- Core stays torch-free: no core module under src/glmbench imports torch
  (only ``runners/*`` may). Asserted by static grep over the source tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import glmbench
from glmbench import registry


def test_import_and_version():
    assert isinstance(glmbench.__version__, str)
    # semantic-ish version string
    assert re.match(r"^\d+\.\d+\.\d+", glmbench.__version__)


def test_torch_not_imported_by_core():
    """Importing glmbench must not pull torch into the process.

    Checked in a **clean subprocess** so the result is independent of whatever
    else this test session imported (e.g. an adapter test that imports torch at
    collection time; that must not make this gate lie). On a torch-free CI this
    is trivially true; here it proves the actual claim — ``import glmbench`` itself
    does not drag torch in.
    """
    import subprocess
    import sys

    code = "import glmbench, glmbench.adapters, glmbench.cli; import sys; assert 'torch' not in sys.modules"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"core import dragged torch into sys.modules:\n{proc.stdout}\n{proc.stderr}"
    )


# --- registry -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    # Isolate each test on its own scratch kinds so we never touch real registries.
    yield
    registry.clear("phase0_kind")
    registry.clear("phase0_other")


def test_register_resolve_build():
    @registry.register("phase0_kind", "thing")
    class Thing:
        def __init__(self, x=1):
            self.x = x

    assert registry.resolve("phase0_kind", "thing") is Thing
    built = registry.build("phase0_kind", "thing", x=7)
    assert isinstance(built, Thing) and built.x == 7


def test_list_is_sorted_and_kind_isolated():
    @registry.register("phase0_kind", "bbb")
    def _b():
        return "b"

    @registry.register("phase0_kind", "aaa")
    def _a():
        return "a"

    @registry.register("phase0_other", "aaa")
    def _other_a():
        return "other"

    assert registry.list("phase0_kind") == ["aaa", "bbb"]
    # Same name under a different kind is independent.
    assert registry.list("phase0_other") == ["aaa"]
    assert registry.resolve("phase0_other", "aaa") is _other_a
    assert "phase0_kind" in registry.kinds()


def test_list_unknown_kind_is_empty():
    assert registry.list("never_registered_kind") == []


def test_duplicate_registration_is_loud():
    @registry.register("phase0_kind", "dup")
    def _one():
        return 1

    with pytest.raises(ValueError, match="already taken"):

        @registry.register("phase0_kind", "dup")
        def _two():
            return 2


def test_resolve_unknown_is_loud():
    with pytest.raises(KeyError, match="not found"):
        registry.resolve("phase0_kind", "missing")


def test_unregister():
    @registry.register("phase0_kind", "temp")
    def _t():
        return 1

    assert "temp" in registry.list("phase0_kind")
    registry.unregister("phase0_kind", "temp")
    assert "temp" not in registry.list("phase0_kind")


# --- core stays torch-free (static) ---------------------------------------

_SRC = Path(__file__).resolve().parents[2] / "src" / "glmbench"
_TORCH_RE = re.compile(r"^\s*(import torch|from torch[ .])", re.MULTILINE)


def test_core_source_is_torch_free():
    """No core module may import torch; only ``runners/*`` may (it runs in the
    adapter's heavy env, not in core). This is the torch-free gate, enforced statically
    so it holds as modules are added."""
    offenders = []
    for py in _SRC.rglob("*.py"):
        rel = py.relative_to(_SRC)
        if rel.parts and rel.parts[0] == "runners":
            continue  # runners run inside the adapter env — torch is allowed there
        if "_vendor" in rel.parts:
            continue  # vendored upstream reference (verbatim, never imported by core);
            # excluded from ruff/mypy in pyproject for the same reason
        if _TORCH_RE.search(py.read_text()):
            offenders.append(str(rel))
    assert not offenders, f"torch imported by core module(s): {offenders}"
