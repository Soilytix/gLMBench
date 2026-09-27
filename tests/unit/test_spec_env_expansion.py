"""Spec portability: ``${VAR}`` expansion in model specs.

The point of this feature is that one committed spec runs on two boxes with different
checkpoint stores and different conda envs. The gate that makes it *safe* is asserted in
``test_relocating_checkpoint_preserves_identity``: a checkpoint's path never enters the
model hash, so pointing a spec at a moved checkpoint keeps its leaderboard row.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from glmbench.adapters.hashing import model_hash
from glmbench.config.model_spec import load_model_spec


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body))
    return p


BASE = """\
    adapter: echo
    adapter_version: "0.1.0"
    model:
      checkpoint: "{ckpt}"
    weights_digest:
      strategy: declared
      value: "v0"
    runner:
      backend: local
      python_exe: "{py}"
"""


def test_expands_var_from_env(tmp_path: Path) -> None:
    p = _write(tmp_path, BASE.format(ckpt="${CKPT_ROOT}/LOAM-25M", py="${PY}"))
    spec = load_model_spec(p, env={"CKPT_ROOT": "/mnt/w", "PY": "/envs/x/bin/python"})
    assert spec.model["checkpoint"] == "/mnt/w/LOAM-25M"
    assert spec.runner.python_exe == "/envs/x/bin/python"


def test_default_is_used_when_var_unset(tmp_path: Path) -> None:
    """The `${VAR:-default}` form is what keeps the original box working unchanged."""
    p = _write(tmp_path, BASE.format(ckpt="${CKPT_ROOT:-/legacy/ckpt}/LOAM-25M", py="/usr/bin/python"))
    spec = load_model_spec(p, env={})
    assert spec.model["checkpoint"] == "/legacy/ckpt/LOAM-25M"


def test_env_overrides_default(tmp_path: Path) -> None:
    p = _write(tmp_path, BASE.format(ckpt="${CKPT_ROOT:-/legacy/ckpt}/LOAM-25M", py="/usr/bin/python"))
    spec = load_model_spec(p, env={"CKPT_ROOT": "/mnt/w"})
    assert spec.model["checkpoint"] == "/mnt/w/LOAM-25M"


def test_unset_var_without_default_fails_loudly(tmp_path: Path) -> None:
    """A missing var must never survive as a literal `${VAR}` into a FileNotFoundError."""
    p = _write(tmp_path, BASE.format(ckpt="${NOPE}/LOAM-25M", py="${ALSO_NOPE}"))
    with pytest.raises(ValueError, match="undefined environment variable"):
        load_model_spec(p, env={})


def test_all_missing_vars_reported_at_once(tmp_path: Path) -> None:
    p = _write(tmp_path, BASE.format(ckpt="${NOPE}/LOAM-25M", py="${ALSO_NOPE}"))
    with pytest.raises(ValueError) as exc:
        load_model_spec(p, env={})
    assert "ALSO_NOPE" in str(exc.value) and "NOPE" in str(exc.value)


def test_bare_dollar_is_left_alone(tmp_path: Path) -> None:
    """Brace-only expansion: docker/gpu strings with a bare `$` must pass through intact."""
    p = _write(tmp_path, BASE.format(ckpt="/a/$NOT_A_VAR/b", py="/usr/bin/python"))
    spec = load_model_spec(p, env={})
    assert spec.model["checkpoint"] == "/a/$NOT_A_VAR/b"


def test_expansion_reaches_nested_lists_and_dicts(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """\
        adapter: echo
        adapter_version: "0.1.0"
        model:
          checkpoint: "/x"
        runner:
          backend: docker
          image: "img"
          extra:
            docker_args: ["-v", "${HOSTDIR}:/data"]
        """,
    )
    spec = load_model_spec(p, env={"HOSTDIR": "/mnt/host"})
    assert spec.runner.extra["docker_args"] == ["-v", "/mnt/host:/data"]


def test_relocating_checkpoint_preserves_identity() -> None:
    """THE portability gate: the checkpoint path is not an input to the model hash.

    Identity = weights_digest (content) + arch (config contents) + adapter@version. Move the
    weights to a new box under a new root and the hash — hence the leaderboard row — is stable.
    """
    arch = {"model": {"d_model": 512, "n_layers": 12}}
    on_old_box = model_hash(
        weights_digest="file_sha256:abc123",
        config=arch,
        adapter_name="loam-hf",
        adapter_version="0.1.0",
    )
    on_new_box = model_hash(  # same weights + arch, relocated to /mnt/weights
        weights_digest="file_sha256:abc123",
        config=arch,
        adapter_name="loam-hf",
        adapter_version="0.1.0",
    )
    assert on_old_box == on_new_box
