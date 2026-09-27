"""Core (torch-free) tests for the ``loam-hf`` adapter (LOAM models in Hugging Face format).

The adapter builds from an export directory, bounds the context at
``max_position_embeddings - 1`` (BOS takes a slot), hashes only the weight files, and refuses
weights that disagree with ``loam_export.json``. All of it runs on a synthetic export
directory; ``test_loam_hf_runner.py`` serves a real (tiny) model end to end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from glmbench import registry
from glmbench.adapters.base import Capability, Readout
from glmbench.adapters.loam_hf import LOAMHFAdapter
from glmbench.config.model_spec import ModelSpec


def _fake_export(tmp: Path, *, max_pos: int = 64, weights: bytes = b"weights", record: str | None = None) -> Path:
    d = tmp / "export"
    d.mkdir(parents=True)
    cfg = {
        "model_type": "loam", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "vocab_size": 58,
        "max_position_embeddings": max_pos, "norm_placement": "peri", "qk_norm": True,
        "loam_tokenizer_hash": "sha256:test", "bos_token_id": 1, "pad_token_id": 0,
    }
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer_config.json").write_text(json.dumps({"loam_tokenizer_name": "nucleotide"}))
    (d / "model.safetensors").write_bytes(weights)
    sha = hashlib.sha256(weights).hexdigest() if record is None else record
    (d / "loam_export.json").write_text(json.dumps({"weight_sha256": {"model.safetensors": sha}}))
    return d


# --- core tier -----------------------------------------------------------------------------


def test_registered_and_declares_block_output_readout() -> None:
    assert registry.resolve("adapter", "loam-hf") is LOAMHFAdapter
    assert LOAMHFAdapter.readout is Readout.BLOCK_OUTPUT
    assert LOAMHFAdapter.readout_evidence == "by_construction"


def test_context_reserves_one_position_for_bos(tmp_path: Path) -> None:
    a = LOAMHFAdapter(checkpoint=str(_fake_export(tmp_path, max_pos=8192)), dry_run=True)
    assert a.max_context == 8191
    assert a.n_layers == 2 and a.embedding_dim == 32


def test_nucleotide_tokenizer_adds_logprob_embedding(tmp_path: Path) -> None:
    a = LOAMHFAdapter(checkpoint=str(_fake_export(tmp_path)), dry_run=True)
    assert Capability.LOGPROB_EMBEDDING in a.capabilities()
    assert Capability.EMBEDDING in a.capabilities()


def test_hash_ignores_non_weight_files(tmp_path: Path) -> None:
    d = _fake_export(tmp_path)
    before = LOAMHFAdapter(checkpoint=str(d), dry_run=True).model_hash()
    (d / "README.md").write_text("a new model card")
    after = LOAMHFAdapter(checkpoint=str(d), dry_run=True).model_hash()
    assert before == after


def test_hash_moves_with_weights_and_serving_dtype(tmp_path: Path) -> None:
    a = LOAMHFAdapter(checkpoint=str(_fake_export(tmp_path / "a", weights=b"w1")), dry_run=True)
    b = LOAMHFAdapter(checkpoint=str(_fake_export(tmp_path / "b", weights=b"w2")), dry_run=True)
    c = LOAMHFAdapter(
        checkpoint=str(_fake_export(tmp_path / "c", weights=b"w1")), pool_dtype="float32", dry_run=True
    )
    assert len({a.model_hash(), b.model_hash(), c.model_hash()}) == 3


def test_refuses_weights_that_disagree_with_the_export_record(tmp_path: Path) -> None:
    d = _fake_export(tmp_path, record="0" * 64)
    with pytest.raises(ValueError, match="loam_export.json"):
        LOAMHFAdapter(checkpoint=str(d), dry_run=True).model_hash()


def test_rejects_non_loam_exports(tmp_path: Path) -> None:
    d = _fake_export(tmp_path)
    cfg = json.loads((d / "config.json").read_text())
    cfg["model_type"] = "llama"
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="model_type"):
        LOAMHFAdapter(checkpoint=str(d), dry_run=True)


def test_relative_checkpoint_resolves_against_the_spec_file(tmp_path: Path) -> None:
    _fake_export(tmp_path)
    spec = ModelSpec(adapter="loam-hf", adapter_version="1.0.0", model={"checkpoint": "export"},
                     source_path=str(tmp_path / "spec.yaml"))
    a = LOAMHFAdapter.from_spec(spec, dry_run=True)
    assert a.model_dir == (tmp_path / "export").resolve()


def test_hub_checkpoint_is_served_by_repo_id_at_the_hashed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hub download is hashed from its cache snapshot but loaded by ``repo id @ commit``.

    Loading the snapshot directory itself fails: its files are symlinks into ``blobs/``, and
    transformers looks for the remote code's relative imports next to the resolved blob.
    """
    commit = "0123456789abcdef0123456789abcdef01234567"
    src = _fake_export(tmp_path)
    snapshot = tmp_path / "hub" / "models--Org--LOAM-T" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for f in src.iterdir():
        (snapshot / f.name).symlink_to(f)
    calls = []

    def fake_snapshot_download(repo_id: str, revision: str | None = None) -> str:
        calls.append((repo_id, revision))
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    hub = LOAMHFAdapter(checkpoint="Org/LOAM-T", dry_run=True)
    local = LOAMHFAdapter(checkpoint=str(src), dry_run=True)

    assert calls == [("Org/LOAM-T", None)]
    params = hub._common_params()
    assert (params["checkpoint"], params["revision"]) == ("Org/LOAM-T", commit)
    assert hub.describe()["hub_commit"] == commit
    assert hub.model_hash() == local.model_hash()  # same weight bytes, same hash
    assert (local._common_params()["checkpoint"], local._common_params()["revision"]) == (
        str(src.resolve()), None)
