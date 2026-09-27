"""Evo2ArcAdapter (Evo2 via the ArcInstitute ``evo2`` package).

CPU-only, **torch-free** gates: the adapter is core, so everything here runs without torch,
without the ``evo2`` package, and without the 16 GB checkpoint. The GPU behaviour (real
forwards, real named-layer taps) needs the real model and is not part of this suite.

What these lock down:

- capability consistency (fail-closed) — declared == overridden;
- the **layer-name ⇄ integer-id** mapping, which is the whole design of this adapter:
  ``layers='last'`` must resolve to the LAST configured name (that is what makes
  ``bacbench-essentiality`` read block 28 with no task-side change), ``'all'`` to every
  configured name, and an int/list to positions in that list;
- ``layer_names`` is **in the model hash** — two specs that differ only by tap must key
  different leaderboard rows, or the second run silently overwrites the first;
- arch-specific layer defaults (7B → ``blocks.28.mlp.l3``, 1B → ``blocks.24.mlp.l3``) and a
  loud failure for an unknown model with no explicit ``layer_names``;
- the wire request the adapter builds (op + params + argv), since a wrong ``layer_names``
  param would silently embed the wrong tensor.
"""

from __future__ import annotations

import sys

import pytest

from glmbench.adapters.base import (
    Capability,
    CapabilityNotImplemented,
    missing_capability_methods,
)
from glmbench.adapters.evo2_arc import MODEL2LAYER, Evo2ArcAdapter
from glmbench.config.model_spec import ModelSpec

SEQS = ["ATGACGTACGTACGT", "ACGTACGTACGTACGTACGTACGT", "TTGACAGCTAGCTCAG"]
DIGEST = "074097e9dc788e8bfe045d6495b9f6153a7c6bfc"


def _adapter(**kw) -> Evo2ArcAdapter:
    base = {
        "model_name": "evo2_7b_base",
        "python_exe": sys.executable,
        "weights_digest_strategy": "hf_revision",
        "weights_digest_value": DIGEST,
    }
    base.update(kw)
    return Evo2ArcAdapter(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# capability contract (fail-closed)
# --------------------------------------------------------------------------- #


def test_capabilities_cover_scoring_and_embedding():
    """All three — the point of the Arc route is that ONE forward serves every column."""
    assert frozenset(
        {
            Capability.SEQUENCE_LOGLIKELIHOOD,
            Capability.EMBEDDING,
            Capability.LOGPROB_EMBEDDING,
        }
    ) == Evo2ArcAdapter.CAPABILITIES
    assert missing_capability_methods(Evo2ArcAdapter) == []


def test_undeclared_capabilities_raise():
    a = _adapter()
    with pytest.raises(CapabilityNotImplemented):
        a.tokenize(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        a.score_variant_llr([])


# --------------------------------------------------------------------------- #
# layer name ⇄ integer id — the load-bearing mapping
# --------------------------------------------------------------------------- #


def test_last_resolves_to_the_last_configured_name():
    """`layers='last'` is what every single-layer embedding task asks for.

    With one configured name that must be block 28 — this single assertion is why
    bacbench-essentiality reads Arc's recommended tap without any task edit.
    """
    a = _adapter()
    assert a.layer_names == ["blocks.28.mlp.l3"]
    assert a._resolve_layers("last") == [0]
    assert a.describe()["embedding_layer"] == "blocks.28.mlp.l3"


def test_last_is_the_last_name_not_the_deepest_block():
    """Order in the spec defines the id; the adapter must not re-sort by block number."""
    a = _adapter(layer_names=["blocks.28.mlp.l3", "blocks.4.mlp.l3"])
    assert a._resolve_layers("last") == [1]
    assert a.describe()["embedding_layer"] == "blocks.4.mlp.l3"


def test_all_resolves_to_every_configured_name():
    names = ["blocks.4.mlp.l3", "blocks.20.mlp.l3", "blocks.28.mlp.l3"]
    a = _adapter(layer_names=names)
    assert a._resolve_layers("all") == [0, 1, 2]
    assert a.describe()["layer_id_to_name"] == {0: names[0], 1: names[1], 2: names[2]}


def test_int_and_list_index_the_name_list():
    a = _adapter(layer_names=["a.l3", "b.l3", "c.l3"])
    assert a._resolve_layers(1) == [1]
    assert a._resolve_layers(-1) == [2]
    assert a._resolve_layers([0, 2]) == [0, 2]


def test_out_of_range_layer_raises_not_silently_clamps():
    a = _adapter(layer_names=["a.l3"])
    with pytest.raises(IndexError):
        a._resolve_layers(3)


def test_bad_layer_request_raises():
    a = _adapter()
    with pytest.raises(ValueError, match="must be 'last', 'all', an int"):
        a._resolve_layers("penultimate")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 'last', 'all', an int"):
        a._resolve_layers([])


def test_duplicate_layer_names_rejected():
    """Two identical names would map one tensor to two ids — ambiguous, so refuse."""
    with pytest.raises(ValueError, match="duplicates"):
        _adapter(layer_names=["blocks.28.mlp.l3", "blocks.28.mlp.l3"])


def test_empty_or_non_string_layer_names_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        _adapter(layer_names=[])
    with pytest.raises(ValueError, match="non-empty list"):
        _adapter(layer_names=[28])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# arch-specific defaults (a wrong name is a silent wrong tensor)
# --------------------------------------------------------------------------- #


def test_default_layer_is_arch_specific():
    assert _adapter(model_name="evo2_7b_base").layer_names == ["blocks.28.mlp.l3"]
    assert _adapter(model_name="evo2_1b_base").layer_names == ["blocks.24.mlp.l3"]
    assert MODEL2LAYER["evo2_7b"] == "blocks.28.mlp.l3"


def test_unknown_model_without_explicit_layer_names_raises():
    """Guessing a layer name for an unknown arch is exactly how you embed the wrong tensor."""
    with pytest.raises(ValueError, match="no default embedding layer known"):
        _adapter(model_name="evo2_99b_imaginary")
    # ...but an explicit name is accepted for any model id.
    a = _adapter(model_name="evo2_99b_imaginary", layer_names=["blocks.1.mlp.l3"])
    assert a.layer_names == ["blocks.1.mlp.l3"]


# --------------------------------------------------------------------------- #
# model hash — layer_names MUST key the row
# --------------------------------------------------------------------------- #


def test_hash_is_deterministic_and_torch_free():
    assert _adapter().model_hash() == _adapter().model_hash()
    assert _adapter().model_hash().startswith("glmb:")


def test_hash_changes_with_the_embedding_tap():
    """The regression that would silently overwrite a leaderboard row.

    Same weights, same adapter, different tap ⇒ different feature extractor ⇒ it must key
    its own row. Without layer_names in the hashed config these collide and the second run
    replaces the first (the leaderboard is idempotent per model_hash x task).
    """
    h28 = _adapter(layer_names=["blocks.28.mlp.l3"]).model_hash()
    h31 = _adapter(layer_names=["blocks.31.mlp.l3"]).model_hash()
    assert h28 != h31


def test_hash_changes_with_weights_model_and_version():
    h = _adapter().model_hash()
    assert _adapter(weights_digest_value="deadbeef").model_hash() != h
    assert _adapter(model_name="evo2_7b").model_hash() != h  # different published weights
    a = _adapter()
    a.adapter_version = "9.9.9"
    assert a.model_hash() != h


# --------------------------------------------------------------------------- #
# spec → adapter
# --------------------------------------------------------------------------- #


def _spec(model: dict, extra: dict | None = None) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "adapter": "evo2-arc",
            "adapter_version": "0.1.0",
            "model": model,
            "weights_digest": {"strategy": "hf_revision", "value": DIGEST},
            "runner": {
                "backend": "local",
                "python_exe": sys.executable,
                "gpus": "0",
                "extra": extra or {},
            },
        }
    )


def test_from_spec_reads_layer_names_and_knobs():
    a = Evo2ArcAdapter.from_spec(
        _spec(
            {
                "model_name": "evo2_7b_base",
                "layer_names": ["blocks.20.mlp.l3", "blocks.28.mlp.l3"],
                "max_context": 8192,
                "score_prefix": "bos",
            },
            {"batch_size": 8, "token_budget": 16384},
        )
    )
    assert a.layer_names == ["blocks.20.mlp.l3", "blocks.28.mlp.l3"]
    assert a.batch_size == 8
    assert a.token_budget == 16384
    assert a.device == "auto"
    assert a._resolve_layers("last") == [1]


def test_from_spec_defaults_to_the_arch_recommended_layer():
    a = Evo2ArcAdapter.from_spec(_spec({"model_name": "evo2_7b_base"}))
    assert a.layer_names == ["blocks.28.mlp.l3"]


def test_from_spec_accepts_a_bare_string_layer_name():
    """A single name written unquoted-as-scalar is a natural spec typo; don't make it a crash."""
    a = Evo2ArcAdapter.from_spec(
        _spec({"model_name": "evo2_7b_base", "layer_names": "blocks.12.mlp.l3"})
    )
    assert a.layer_names == ["blocks.12.mlp.l3"]


def test_shipped_spec_is_valid_and_taps_the_residual_stream():
    """The committed spec must actually resolve — a typo here costs a full benchmark run."""
    import pathlib

    from glmbench.config.model_spec import load_model_spec

    p = (
        pathlib.Path(__file__).resolve().parents[2]
        / "specs" / "external" / "evo2-arc-7b-resid.yaml"
    )
    # load_model_spec, not model_validate: it expands ${GLMBENCH_EVO2_PYTHON:-...} exactly
    # as the CLI does, so an unset-var-with-no-default in the spec fails HERE, not mid-run.
    a = Evo2ArcAdapter.from_spec(load_model_spec(p))
    assert a.model_name == "evo2_7b_base"
    # block outputs (`blocks.N`), not the `blocks.N.mlp.l3` branch the adapter defaults to
    assert len(a.layer_names) == 17
    assert all(n.count(".") == 1 and n.startswith("blocks.") for n in a.layer_names)
    assert a.describe()["embedding_layer"] == "blocks.31"
    assert a.max_context == 8192


def test_docker_backend_rejected():
    """This route's whole point is one conda env; a docker spec is a copy-paste mistake."""
    with pytest.raises(ValueError, match="must be 'local'"):
        _adapter(backend="docker")


# --------------------------------------------------------------------------- #
# wire request shape (dry run — no model, no GPU)
# --------------------------------------------------------------------------- #


def test_dry_run_emits_the_evo2_arc_runner_command(capsys):
    a = _adapter(python_exe="/envs/evo2/bin/python", dry_run=True)
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.embed(SEQS)
    out = capsys.readouterr().out
    assert "/envs/evo2/bin/python -m glmbench.runners.evo2_arc_runner" in out


def test_embed_rejects_unsupported_pooling():
    a = _adapter()
    with pytest.raises(ValueError, match="pool 'mean' or 'max'"):
        a.embed(SEQS, pool="none")


def test_score_rejects_unsupported_reduction():
    a = _adapter()
    with pytest.raises(ValueError, match="reduction 'mean'|'sum'"):
        a.score_sequences(SEQS, reduction="median")


def test_describe_records_the_id_to_name_map():
    """Without this map the `layer_N_*` leaderboard columns are unattributable."""
    d = _adapter(layer_names=["blocks.20.mlp.l3", "blocks.28.mlp.l3"]).describe()
    assert d["name"] == "evo2-arc"
    assert d["capabilities"] == ["embedding", "logprob_embedding", "sequence_loglikelihood"]
    assert d["layer_id_to_name"] == {0: "blocks.20.mlp.l3", 1: "blocks.28.mlp.l3"}
    assert d["arch"]["layer_names"] == ["blocks.20.mlp.l3", "blocks.28.mlp.l3"]
    assert d["model_name"] == "evo2_7b_base"


def test_registry_resolves_the_adapter():
    from glmbench import registry

    assert registry.resolve("adapter", "evo2-arc") is Evo2ArcAdapter

