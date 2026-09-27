"""Evo 1.5 — EvoAdapter + evo_runner (StripedHyena, byte-level, trust_remote_code).

Unlike GenomeOcean, a *tiny random* stand-in model can't run on CPU: StripedHyena's
``AttentionBlock`` instantiates ``flash_attn``'s ``MHA`` with no SDPA fallback (needs CUDA +
a source-built flash-attn). So these unit tests exercise everything that does **not** need a
real forward pass — the whole torch-free adapter contract plus the byte-tokenizer and the
``tokenize`` wire op (which returns before any model load, so it round-trips through a real
subprocess on CPU with no torch/flash-attn):

- declared-vs-overridden capability consistency (fail-closed) — the full single-nt set;
- ``model_hash`` deterministic + sensitive to weights digest / config / adapter version,
  computable without a forward (torch-free);
- ``from_spec`` (trust_remote_code + code_revision defaults; checkpoint required); dry-run
  command emission; ``describe`` echo;
- the byte mapping (``id = clamp(ord, 32, 511)``, 1 char/token, ``max_tokens`` cap) both as a
  direct helper and round-tripped through the real ``tokenize`` subprocess;
- the response guards (NaN / length drift) via fake backends.

The real-model forward path (score > shuffle, mean(z)==score(mean), hooked embeddings) needs
a GPU and the real checkpoint, so it is not part of this suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from glmbench.adapters.base import (
    Capability,
    CapabilityNotImplemented,
    missing_capability_methods,
)
from glmbench.adapters.evo import DEFAULT_CODE_REVISION, EvoAdapter
from glmbench.config.model_spec import ModelSpec

SEQS = ["ATGACGTACGTACGT", "ACGTACGTACGTACGTACGTACGT", "TTGACAGCTAGCTCAG"]

# A minimal StripedHyena config.json (the arch fields EvoAdapter lifts into the model hash).
_TINY_CONFIG = {
    "architectures": ["StripedHyenaModelForCausalLM"],
    "model_type": "stripedhyena",
    "num_layers": 4,
    "hidden_size": 64,
    "inner_mlp_size": 128,
    "num_attention_heads": 4,
    "num_filters": 64,
    "attn_layer_idxs": [2],
    "hyena_layer_idxs": [0, 1, 3],
    "vocab_size": 512,
    "max_seqlen": 8192,
    "tie_embeddings": True,
    "torch_dtype": "bfloat16",
}


def _write_config(dir_path: Path, **overrides) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    cfg = {**_TINY_CONFIG, **overrides}
    (dir_path / "config.json").write_text(json.dumps(cfg))
    return dir_path


@pytest.fixture
def ckpt_dir(tmp_path):
    return _write_config(tmp_path / "evo-tiny")


@pytest.fixture
def adapter(ckpt_dir):
    return EvoAdapter(
        checkpoint=str(ckpt_dir),
        revision="rev-abc",
        python_exe=sys.executable,
        device="cpu",
        weights_digest_strategy="hf_revision",
        weights_digest_value="rev-abc",
    )


# --------------------------------------------------------------------------- #
# capability contract (fail-closed)
# --------------------------------------------------------------------------- #


def test_capabilities_are_the_single_nt_causal_set():
    assert frozenset(
        {
            Capability.TOKENIZE,
            Capability.SEQUENCE_LOGLIKELIHOOD,
            Capability.PER_TOKEN_LOGPROBS,
            Capability.EMBEDDING,
            Capability.LOGPROB_EMBEDDING,
        }
    ) == EvoAdapter.CAPABILITIES
    assert missing_capability_methods(EvoAdapter) == []
    # MASKED_MARGINAL_LLR is N/A for a causal LM — not declared.
    assert Capability.MASKED_MARGINAL_LLR not in EvoAdapter.CAPABILITIES


def test_undeclared_capability_raises(adapter):
    with pytest.raises(CapabilityNotImplemented):
        adapter.score_variant_llr([{"reference": "ACGT", "mutations": [[0, "A", "C"]]}])


def test_unknown_backend_raises(ckpt_dir):
    with pytest.raises(ValueError, match="local' or 'docker'"):
        EvoAdapter(checkpoint=str(ckpt_dir), backend="slurm")


# --------------------------------------------------------------------------- #
# model hash — torch-free, stable, sensitive
# --------------------------------------------------------------------------- #


def test_model_hash_no_forward_stable_and_sensitive(tmp_path):
    d = _write_config(tmp_path / "a")
    a = EvoAdapter(
        checkpoint=str(d), revision="r1", weights_digest_strategy="hf_revision",
        weights_digest_value="r1", python_exe=sys.executable, device="cpu",
    )
    h1, h2 = a.model_hash(), a.model_hash()
    assert h1 == h2 and h1.startswith("glmb:")

    # different weights digest (different revision) → different hash
    hb = EvoAdapter(
        checkpoint=str(d), revision="r2", weights_digest_strategy="hf_revision",
        weights_digest_value="r2", python_exe=sys.executable, device="cpu",
    ).model_hash()
    assert hb != h1

    # different arch (config.json num_layers) → different hash
    _write_config(d, num_layers=8)
    hc = EvoAdapter(
        checkpoint=str(d), revision="r1", weights_digest_strategy="hf_revision",
        weights_digest_value="r1", python_exe=sys.executable, device="cpu",
    ).model_hash()
    assert hc != h1

    # different adapter version → different hash
    a.adapter_version = "9.9.9"
    assert a.model_hash() != h1


def test_describe_echoes_config(adapter):
    d = adapter.describe()
    assert d["name"] == "evo"
    assert d["capabilities"] == [
        "embedding", "logprob_embedding", "per_token_logprobs",
        "sequence_loglikelihood", "tokenize",
    ]
    assert d["backend"] == "local"
    assert d["embedding_dim"] == 64
    assert d["n_layers"] == 4
    assert d["max_context"] == 8192
    assert d["trust_remote_code"] is True
    assert d["code_revision"] == DEFAULT_CODE_REVISION
    assert d["eod_token_id"] == 0
    assert d["arch"]["family"] == "evo"
    assert d["arch"]["model_type"] == "stripedhyena"


# --------------------------------------------------------------------------- #
# spec → adapter
# --------------------------------------------------------------------------- #


def test_from_spec_defaults_trust_and_code_revision(ckpt_dir):
    spec = ModelSpec.model_validate(
        {
            "adapter": "evo",
            "adapter_version": "0.1.0",
            "model": {"checkpoint": str(ckpt_dir), "revision": "rev-abc"},
            "weights_digest": {"strategy": "hf_revision", "value": "rev-abc"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 2}},
        }
    )
    a = EvoAdapter.from_spec(spec)
    assert a.backend == "local"
    assert a.device == "cpu"
    assert a.batch_size == 2
    assert a.trust_remote_code is True          # forced default (custom arch)
    assert a.code_revision == DEFAULT_CODE_REVISION
    assert a.max_context == 8192


def test_token_budget_reaches_the_runner_request(ckpt_dir):
    """`extra.token_budget` must survive spec -> adapter -> runner params.

    Regression: the runner has always read ``params["token_budget"]``, but the adapter
    never emitted it, so setting it in a spec silently did nothing and a fixed
    ``batch_size`` OOM'd on the long tail of the length-sorted pass. A knob that is
    read but never sent is worse than a missing one -- it reads as configured.
    """
    spec = ModelSpec.model_validate(
        {
            "adapter": "evo",
            "adapter_version": "0.1.0",
            "model": {"checkpoint": str(ckpt_dir), "revision": "rev-abc"},
            "weights_digest": {"strategy": "hf_revision", "value": "rev-abc"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 64, "token_budget": 32768}},
        }
    )
    a = EvoAdapter.from_spec(spec)
    assert a.token_budget == 32768
    assert a._common_params()["token_budget"] == 32768
    assert a.describe()["token_budget"] == 32768


def test_token_budget_defaults_to_none_when_unset(ckpt_dir):
    """Absent/zero token_budget -> None, i.e. batch_size alone applies (prior behaviour)."""
    spec = ModelSpec.model_validate(
        {
            "adapter": "evo",
            "adapter_version": "0.1.0",
            "model": {"checkpoint": str(ckpt_dir), "revision": "rev-abc"},
            "weights_digest": {"strategy": "hf_revision", "value": "rev-abc"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 4}},
        }
    )
    a = EvoAdapter.from_spec(spec)
    assert a.token_budget is None
    assert a._common_params()["token_budget"] is None


def test_token_budget_does_not_change_model_hash(ckpt_dir):
    """Batching is a throughput knob, not model identity: the hash must not move.

    If it did, retuning the batch size would fork the leaderboard row.
    """
    def _spec(extra):
        return ModelSpec.model_validate(
            {
                "adapter": "evo",
                "adapter_version": "0.1.0",
                "model": {"checkpoint": str(ckpt_dir), "revision": "rev-abc"},
                "weights_digest": {"strategy": "hf_revision", "value": "rev-abc"},
                "runner": {"backend": "local", "python_exe": sys.executable,
                           "gpus": "cpu", "extra": extra},
            }
        )

    a = EvoAdapter.from_spec(_spec({"batch_size": 4}))
    b = EvoAdapter.from_spec(_spec({"batch_size": 64, "token_budget": 32768}))
    assert a.model_hash() == b.model_hash()


def test_from_spec_requires_checkpoint():
    spec = ModelSpec.model_validate(
        {"adapter": "evo", "adapter_version": "0.1.0", "model": {}}
    )
    with pytest.raises(ValueError, match="must set 'checkpoint'"):
        EvoAdapter.from_spec(spec)


def test_dry_run_emits_local_command(ckpt_dir, capsys):
    a = EvoAdapter(
        checkpoint=str(ckpt_dir), python_exe="/envs/loam/bin/python",
        weights_digest_strategy="hf_revision", weights_digest_value="r", dry_run=True,
    )
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.score_sequences(SEQS)
    out = capsys.readouterr().out
    assert "/envs/loam/bin/python -m glmbench.runners.evo_runner" in out


# --------------------------------------------------------------------------- #
# byte tokenizer — direct helper + real subprocess round-trip (no model load)
# --------------------------------------------------------------------------- #


def test_byte_tokenize_helper():
    from glmbench.runners.evo_runner import _byte_tokenize

    assert _byte_tokenize("ACGT", None) == [ord("A"), ord("C"), ord("G"), ord("T")]
    assert _byte_tokenize("ACGT", None) == [65, 67, 71, 84]
    # 1 char/token
    assert len(_byte_tokenize("ACGTACGT", None)) == 8
    # max_tokens cap
    assert _byte_tokenize("ACGTACGT", 3) == [65, 67, 71]


def test_tokenize_round_trip_through_subprocess(adapter):
    """tokenize returns before any model load ⇒ a real subprocess round-trip on CPU."""
    toks = adapter.tokenize(SEQS)
    assert len(toks) == len(SEQS)
    for seq, t in zip(SEQS, toks, strict=True):
        assert t == [min(max(ord(c), 32), 511) for c in seq]  # 1 char/token, clamped ord
        assert len(t) == len(seq)


# --------------------------------------------------------------------------- #
# response guards — NaN / length drift trip loudly (fake backends, no torch)
# --------------------------------------------------------------------------- #


def _run_result(payload):
    from glmbench.adapters.runner_backend import RunResult
    from glmbench.adapters.wire import Response

    return RunResult(
        response=Response(op="x", status="ok", output_path="x.npz"),
        payload=payload,
        scratch_dir="x",
    )


def test_score_nan_guard_trips(adapter):
    from glmbench.adapters.wire import WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            return _run_result({"scores": np.full(len(sequences), np.nan, dtype=np.float64)})

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.score_sequences(SEQS)


def test_logprob_embedding_length_guard_trips(adapter):
    from glmbench.adapters.wire import WireProtocolError

    class _BadLenBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            payload = {"n": np.asarray(len(sequences), dtype=np.int64)}
            for i, _s in enumerate(sequences):
                payload[f"arr_{i}"] = np.zeros(3, dtype=np.float64)  # wrong length
            return _run_result(payload)

    adapter._backend = _BadLenBackend()
    with pytest.raises(WireProtocolError, match="nt"):
        adapter.logprob_embedding(SEQS)


def test_logprob_embedding_nan_guard_trips(adapter):
    from glmbench.adapters.wire import WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            payload = {"n": np.asarray(len(sequences), dtype=np.int64)}
            for i, s in enumerate(sequences):
                payload[f"arr_{i}"] = np.full(len(s), np.nan, dtype=np.float64)
            return _run_result(payload)

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="non-finite"):
        adapter.logprob_embedding(SEQS)


def test_embed_nan_guard_trips(adapter):
    from glmbench.adapters.wire import WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            stacked = np.full((len(sequences), 1, 64), np.nan, dtype=np.float32)
            return _run_result(
                {
                    "arrays": stacked,
                    "layer_ids": np.asarray([4], dtype=np.int64),
                    "embedding_dim": np.asarray(64, dtype=np.int64),
                }
            )

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.embed(SEQS, layers="last", pool="mean")


def test_per_token_nan_guard_trips(adapter):
    from glmbench.adapters.wire import WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            payload = {"n": np.asarray(len(sequences), dtype=np.int64)}
            for i, s in enumerate(sequences):
                payload[f"arr_{i}"] = np.full(len(s), np.nan, dtype=np.float64)
            return _run_result(payload)

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="non-finite"):
        adapter.per_token_logprobs(SEQS)
