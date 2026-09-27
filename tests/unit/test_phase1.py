"""Adapter ABC, capabilities, hashing, wire protocol.

Gate: capability negotiation + wire round trip on the EchoAdapter pass; ``model_hash`` is
stable & weight/config/version-sensitive; core stays torch-free. These tests are model-free
(no torch, no GPU) — the EchoAdapter shells out to the deterministic echo runner.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from glmbench import registry
from glmbench.adapters import (
    Capability,
    CapabilityNotImplemented,
    DockerRunner,
    EchoAdapter,
    EmbeddingResult,
    ModelAdapter,
    Request,
    Response,
    WireVersionError,
    canonical_config_json,
    compute_weights_digest,
    missing_capability_methods,
    model_hash,
)
from glmbench.adapters import wire as wiremod

# --- capability gating -----------------------------------------------------


def test_echo_reports_declared_capabilities_only():
    a = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    assert a.capabilities() == frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})


def test_calling_undeclared_capability_raises():
    a = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    with pytest.raises(CapabilityNotImplemented) as exc:
        a.embed(["ACGT"])
    assert exc.value.capability is Capability.EMBEDDING
    assert exc.value.adapter_name == "echo"


def test_base_default_methods_raise():
    class Bare(ModelAdapter):
        name = "bare"
        adapter_version = "0.0.0"

        def model_hash(self) -> str:
            return "glmb:bare"

        def describe(self) -> dict:
            return {}

    bare = Bare()
    assert bare.capabilities() == frozenset()
    for call in (
        lambda: bare.tokenize(["A"]),
        lambda: bare.score_sequences(["A"]),
        lambda: bare.per_token_logprobs(["A"]),
        lambda: bare.embed(["A"]),
    ):
        with pytest.raises(CapabilityNotImplemented):
            call()


def test_abc_cannot_instantiate_without_required_methods():
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


# --- declared-vs-overridden consistency -------------------------------------


def test_every_registered_adapter_is_consistent():
    """For every registered adapter, each declared capability's method is overridden."""
    names = registry.list("adapter")
    assert "echo" in names
    for name in names:
        cls = registry.resolve("adapter", name)
        missing = missing_capability_methods(cls)
        assert not missing, f"adapter '{name}' declares but does not implement: {missing}"


def test_consistency_check_catches_half_built_adapter():
    class HalfBuilt(ModelAdapter):
        name = "half"
        adapter_version = "0.0.0"
        CAPABILITIES = frozenset({Capability.EMBEDDING})  # declared but not overridden

        def model_hash(self) -> str:
            return "x"

        def describe(self) -> dict:
            return {}

    assert missing_capability_methods(HalfBuilt) == [Capability.EMBEDDING]


# --- model hash ------------------------------------------------------------


def _hash(weights="w", config=None, name="echo", version="0.1.0"):
    return model_hash(
        weights_digest=weights,
        config=config or {"a": 1, "b": 2},
        adapter_name=name,
        adapter_version=version,
    )


def test_model_hash_is_deterministic_and_prefixed():
    h1 = _hash()
    h2 = _hash()
    assert h1 == h2
    assert h1.startswith("glmb:")
    assert len(h1) == len("glmb:") + 24


def test_model_hash_changes_with_each_input():
    base = _hash()
    assert _hash(weights="other") != base
    assert _hash(config={"a": 1, "b": 3}) != base
    assert _hash(version="0.2.0") != base
    assert _hash(name="loam-hf") != base


def test_canonical_config_is_key_order_independent():
    assert canonical_config_json({"b": 2, "a": 1}) == canonical_config_json({"a": 1, "b": 2})
    assert _hash(config={"b": 2, "a": 1}) == _hash(config={"a": 1, "b": 2})


def test_model_hash_needs_no_model():
    # EchoAdapter.model_hash() uses the 'declared' digest — no subprocess, no model.
    a = EchoAdapter(weights_value="vX", config={"k": 1})
    b = EchoAdapter(weights_value="vY", config={"k": 1})
    assert a.model_hash() != b.model_hash()
    assert a.model_hash() == EchoAdapter(weights_value="vX", config={"k": 1}).model_hash()


# --- weights digest strategies ---------------------------------------------


def test_weights_digest_file_and_dir(tmp_path):
    f = tmp_path / "ckpt.bin"
    f.write_bytes(b"weights-bytes")
    d1 = compute_weights_digest("file_sha256", path=f)
    assert d1.startswith("file_sha256:")
    # editing the file changes the digest
    f.write_bytes(b"weights-bytes-2")
    assert compute_weights_digest("file_sha256", path=f) != d1

    ckptdir = tmp_path / "ckpt"
    ckptdir.mkdir()
    (ckptdir / "a.bin").write_bytes(b"a")
    (ckptdir / "b.bin").write_bytes(b"b")
    dd = compute_weights_digest("file_sha256", path=ckptdir)
    assert dd.startswith("file_sha256:")
    (ckptdir / "b.bin").write_bytes(b"b-changed")
    assert compute_weights_digest("file_sha256", path=ckptdir) != dd


def test_weights_digest_cache_sidecar(tmp_path):
    f = tmp_path / "ckpt.bin"
    f.write_bytes(b"abc")
    cache = tmp_path / "ckpt.sha256"
    d1 = compute_weights_digest("file_sha256", path=f, cache_path=cache)
    assert cache.exists()
    # mutate the file but keep the cache → cached value is returned (computed once)
    f.write_bytes(b"different")
    d2 = compute_weights_digest("file_sha256", path=f, cache_path=cache)
    assert d1 == d2


def test_weights_digest_hf_and_declared():
    assert compute_weights_digest("hf_revision", value="abc123") == "hf:abc123"
    assert compute_weights_digest("declared", value="zzz") == "declared:zzz"
    with pytest.raises(ValueError):
        compute_weights_digest("hf_revision")
    with pytest.raises(ValueError):
        compute_weights_digest("bogus", value="x")


# --- wire protocol round trip ----------------------------------------------


def test_request_response_json_round_trip(tmp_path):
    req = Request(
        op="score_sequences",
        params={"reduction": "mean"},
        input_path=str(tmp_path / "in.jsonl"),
        output_path=str(tmp_path / "out.npz"),
    )
    req2 = Request.from_json(req.to_json())
    assert req2 == req

    resp = Response(op="score_sequences", status="ok", output_path=str(tmp_path / "out.npz"))
    resp2 = Response.from_json(resp.to_json())
    assert resp2 == resp


def test_wire_version_mismatch_is_loud(tmp_path):
    req = Request(op="tokenize", params={}, input_path="i", output_path="o")
    bad = req.to_json().replace(f'"{wiremod.WIRE_VERSION}"', '"99.0"')
    with pytest.raises(WireVersionError):
        Request.from_json(bad)


def test_unknown_op_is_loud():
    with pytest.raises(wiremod.WireProtocolError):
        Request.from_json('{"op":"nope","params":{},"input_path":"i","output_path":"o",'
                          f'"wire_version":"{wiremod.WIRE_VERSION}"}}')


def test_jsonl_sequence_round_trip(tmp_path):
    p = tmp_path / "seqs.jsonl"
    seqs = ["ACGT", "TTTT", ""]
    wiremod.write_sequences_jsonl(p, seqs)
    ids, back = wiremod.read_sequences_jsonl(p)
    assert back == seqs
    assert ids == ["0", "1", "2"]


def test_npz_arrays_survive_byte_exact(tmp_path):
    p = tmp_path / "a.npz"
    scores = np.array([0.1, -2.5, 3.0], dtype=np.float64)
    wiremod.write_arrays_npz(p, {"scores": scores})
    got = wiremod.read_arrays_npz(p)["scores"]
    assert np.array_equal(got, scores)
    assert got.dtype == scores.dtype


def test_ragged_npz_round_trip(tmp_path):
    p = tmp_path / "r.npz"
    arrs = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0, 6.0])]
    wiremod.write_ragged_npz(p, arrs)
    back = wiremod.read_ragged_npz(p)
    assert len(back) == 3
    for a, b in zip(arrs, back, strict=True):
        assert np.array_equal(a, b)


# --- variant JSONL round trip (masked-marginal LLR wire format) -------------


def test_variant_jsonl_round_trip(tmp_path):
    p = tmp_path / "variants.jsonl"
    items = [
        {"reference": "ACGTACGT", "mutations": [[2, "G", "A"]]},
        {"reference": "ACGTACGT", "mutations": [[1, "C", "T"], [6, "G", "A"]]},
    ]
    wiremod.write_variant_jsonl(p, items)
    ids, back = wiremod.read_variant_jsonl(p)
    assert ids == ["0", "1"]
    assert back[0] == {"reference": "ACGTACGT", "mutations": [[2, "G", "A"]]}
    assert back[1]["mutations"] == [[1, "C", "T"], [6, "G", "A"]]
    # positions come back as ints, bases as str
    assert isinstance(back[1]["mutations"][0][0], int)


def test_variant_jsonl_custom_ids(tmp_path):
    p = tmp_path / "v.jsonl"
    items = [{"reference": "ACGT", "mutations": [[0, "A", "C"]]}]
    wiremod.write_variant_jsonl(p, items, ids=["myvar"])
    ids, back = wiremod.read_variant_jsonl(p)
    assert ids == ["myvar"]


def test_variant_jsonl_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.jsonl"
    with pytest.raises(wiremod.WireProtocolError):
        wiremod.write_variant_jsonl(p, [{"reference": "ACGT"}])  # no 'mutations'


def test_score_variant_llr_op_is_accepted_by_request():
    """The new op is a valid wire op; bad ops are still loudly rejected."""
    req = Request(
        op="score_variant_llr", params={}, input_path="i", output_path="o"
    )
    assert Request.from_json(req.to_json()).op == "score_variant_llr"
    assert "score_variant_llr" in wiremod.WIRE_OPS


# --- DockerRunner dry-run command ------------------------------------------


def test_docker_runner_dry_run_command():
    r = DockerRunner(
        image="example.org/glm-runner:1.0",
        runner_module="glmbench.runners.echo_runner",
        python_exe="python",
        gpus="all",
        dry_run=True,
    )
    cmd = r.build_command("/scratch/abc/request.json", "/scratch/abc")
    assert cmd == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        "/scratch/abc:/scratch/abc",
        "example.org/glm-runner:1.0",
        "python",
        "-m",
        "glmbench.runners.echo_runner",
        "/scratch/abc/request.json",
    ]


def test_docker_dry_run_execute_emits_and_skips(capsys, tmp_path):
    r = DockerRunner(
        image="img:1", runner_module="glmbench.runners.echo_runner", dry_run=True
    )
    result = r.execute("score_sequences", {"reduction": "mean"}, ["ACGT"])
    out = capsys.readouterr().out
    assert "docker run --rm" in out
    assert result.response.status == "dry_run"
    assert result.payload == {}


# --- EmbeddingResult contract ----------------------------------------------


def test_embedding_result_pool_none_requires_spans():
    arr = np.zeros((3, 1, 8))
    with pytest.raises(ValueError, match="token_spans"):
        EmbeddingResult(arrays=[arr], layers=[0], pool="none", token_spans=None, embedding_dim=8)
    # with spans it's fine
    ok = EmbeddingResult(
        arrays=[arr], layers=[0], pool="none",
        token_spans=[[(0, 1), (1, 2), (2, 3)]], embedding_dim=8,
    )
    assert ok.pool == "none"


# --- EchoAdapter end-to-end through the subprocess wire boundary -----------


@pytest.fixture
def full_echo(tmp_path):
    return EchoAdapter(python_exe=sys.executable, scratch_dir=str(tmp_path))


def test_echo_score_sequences_deterministic(full_echo):
    seqs = ["ACGTACGT", "TTTTAAAA", "GG"]
    s1 = full_echo.score_sequences(seqs)
    s2 = full_echo.score_sequences(seqs)
    assert s1 == s2
    assert len(s1) == 3
    assert all(isinstance(x, float) for x in s1)
    # mean vs sum differ
    assert full_echo.score_sequences(seqs, reduction="sum") != s1


def test_echo_tokenize_round_trips(full_echo):
    toks = full_echo.tokenize(["ACGT"])
    assert toks == [[ord(c) for c in "ACGT"]]


def test_echo_per_token_logprobs(full_echo):
    seqs = ["ACGT", "TT"]
    pt = full_echo.per_token_logprobs(seqs)
    assert len(pt) == 2
    assert pt[0].shape == (4,)
    assert pt[1].shape == (2,)


# --- LOGPROB_EMBEDDING (the conditional-log-prob surprisal embedding) --------


def test_echo_logprob_embedding_round_trip(full_echo):
    """list[np.ndarray], one per input, length == nt count, input order, finite float64."""
    seqs = ["ACGTACGT", "TT", "GGGCC"]
    z = full_echo.logprob_embedding(seqs)
    assert isinstance(z, list) and len(z) == len(seqs)
    for seq, arr in zip(seqs, z, strict=True):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (len(seq),)  # one log-prob per nucleotide
        assert np.isfinite(arr).all()
    # deterministic + input order preserved
    z2 = full_echo.logprob_embedding(seqs)
    for a, b in zip(z, z2, strict=True):
        assert np.array_equal(a, b)


def test_echo_logprob_embedding_mean_equals_score(full_echo):
    """The load-bearing identity: mean(z) == score_sequences(mean) per sequence."""
    seqs = ["ACGTACGT", "TTTTAAAA", "GG"]
    z = full_echo.logprob_embedding(seqs)
    mean_scores = full_echo.score_sequences(seqs, reduction="mean")
    for arr, sc in zip(z, mean_scores, strict=True):
        np.testing.assert_allclose(float(arr.mean()), sc, rtol=0, atol=1e-12)


def test_logprob_embedding_negotiation_is_na():
    """An adapter without LOGPROB_EMBEDDING fails closed naming the missing capability."""
    a = EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    assert Capability.LOGPROB_EMBEDDING not in a.capabilities()
    with pytest.raises(CapabilityNotImplemented) as exc:
        a.logprob_embedding(["ACGT"])
    assert exc.value.capability is Capability.LOGPROB_EMBEDDING


def test_consistency_check_catches_half_built_logprob_embedding():
    class HalfBuilt(ModelAdapter):
        name = "half-lpe"
        adapter_version = "0.0.0"
        CAPABILITIES = frozenset({Capability.LOGPROB_EMBEDDING})  # declared, not overridden

        def model_hash(self) -> str:
            return "x"

        def describe(self) -> dict:
            return {}

    assert missing_capability_methods(HalfBuilt) == [Capability.LOGPROB_EMBEDDING]


def test_echo_embed_pooled(full_echo):
    res = full_echo.embed(["ACGT", "TTTT"], pool="mean", layers="last")
    assert isinstance(res, EmbeddingResult)
    assert res.pool == "mean"
    assert res.token_spans is None
    assert len(res.arrays) == 2
    assert res.arrays[0].shape == (1, res.embedding_dim)


def test_echo_embed_none_returns_token_spans(full_echo):
    res = full_echo.embed(["ACGT"], pool="none", layers="all")
    assert res.pool == "none"
    assert res.token_spans is not None
    # spans cover the full nucleotide string (echo = 1 bp/token)
    spans = res.token_spans[0]
    assert spans == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert res.arrays[0].shape[0] == 4  # n_tokens axis


def test_echo_model_hash_via_subprocess_free(full_echo):
    h = full_echo.model_hash()
    assert h.startswith("glmb:")
    assert full_echo.describe()["name"] == "echo"
