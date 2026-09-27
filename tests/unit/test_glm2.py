"""GLM2Adapter + glm2_runner — the mixed-modality masked-LM ENCODER path (embeddings-only).

CPU gate (no GPU). A **tiny real gLM2** (`depth=2, dim=32, heads=2`) is built from the
cached remote `gLM2Config` + `gLM2Model` (`AutoModel`, `trust_remote_code=True`) with the
real `gLM2Tokenizer`, `save_pretrained` to a temp dir, and driven through
`LocalSubprocessRunner` with `python_exe=sys.executable`. Needs einops, and the Hub for the
remote code: those tests are marked ``network`` (see ``conftest.py``).

The point of this suite is the gLM2-specific behavior that differs from every other (causal)
adapter:

- capabilities are ``{TOKENIZE, EMBEDDING}`` only; ``SEQUENCE_LOGLIKELIHOOD`` /
  ``PER_TOKEN_LOGPROBS`` / ``LOGPROB_EMBEDDING`` are **N/A** (MLM encoder) — fail-closed;
- **R1 nucleotide case-folding**: uppercase ``ACGT`` is lowercased to nucleotide ids
  (not read as amino acids) — tokenize + embed are case-invariant;
- **R2 strand marker** is prepended (``<+>``) and is the leading token id;
- **R3 pooling** excludes the marker by default (``pool_include_special`` flips it);
- **R4 token_spans** are unit-width, contiguous, cover exactly ``len(seq)`` (marker excluded);
- robust **layer indexing** against the actual ``hidden_states`` length (gLM2 returns ``depth``);
- embeddings are batch-size invariant (right-pad + masked pooling); NaN guard trips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("einops")

from glmbench.adapters.base import (  # noqa: E402
    Capability,
    CapabilityNotImplemented,
    missing_capability_methods,
)
from glmbench.adapters.glm2 import GLM2Adapter  # noqa: E402
from glmbench.config.model_spec import ModelSpec  # noqa: E402

REPO = "tattabio/gLM2_150M"  # source of the real config + tokenizer (downsized for the test)
# Uppercase DNA on purpose — the runner must lowercase it to nucleotide ids (R1).
SEQS = [
    "ACGTACGTACGTACGTTTGGCCAAACGTACGT",
    "TTGGCCAATTGGCCAAACGTACGTTTGGCCAA",
    "ACGTTTGGACGTACGTCCAAACGTACGTACGT",
]
# Probe-confirmed real vocab ids (stable across the downsizing — only depth/dim/heads change).
MARKER_PLUS_ID = 33
NT_IDS = {"a": 29, "t": 30, "c": 31, "g": 32}


def _build_tiny_glm2_dir(dir_path: Path, *, seed: int = 0) -> Path:
    """A tiny random gLM2 (`AutoModel` = gLM2Model) + the real gLM2Tokenizer, saved to disk.

    Pulls the real ``gLM2Config`` (carrying ``auto_map`` to the bundled modeling code) and the
    real tokenizer, downsizes the arch to a couple of tiny layers, random-inits, and
    ``save_pretrained`` (which copies the remote ``*.py`` so the runner can reload with
    ``trust_remote_code=True``). Skips if the remote code can't be fetched.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - offline / hub down → skip, not fail
        pytest.skip(f"gLM2 remote code/tokenizer unavailable ({type(e).__name__}: {e})")
    cfg.depth = 2
    cfg.dim = 32
    cfg.heads = 2
    torch.manual_seed(seed)
    model = AutoModel.from_config(cfg, trust_remote_code=True)
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    if not (dir_path / "modeling_glm2.py").exists():  # belt-and-suspenders for the runner reload
        _copy_remote_code(REPO, dir_path)
    return dir_path


def _copy_remote_code(repo: str, dir_path: Path) -> None:
    """Fallback: copy the cached remote ``*.py`` next to the checkpoint if save didn't."""
    import shutil

    from transformers.utils import HF_MODULES_CACHE

    src = Path(HF_MODULES_CACHE) / "transformers_modules" / repo.replace("/", "--")
    cands = list(src.rglob("*.py")) if src.exists() else []
    for p in cands:
        shutil.copy(p, dir_path / p.name)


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory, hub_access):
    return _build_tiny_glm2_dir(tmp_path_factory.mktemp("glm2") / "glm2-tiny")


@pytest.fixture
def adapter(hf_dir):
    return GLM2Adapter(
        checkpoint=str(hf_dir),
        python_exe=sys.executable,
        device="cpu",
        dtype="float32",
        weights_digest_strategy="file_sha256",
    )


# --------------------------------------------------------------------------- #
# capability contract (fail-closed) — encoder ⇒ scoring caps are N/A
# --------------------------------------------------------------------------- #


def test_capabilities_are_tokenize_embedding_and_llr():
    """Encoder serves embeddings + masked-marginal LLR; causal scoring stays N/A."""
    assert (
        frozenset(
            {
                Capability.TOKENIZE,
                Capability.EMBEDDING,
                Capability.MASKED_MARGINAL_LLR,
            }
        )
        == GLM2Adapter.CAPABILITIES
    )
    for absent in (
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.LOGPROB_EMBEDDING,
    ):
        assert absent not in GLM2Adapter.CAPABILITIES
    # consistency: every declared capability (incl. the new one) is overridden.
    assert missing_capability_methods(GLM2Adapter) == []


def test_scoring_capabilities_are_na_fail_closed(adapter):
    """MLM encoder ⇒ no *causal* scoring: each causal op raises (clean first-class N/A)."""
    with pytest.raises(CapabilityNotImplemented):
        adapter.score_sequences(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.per_token_logprobs(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.logprob_embedding(SEQS)


def test_unknown_backend_raises(hf_dir):
    with pytest.raises(ValueError, match="local' or 'docker'"):
        GLM2Adapter(checkpoint=str(hf_dir), backend="slurm")


def test_bad_strand_prefix_raises(hf_dir):
    with pytest.raises(ValueError, match="strand_prefix"):
        GLM2Adapter(checkpoint=str(hf_dir), strand_prefix="fwd")


# --------------------------------------------------------------------------- #
# model hash — torch-free, stable, sensitive
# --------------------------------------------------------------------------- #


def test_model_hash_no_forward_stable_and_sensitive(tmp_path):
    a_dir = _build_tiny_glm2_dir(tmp_path / "a", seed=1)
    a = GLM2Adapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    )
    h1, h2 = a.model_hash(), a.model_hash()
    assert h1 == h2 and h1.startswith("glmb:")

    b_dir = _build_tiny_glm2_dir(tmp_path / "b", seed=2)
    hb = GLM2Adapter(
        checkpoint=str(b_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hb != h1  # different weights → different hash

    cfg_path = a_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    data["depth"] = int(data["depth"]) + 1
    cfg_path.write_text(json.dumps(data))
    hc = GLM2Adapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hc != h1  # arch field (depth) feeds the hash


def test_model_hash_hf_revision_no_path():
    """hf_revision keys the hash off the pinned commit — no checkpoint read needed."""
    a = GLM2Adapter(
        checkpoint="tattabio/gLM2_650M", revision="deadbeef",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    h = a.model_hash()
    assert h.startswith("glmb:")
    b = GLM2Adapter(
        checkpoint="tattabio/gLM2_650M", revision="feedface",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    assert b.model_hash() != h


def test_describe_echoes_config(adapter):
    d = adapter.describe()
    assert d["name"] == "glm2"
    assert d["capabilities"] == ["embedding", "masked_marginal_llr", "tokenize"]
    assert d["backend"] == "local"
    assert d["embedding_dim"] == 32 and d["n_layers"] == 2
    assert d["max_context"] == 4095 and d["max_tokens"] == 4096
    assert d["strand_prefix"] == "+" and d["pool_include_special"] is False
    assert d["trust_remote_code"] is True
    assert d["arch"]["family"] == "glm2" and d["arch"]["model_type"] == "gLM2"


# --------------------------------------------------------------------------- #
# spec → adapter + dry-run command
# --------------------------------------------------------------------------- #


def test_from_spec_builds_local_adapter(hf_dir):
    spec = ModelSpec.model_validate(
        {
            "adapter": "glm2",
            "adapter_version": "0.1.0",
            "model": {"checkpoint": str(hf_dir), "dtype": "float32", "max_context": 2048},
            "weights_digest": {"strategy": "file_sha256"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 2}},
        }
    )
    a = GLM2Adapter.from_spec(spec)
    assert a.backend == "local" and a.device == "cpu" and a.batch_size == 2
    assert a.max_context == 2048 and a.trust_remote_code is True
    res = a.embed(SEQS, layers="last", pool="mean")
    assert len(res.arrays) == len(SEQS)


def test_from_spec_requires_checkpoint():
    spec = ModelSpec.model_validate(
        {"adapter": "glm2", "adapter_version": "0.1.0", "model": {}}
    )
    with pytest.raises(ValueError, match="must set 'checkpoint'"):
        GLM2Adapter.from_spec(spec)


def test_dry_run_emits_local_command(hf_dir, capsys):
    a = GLM2Adapter(checkpoint=str(hf_dir), python_exe="/envs/loam/bin/python", dry_run=True)
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.embed(SEQS, layers="last", pool="mean")
    out = capsys.readouterr().out
    assert "/envs/loam/bin/python -m glmbench.runners.glm2_runner" in out


# --------------------------------------------------------------------------- #
# tokenize — R1 case-folding + R2 strand marker
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases_and_prepends_marker(adapter):
    toks = adapter.tokenize(["ACGT"])
    assert toks[0][0] == MARKER_PLUS_ID  # leading strand marker
    assert toks[0][1:] == [NT_IDS["a"], NT_IDS["c"], NT_IDS["g"], NT_IDS["t"]]  # nt ids, NOT AAs


def test_tokenize_case_invariant(adapter):
    assert adapter.tokenize(["ACGTACGT"]) == adapter.tokenize(["acgtacgt"])


def test_tokenize_length_is_seq_plus_marker(adapter):
    toks = adapter.tokenize(SEQS)
    for seq, t in zip(SEQS, toks, strict=True):
        assert len(t) == len(seq) + 1  # 1 nt/token + the leading marker


def test_tokenize_no_marker_when_strand_none(hf_dir):
    a = GLM2Adapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                    dtype="float32", weights_digest_strategy="file_sha256", strand_prefix="none")
    toks = a.tokenize(["ACGT"])
    assert toks[0] == [NT_IDS["a"], NT_IDS["c"], NT_IDS["g"], NT_IDS["t"]]  # no marker


# --------------------------------------------------------------------------- #
# embeddings — shapes, layers, case-invariance, spans, pooling, NaN guard
# --------------------------------------------------------------------------- #


def test_embed_pool_mean_shape(adapter):
    res = adapter.embed(SEQS, layers="last", pool="mean")
    assert res.pool == "mean" and res.token_spans is None
    assert res.layers == [1] and res.embedding_dim == 32  # depth=2 ⇒ last index 1
    for arr in res.arrays:
        assert arr.shape == (1, 32) and np.isfinite(arr).all()


def test_embed_layers_all_one_pass_and_distinct(adapter):
    res = adapter.embed(SEQS, layers="all", pool="mean")
    assert res.layers == [0, 1]  # gLM2 returns `depth` states (here 2), indexed against actual len
    for arr in res.arrays:
        assert arr.shape == (2, 32)
        assert not np.allclose(arr[0], arr[1]), "all layers identical"


def test_embed_case_invariant(adapter):
    """R1: uppercase input embeds identically to its lowercased form."""
    up = adapter.embed(["ACGTACGTTTGG"], layers="last", pool="mean")
    lo = adapter.embed(["acgtacgtttgg"], layers="last", pool="mean")
    np.testing.assert_allclose(up.arrays[0], lo.arrays[0], rtol=0, atol=1e-5)


def test_embed_pool_none_spans_cover_sequence_marker_excluded(adapter):
    """R4: 1 nt/token ⇒ unit-width contiguous spans covering exactly len(seq); no marker span."""
    res = adapter.embed(SEQS, layers="all", pool="none")
    assert res.pool == "none" and res.token_spans is not None
    for seq, arr, spans in zip(SEQS, res.arrays, res.token_spans, strict=True):
        assert len(spans) == len(seq)  # marker excluded → one row per nucleotide
        assert arr.shape == (len(seq), 2, 32)
        assert spans[0] == (0, 1) and spans[-1] == (len(seq) - 1, len(seq))
        for j, (a, b) in enumerate(spans):
            assert (a, b) == (j, j + 1)  # unit-width, contiguous, no gaps


def test_embed_pool_include_special_changes_mean(hf_dir):
    """R3: including vs excluding the strand marker changes the mean embedding."""
    incl = GLM2Adapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                       dtype="float32", weights_digest_strategy="file_sha256",
                       pool_include_special=True)
    excl = GLM2Adapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                       dtype="float32", weights_digest_strategy="file_sha256",
                       pool_include_special=False)
    a = incl.embed(SEQS, layers="last", pool="mean")
    b = excl.embed(SEQS, layers="last", pool="mean")
    assert not np.allclose(a.arrays[0], b.arrays[0])


def test_embed_batch_size_invariant(hf_dir):
    """Right-pad + masked content pooling ⇒ embeddings independent of batch size."""
    a1 = GLM2Adapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                     dtype="float32", weights_digest_strategy="file_sha256", batch_size=1)
    aN = GLM2Adapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                     dtype="float32", weights_digest_strategy="file_sha256", batch_size=8)
    e1 = a1.embed(SEQS, layers="last", pool="mean")
    eN = aN.embed(SEQS, layers="last", pool="mean")
    for x, y in zip(e1.arrays, eN.arrays, strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-5)


def test_embed_nan_guard_trips(adapter):
    from glmbench.adapters.runner_backend import RunResult
    from glmbench.adapters.wire import Response, WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            stacked = np.full((len(sequences), 1, 32), np.nan, dtype=np.float32)
            return RunResult(
                response=Response(op=op, status="ok", output_path="x.npz"),
                payload={
                    "arrays": stacked,
                    "layer_ids": np.asarray([1], dtype=np.int64),
                    "embedding_dim": np.asarray(32, dtype=np.int64),
                },
                scratch_dir="x",
            )

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.embed(SEQS, layers="last", pool="mean")


# --------------------------------------------------------------------------- #
# masked-marginal LLR (MASKED_MARGINAL_LLR) — the MLM variant-effect scoring path
# --------------------------------------------------------------------------- #
#
# These load gLM2ForMaskedLM (AutoModelForMaskedLM, for the logits head). A tiny model is
# built + saved WITH its head so loads are byte-identical across processes (the base-only
# `hf_dir` fixture would random-init the head per load → non-deterministic). All CPU.


def _build_tiny_glm2_mlm_dir(dir_path: Path, *, seed: int = 0) -> Path:
    """A tiny random gLM2ForMaskedLM (AutoModelForMaskedLM) + the real tokenizer, saved.

    Saves the masked-LM head weights so every reload is identical (determinism + caching
    tests). Skips if the remote code or the AutoModelForMaskedLM mapping is unavailable.
    """
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - offline / hub down → skip, not fail
        pytest.skip(f"gLM2 remote code/tokenizer unavailable ({type(e).__name__}: {e})")
    cfg.depth = 2
    cfg.dim = 32
    cfg.heads = 2
    torch.manual_seed(seed)
    try:
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - no MLM auto_map → skip the LLR suite
        pytest.skip(f"gLM2 has no AutoModelForMaskedLM mapping ({type(e).__name__}: {e})")
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    if not (dir_path / "modeling_glm2.py").exists():
        _copy_remote_code(REPO, dir_path)
    return dir_path


@pytest.fixture(scope="module")
def hf_dir_mlm(tmp_path_factory, hub_access):
    return _build_tiny_glm2_mlm_dir(tmp_path_factory.mktemp("glm2mlm") / "glm2-mlm-tiny")


@pytest.fixture
def adapter_llr(hf_dir_mlm):
    return GLM2Adapter(
        checkpoint=str(hf_dir_mlm),
        python_exe=sys.executable,
        device="cpu",
        dtype="float32",
        weights_digest_strategy="file_sha256",
    )


def _load_mlm_in_process(hf_dir_mlm: Path):
    """Load the tiny gLM2ForMaskedLM + tokenizer in-process for white-box logit checks."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(str(hf_dir_mlm), trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(
        str(hf_dir_mlm), trust_remote_code=True, padding_side="right"
    )
    return model, tok


REF = "acgtacgtacgtacgt"  # index 2='g', 5='c', 8='a' — used for reference-consistent WT bases


def test_llr_adapter_roundtrip_finite_and_1to1(adapter_llr):
    """Exit gate 3(e): adapter.score_variant_llr round-trips via the runner, finite, 1:1."""
    items = [
        {"reference": REF, "mutations": [[2, "G", "A"]]},
        {"reference": REF, "mutations": [[2, "G", "T"], [5, "C", "A"]]},
    ]
    scores = adapter_llr.score_variant_llr(items)
    assert len(scores) == len(items)
    assert all(np.isfinite(s) for s in scores)


def test_llr_logit_diff_equals_log_softmax_diff(hf_dir_mlm):
    """Exit gate 3(a): LLR == raw logit diff == log-softmax diff (normalizer cancels)."""
    from glmbench.runners.glm2_runner import _encode_ids, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    item = {"reference": REF, "mutations": [[2, "G", "A"]]}
    out = _score_variant_llr(
        model, tok, [item], device=device, marker_id=MARKER_PLUS_ID,
        batch_size=8, max_tokens=4096,
    )
    llr = float(out["scores"][0])

    ids = _encode_ids(tok, [REF], MARKER_PLUS_ID, max_tokens=4096)[0]
    t = 2 + 1  # marker offset = 1
    ids_masked = list(ids)
    ids_masked[t] = int(tok.mask_token_id)
    inp = torch.tensor([ids_masked], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0, t].float()
    mut_id = int(tok.convert_tokens_to_ids("a"))
    wt_id = int(tok.convert_tokens_to_ids("g"))
    logit_diff = float(logits[mut_id] - logits[wt_id])
    logsm = torch.log_softmax(logits, dim=-1)
    logsm_diff = float(logsm[mut_id] - logsm[wt_id])
    assert llr == pytest.approx(logit_diff, abs=1e-4)
    assert llr == pytest.approx(logsm_diff, abs=1e-4)


def test_llr_masks_right_token_under_marker_offset(hf_dir_mlm):
    """Exit gate 3(b): the masked token index is pos0 + marker_offset (marker shifts it)."""
    from glmbench.runners.glm2_runner import _encode_ids, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    item = {"reference": REF, "mutations": [[4, "A", "T"]]}

    # With a marker, the runner must mask token index 4+1=5; reproduce that exactly.
    with_marker = float(
        _score_variant_llr(
            model, tok, [item], device=device, marker_id=MARKER_PLUS_ID,
            batch_size=8, max_tokens=4096,
        )["scores"][0]
    )
    ids = _encode_ids(tok, [REF], MARKER_PLUS_ID, max_tokens=4096)[0]
    ids_m = list(ids)
    ids_m[5] = int(tok.mask_token_id)
    inp = torch.tensor([ids_m], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0, 5].float()
    expect = float(logits[int(tok.convert_tokens_to_ids("t"))] - logits[int(tok.convert_tokens_to_ids("a"))])
    assert with_marker == pytest.approx(expect, abs=1e-4)

    # Without a marker, the same nucleotide is at token index 4 (offset 0).
    no_marker = float(
        _score_variant_llr(
            model, tok, [item], device=device, marker_id=None,
            batch_size=8, max_tokens=4096,
        )["scores"][0]
    )
    ids_nm = _encode_ids(tok, [REF], None, max_tokens=4096)[0]
    ids_nm_m = list(ids_nm)
    ids_nm_m[4] = int(tok.mask_token_id)
    inp2 = torch.tensor([ids_nm_m], dtype=torch.long)
    with torch.inference_mode():
        logits2 = model(input_ids=inp2, attention_mask=torch.ones_like(inp2)).logits[0, 4].float()
    expect2 = float(logits2[int(tok.convert_tokens_to_ids("t"))] - logits2[int(tok.convert_tokens_to_ids("a"))])
    assert no_marker == pytest.approx(expect2, abs=1e-4)


def test_llr_multisite_equals_sum_of_single_sites(hf_dir_mlm):
    """Exit gate 3(c): multi-site LLR == sum of independent single-site LLRs."""
    from glmbench.runners.glm2_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    kw = dict(device=device, marker_id=MARKER_PLUS_ID, batch_size=8, max_tokens=4096)
    two = float(
        _score_variant_llr(
            model, tok, [{"reference": REF, "mutations": [[2, "G", "A"], [5, "C", "T"]]}], **kw
        )["scores"][0]
    )
    s1 = float(
        _score_variant_llr(model, tok, [{"reference": REF, "mutations": [[2, "G", "A"]]}], **kw)[
            "scores"
        ][0]
    )
    s2 = float(
        _score_variant_llr(model, tok, [{"reference": REF, "mutations": [[5, "C", "T"]]}], **kw)[
            "scores"
        ][0]
    )
    assert two == pytest.approx(s1 + s2, abs=1e-4)


def test_llr_favored_base_is_positive(hf_dir_mlm):
    """Exit gate 3(d): mutating to the model-favored base yields LLR ≥ 0 (strictly > if distinct)."""
    from glmbench.runners.glm2_runner import _encode_ids, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    ids = _encode_ids(tok, [REF], MARKER_PLUS_ID, max_tokens=4096)[0]
    t = 2 + 1
    ids_m = list(ids)
    ids_m[t] = int(tok.mask_token_id)
    inp = torch.tensor([ids_m], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0, t].float()
    nt_ids = {b: int(tok.convert_tokens_to_ids(b)) for b in "acgt"}
    fav = max(nt_ids, key=lambda b: float(logits[nt_ids[b]]))
    least = min(nt_ids, key=lambda b: float(logits[nt_ids[b]]))
    if fav == least:
        pytest.skip("degenerate logits (all nucleotide tokens equal)")
    s = float(
        _score_variant_llr(
            model, tok,
            [{"reference": REF, "mutations": [[2, least.upper(), fav.upper()]]}],
            device=device, marker_id=MARKER_PLUS_ID, batch_size=8, max_tokens=4096,
        )["scores"][0]
    )
    assert s > 0.0


def test_llr_caching_matches_naive_per_variant(hf_dir_mlm):
    """Exit gate 3(f): the cached (dedup-by-(ref,pos)) path equals a naive per-site path."""
    from glmbench.runners.glm2_runner import _encode_ids, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    items = [
        {"reference": REF, "mutations": [[2, "G", "A"]]},
        {"reference": REF, "mutations": [[2, "G", "T"], [5, "C", "A"]]},
        {"reference": REF, "mutations": [[8, "A", "C"]]},
    ]
    cached = _score_variant_llr(
        model, tok, items, device=device, marker_id=MARKER_PLUS_ID, batch_size=8, max_tokens=4096
    )["scores"]

    def _naive(item):
        ids = _encode_ids(tok, [item["reference"].lower()], MARKER_PLUS_ID, max_tokens=4096)[0]
        total = 0.0
        for pos0, wt, mut in item["mutations"]:
            t = int(pos0) + 1
            ids_m = list(ids)
            ids_m[t] = int(tok.mask_token_id)
            inp = torch.tensor([ids_m], dtype=torch.long)
            with torch.inference_mode():
                row = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0, t].float()
            total += float(row[int(tok.convert_tokens_to_ids(mut.lower()))]
                           - row[int(tok.convert_tokens_to_ids(wt.lower()))])
        return total

    naive = np.array([_naive(it) for it in items], dtype=np.float64)
    np.testing.assert_allclose(cached, naive, rtol=0, atol=1e-4)


def test_llr_deterministic(adapter_llr):
    """Determinism: identical inputs → identical LLR (inference_mode, fixed dtype)."""
    items = [{"reference": REF, "mutations": [[2, "G", "A"], [8, "A", "C"]]}]
    a = adapter_llr.score_variant_llr(items)
    b = adapter_llr.score_variant_llr(items)
    assert a == b


def test_llr_no_mask_token_fails_loud(hf_dir_mlm):
    """A tokenizer without a [MASK] token id fails loudly (no silent miscompute)."""
    from glmbench.runners.glm2_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)

    class _NoMask:
        def __getattr__(self, k):
            return getattr(tok, k)

        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        _score_variant_llr(
            model, _NoMask(), [{"reference": REF, "mutations": [[2, "G", "A"]]}],
            device=torch.device("cpu"), marker_id=MARKER_PLUS_ID, batch_size=8, max_tokens=4096,
        )


# --------------------------------------------------------------------------- #
# bucketing order gate
# --------------------------------------------------------------------------- #
#
# The runner length-buckets its batches and scatters results back by input index. A bad
# scatter is SILENT: every metric still computes, just on rows belonging to other sequences.
# These pin the invariant with a deliberately wide length spread, which is what makes the
# bucketer reorder aggressively (the default batch_size=8 also forces >1 batch here).

_WIDE = [
    "ACGT",
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 3,
    "TTGA" * 2,
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    "A" * 200,
    "ACG",
    "TTGGCCAA" * 9,
    "CAGT" * 5,
    "GGGGTTTTAAAACCCC",
    "TACG" * 30,
]


def test_bucketing_preserves_input_order_for_embeddings(adapter):
    """GATE: pooled embeddings under bucketing == embeddings computed one at a time."""
    batched = adapter.embed(_WIDE, layers="last", pool="mean").arrays
    solo = [adapter.embed([s], layers="last", pool="mean").arrays[0] for s in _WIDE]
    assert len(batched) == len(_WIDE)
    for b, s in zip(batched, solo, strict=True):
        np.testing.assert_allclose(b, s, rtol=0, atol=1e-4)


def test_bucketing_preserves_input_order_for_pool_none(adapter):
    """GATE: per-token arrays/spans stay keyed to their INPUT position (length swap = bug)."""
    res = adapter.embed(_WIDE, layers="last", pool="none")
    assert [arr.shape[0] for arr in res.arrays] == [len(s) for s in _WIDE]
    assert res.token_spans is not None
    assert [len(sp) for sp in res.token_spans] == [len(s) for s in _WIDE]


def test_bucketing_preserves_input_order_for_llr(adapter_llr):
    """GATE: masked-marginal LLR jobs bucket by reference length; scores stay 1:1 with items."""
    short = "acgtacgt"
    mid = "acgtacgtacgtacgt"
    long = "acgtttggccaa" * 12
    items = [
        {"reference": long, "mutations": [[5, long[5].upper(), "A"]]},
        {"reference": short, "mutations": [[2, "G", "T"]]},
        {"reference": mid, "mutations": [[2, "G", "A"], [8, "A", "C"]]},
        {"reference": long, "mutations": [[100, long[100].upper(), "G"]]},
        {"reference": short, "mutations": [[6, "G", "C"]]},
        {"reference": mid, "mutations": [[5, "C", "T"]]},
    ]
    batched = adapter_llr.score_variant_llr(items)
    solo = [adapter_llr.score_variant_llr([it])[0] for it in items]
    assert len(batched) == len(items)
    np.testing.assert_allclose(batched, solo, rtol=0, atol=1e-4)


def test_rnagym_dms_fallback_end_to_end(adapter_llr):
    """Exit gate 5: RnagymDmsTask scored with gLM2 falls back to LLR → OK + method."""
    from glmbench.tasks.base import ResultStatus
    from glmbench.tasks.rnagym_dms import RnagymDmsTask

    tiny_manifest = (
        Path(__file__).resolve().parents[1] / "data" / "rnagym_tiny" / "tiny_manifest.yaml"
    )
    result = RnagymDmsTask(tiny_manifest).evaluate(adapter_llr)
    assert result.status is ResultStatus.OK
    assert result.metadata["scoring_method"] == "masked_marginal_llr"
    assert np.isfinite(result.metrics["macro_spearman"])
    assert result.metadata["n_unscoreable"] == 0  # tiny fixture is all substitutions
