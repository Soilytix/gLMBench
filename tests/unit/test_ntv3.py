"""NTv3Adapter + ntv3_runner — the U-Net masked-LM ENCODER path (embeddings + LLR).

CPU gate (no GPU). A **tiny real NTv3** (`num_layers=2, embed_dim=32, heads=2,
num_downsamples=3`) is built from the cached remote `Ntv3PreTrainedConfig` + `NTv3PreTrained`
(`AutoModelForMaskedLM`, `trust_remote_code=True`) with the real `NTv3Tokenizer`,
`save_pretrained` to a temp dir, and driven through `LocalSubprocessRunner` with
`python_exe=sys.executable`. Needs the Hub for the remote code: those tests are marked
``network`` (see ``conftest.py``). `num_downsamples=3` ⇒ the U-Net downsamples 2^3=8×, so
the test uses `pad_multiple=8` (a light stand-in for the real 128) — the padding/bucketing
logic is identical.

The point of this suite is the NTv3-specific behavior that differs from gLM2 (the 1-nt encoder
it was copied from):

- capabilities are `{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}`; the causal caps are N/A;
- **no wrapping / no strand marker** — tokenize length is exactly `len(seq)`, 1 nt/token;
- **R5 per-sequence 128-multiple padding + a mask-free torso** ⇒ scores/embeddings are
  batch-invariant only because each sequence is padded to its OWN next multiple (bucketed);
- **R2 mixed-resolution hidden states** — `pool='none'` at one resolution gives spans covering
  `len(seq)`; requesting mixed-resolution layers with `pool='none'` fails loud;
- **masked-marginal LLR** — logit diff == log-softmax diff; multi-site == sum of single sites;
  caching == naive; favored base positive; fails loud w/o [MASK]; NaN guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from glmbench.adapters.base import (  # noqa: E402
    Capability,
    CapabilityNotImplemented,
    missing_capability_methods,
)
from glmbench.adapters.ntv3 import NTv3Adapter  # noqa: E402
from glmbench.config.model_spec import ModelSpec  # noqa: E402

REPO = "InstaDeepAI/NTv3_100M_pre"  # source of the real config + tokenizer (downsized for the test)
PAD_MULTIPLE = 8  # matches the tiny model's num_downsamples=3 (2^3); real model = 128
# Lengths deliberately NOT multiples of 8 → exercise padding + cross-bucket batch-invariance.
SEQS = [
    "ACGTACGTACGT",  # 12 → pads to 16
    "TTGGCCAATTGGCCAAACGT",  # 20 → pads to 24
    "ACGTTTGGACGTACGTCCAAACGTACGT",  # 28 → pads to 32
]
# Probe-confirmed real vocab ids (stable across the downsizing — only arch dims change).
NT_IDS = {"A": 6, "T": 7, "C": 8, "G": 9, "N": 10}
UNK_ID, PAD_ID, MASK_ID = 0, 1, 2


def _build_tiny_ntv3_dir(dir_path: Path, *, seed: int = 0) -> Path:
    """A tiny random NTv3PreTrained (AutoModelForMaskedLM) + the real NTv3Tokenizer, saved.

    Pulls the real `Ntv3PreTrainedConfig` (carrying the cross-repo `auto_map` to
    `InstaDeepAI/ntv3_base_model`) + the real tokenizer, downsizes the arch (small dims +
    `num_downsamples=3`), random-inits, and `save_pretrained`. Skips if the remote code can't
    be fetched (offline + cold cache).
    """
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - offline / hub down → skip, not fail
        pytest.skip(f"NTv3 remote code/tokenizer unavailable ({type(e).__name__}: {e})")
    cfg.num_layers = 2
    cfg.embed_dim = 32
    cfg.conv_init_embed_dim = 32  # equal to embed_dim → uniform conv filters (tiny + simple)
    cfg.attention_heads = 2
    cfg.key_size = 16
    cfg.num_downsamples = 3  # 2^3 = 8 → pad_multiple 8 for the test
    cfg.ffn_embed_dim = 64
    cfg.token_embed_dim = 8
    torch.manual_seed(seed)
    try:
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - no such auto_map → skip
        pytest.skip(f"NTv3 AutoModelForMaskedLM unavailable ({type(e).__name__}: {e})")
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    _copy_remote_code(dir_path)
    return dir_path


def _copy_remote_code(dir_path: Path) -> None:
    """Copy the cached remote `*.py` next to the checkpoint so the subprocess reload is offline-safe.

    NTv3's modeling/config/tokenization code lives in the SEPARATE `InstaDeepAI/ntv3_base_model`
    repo (cross-repo `auto_map`), not the weights repo — copy from both.
    """
    import shutil

    from transformers.utils import HF_MODULES_CACHE

    for repo_key in (REPO, "InstaDeepAI/ntv3_base_model"):
        src = Path(HF_MODULES_CACHE) / "transformers_modules" / repo_key.replace("/", "--")
        if src.exists():
            for p in src.rglob("*.py"):
                shutil.copy(p, dir_path / p.name)


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory, hub_access):
    return _build_tiny_ntv3_dir(tmp_path_factory.mktemp("ntv3") / "ntv3-tiny")


def _mk(hf_dir, **kw) -> NTv3Adapter:
    base = dict(
        checkpoint=str(hf_dir),
        python_exe=sys.executable,
        device="cpu",
        dtype="float32",
        weights_digest_strategy="file_sha256",
        pad_multiple=PAD_MULTIPLE,
    )
    base.update(kw)
    return NTv3Adapter(**base)


@pytest.fixture
def adapter(hf_dir):
    return _mk(hf_dir)


# --------------------------------------------------------------------------- #
# capability contract (fail-closed) — encoder ⇒ causal caps are N/A
# --------------------------------------------------------------------------- #


def test_capabilities_are_tokenize_embedding_and_llr():
    assert (
        frozenset({Capability.TOKENIZE, Capability.EMBEDDING, Capability.MASKED_MARGINAL_LLR})
        == NTv3Adapter.CAPABILITIES
    )
    for absent in (
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.LOGPROB_EMBEDDING,
    ):
        assert absent not in NTv3Adapter.CAPABILITIES
    assert missing_capability_methods(NTv3Adapter) == []


def test_scoring_capabilities_are_na_fail_closed(adapter):
    with pytest.raises(CapabilityNotImplemented):
        adapter.score_sequences(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.per_token_logprobs(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.logprob_embedding(SEQS)


def test_unknown_backend_raises(hf_dir):
    with pytest.raises(ValueError, match="local' or 'docker'"):
        NTv3Adapter(checkpoint=str(hf_dir), backend="slurm")


def test_bad_pad_multiple_raises(hf_dir):
    with pytest.raises(ValueError, match="pad_multiple"):
        NTv3Adapter(checkpoint=str(hf_dir), pad_multiple=0)


# --------------------------------------------------------------------------- #
# model hash — torch-free, stable, sensitive
# --------------------------------------------------------------------------- #


def test_model_hash_no_forward_stable_and_sensitive(tmp_path):
    a_dir = _build_tiny_ntv3_dir(tmp_path / "a", seed=1)
    a = _mk(a_dir)
    h1, h2 = a.model_hash(), a.model_hash()
    assert h1 == h2 and h1.startswith("glmb:")

    b_dir = _build_tiny_ntv3_dir(tmp_path / "b", seed=2)
    assert _mk(b_dir).model_hash() != h1  # different weights → different hash

    cfg_path = a_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    data["num_layers"] = int(data["num_layers"]) + 1
    cfg_path.write_text(json.dumps(data))
    assert _mk(a_dir).model_hash() != h1  # arch field (num_layers) feeds the hash


def test_model_hash_hf_revision_no_path():
    a = NTv3Adapter(
        checkpoint=REPO, revision="deadbeef", code_revision="c0ffee",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    h = a.model_hash()
    assert h.startswith("glmb:")
    b = NTv3Adapter(
        checkpoint=REPO, revision="feedface", code_revision="c0ffee",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    assert b.model_hash() != h


def test_describe_echoes_config(adapter):
    d = adapter.describe()
    assert d["name"] == "ntv3"
    assert d["capabilities"] == ["embedding", "masked_marginal_llr", "tokenize"]
    assert d["backend"] == "local"
    assert d["embedding_dim"] == 32 and d["n_layers"] == 2
    assert d["pad_multiple"] == PAD_MULTIPLE and d["pool_include_special"] is False
    assert d["trust_remote_code"] is True
    assert d["arch"]["family"] == "ntv3" and d["arch"]["model_type"] == "ntv3"


# --------------------------------------------------------------------------- #
# spec → adapter + dry-run command
# --------------------------------------------------------------------------- #


def test_from_spec_builds_local_adapter(hf_dir):
    spec = ModelSpec.model_validate(
        {
            "adapter": "ntv3",
            "adapter_version": "0.1.0",
            "model": {
                "checkpoint": str(hf_dir), "dtype": "float32",
                "max_context": 2048, "pad_multiple": PAD_MULTIPLE,
                "code_revision": "abc123",
            },
            "weights_digest": {"strategy": "file_sha256"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 2}},
        }
    )
    a = NTv3Adapter.from_spec(spec)
    assert a.backend == "local" and a.device == "cpu" and a.batch_size == 2
    assert a.max_context == 2048 and a.trust_remote_code is True
    assert a.code_revision == "abc123" and a.pad_multiple == PAD_MULTIPLE
    res = a.embed(SEQS, layers="last", pool="mean")
    assert len(res.arrays) == len(SEQS)


def test_from_spec_requires_checkpoint():
    spec = ModelSpec.model_validate({"adapter": "ntv3", "adapter_version": "0.1.0", "model": {}})
    with pytest.raises(ValueError, match="must set 'checkpoint'"):
        NTv3Adapter.from_spec(spec)


def test_dry_run_emits_local_command(hf_dir, capsys):
    a = NTv3Adapter(checkpoint=str(hf_dir), python_exe="/envs/loam/bin/python", dry_run=True)
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.embed(SEQS, layers="last", pool="mean")
    out = capsys.readouterr().out
    assert "/envs/loam/bin/python -m glmbench.runners.ntv3_runner" in out


# --------------------------------------------------------------------------- #
# tokenize — 1 nt/token, no wrapping, no marker
# --------------------------------------------------------------------------- #


def test_tokenize_is_one_nt_per_token_no_wrapping(adapter):
    toks = adapter.tokenize(["ACGTN"])
    assert toks[0] == [NT_IDS["A"], NT_IDS["C"], NT_IDS["G"], NT_IDS["T"], NT_IDS["N"]]


def test_tokenize_length_equals_seq_len(adapter):
    for seq, t in zip(SEQS, adapter.tokenize(SEQS), strict=True):
        assert len(t) == len(seq)  # 1 nt/token, no <cls>/<eos>/<bos>


def test_tokenize_non_acgt_maps_to_unk(adapter):
    # 'X'/'-' are not ACGTN → <unk>=0 (the tokenizer uppercases then falls back).
    toks = adapter.tokenize(["AXG-"])
    assert toks[0] == [NT_IDS["A"], UNK_ID, NT_IDS["G"], UNK_ID]


# --------------------------------------------------------------------------- #
# embeddings — shapes, layers, spans, pooling, batch-invariance (R5), NaN guard
# --------------------------------------------------------------------------- #


def test_embed_pool_mean_shape(adapter):
    res = adapter.embed(SEQS, layers="last", pool="mean")
    assert res.pool == "mean" and res.token_spans is None
    # num_downsamples=3, num_layers=2 ⇒ 2*3+2 = 8 hidden states; last index 7.
    assert res.layers == [7] and res.embedding_dim == 32
    for arr in res.arrays:
        assert arr.shape == (1, 32) and np.isfinite(arr).all()


def test_embed_layers_all_one_pass(adapter):
    res = adapter.embed(SEQS, layers="all", pool="mean")
    assert res.layers == list(range(8))  # indexed against the actual 8-state tuple
    for arr in res.arrays:
        assert arr.shape == (8, 32) and np.isfinite(arr).all()


def test_embed_pool_none_spans_cover_sequence(adapter):
    """R2: final layer is nucleotide resolution ⇒ unit-width spans covering len(seq)."""
    res = adapter.embed(SEQS, layers="last", pool="none")
    assert res.pool == "none" and res.token_spans is not None
    for seq, arr, spans in zip(SEQS, res.arrays, res.token_spans, strict=True):
        assert len(spans) == len(seq)  # one row per nucleotide (final layer, f=1)
        assert arr.shape == (len(seq), 1, 32)
        assert spans[0] == (0, 1) and spans[-1] == (len(seq) - 1, len(seq))
        for j, (a, b) in enumerate(spans):
            assert (a, b) == (j, j + 1)  # unit-width, contiguous, no gaps


def test_embed_pool_none_mixed_resolution_fails_loud(adapter):
    """R2: pool='none' across mixed-resolution layers is refused (can't share token_spans)."""
    from glmbench.adapters.runner_backend import RunnerError

    # layers 0 (nt res, len=plen) and 3 (torso, len=plen/8) differ in length → error.
    with pytest.raises((RunnerError, ValueError), match="one resolution|ONE resolution"):
        adapter.embed(SEQS, layers=[0, 3], pool="none")


def test_embed_batch_size_invariant(hf_dir):
    """R5: per-sequence 128-multiple padding + bucketing ⇒ embeddings independent of batch size,
    even across sequences that pad to DIFFERENT lengths (the mask-free-torso hazard)."""
    e1 = _mk(hf_dir, batch_size=1).embed(SEQS, layers="last", pool="mean")
    eN = _mk(hf_dir, batch_size=8).embed(SEQS, layers="last", pool="mean")
    for x, y in zip(e1.arrays, eN.arrays, strict=True):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-5)


def test_embed_pool_include_special_changes_mean(hf_dir):
    """pool_include_special=True pools over the padded length (incl. the pad region) → differs."""
    incl = _mk(hf_dir, pool_include_special=True).embed(SEQS, layers="last", pool="mean")
    excl = _mk(hf_dir, pool_include_special=False).embed(SEQS, layers="last", pool="mean")
    # SEQS[0] (len 12) pads to 16 → 4 pad positions, so including them shifts the mean.
    assert not np.allclose(incl.arrays[0], excl.arrays[0])


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
                    "layer_ids": np.asarray([7], dtype=np.int64),
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

REF = "ACGTACGTACGTACGT"  # len 16 (pads to 16); index 2='G', 5='C', 8='A'


def _load_mlm_in_process(hf_dir: Path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(str(hf_dir), trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True, padding_side="right")
    return model, tok


def test_llr_adapter_roundtrip_finite_and_1to1(adapter):
    items = [
        {"reference": REF, "mutations": [[2, "G", "A"]]},
        {"reference": REF, "mutations": [[2, "G", "T"], [5, "C", "A"]]},
    ]
    scores = adapter.score_variant_llr(items)
    assert len(scores) == len(items)
    assert all(np.isfinite(s) for s in scores)


def test_llr_logit_diff_equals_log_softmax_diff(hf_dir):
    """LLR == raw logit diff == log-softmax diff (the shared masked context cancels the normalizer)."""
    from glmbench.runners.ntv3_runner import _encode_ids, _pad_len, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir)
    device = torch.device("cpu")
    out = _score_variant_llr(
        model, tok, [{"reference": REF, "mutations": [[2, "G", "A"]]}],
        device=device, batch_size=8, max_tokens=4096, pad_multiple=PAD_MULTIPLE,
    )
    llr = float(out["scores"][0])

    ids = _encode_ids(tok, [REF], max_tokens=4096)[0]
    plen = _pad_len(len(ids), PAD_MULTIPLE)
    t = 2  # no marker offset for NTv3
    ids_m = list(ids) + [int(tok.pad_token_id)] * (plen - len(ids))
    ids_m[t] = int(tok.mask_token_id)
    inp = torch.tensor([ids_m], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp).logits[0, t].float()
    mut_id = int(tok.convert_tokens_to_ids("A"))
    wt_id = int(tok.convert_tokens_to_ids("G"))
    assert llr == pytest.approx(float(logits[mut_id] - logits[wt_id]), abs=1e-4)
    logsm = torch.log_softmax(logits, dim=-1)
    assert llr == pytest.approx(float(logsm[mut_id] - logsm[wt_id]), abs=1e-4)


def test_llr_wt_to_wt_is_zero(adapter):
    """A no-op substitution (WT→WT) scores exactly 0 (logit_wt − logit_wt)."""
    scores = adapter.score_variant_llr([{"reference": REF, "mutations": [[5, "C", "C"]]}])
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_llr_multisite_equals_sum_of_single_sites(hf_dir):
    from glmbench.runners.ntv3_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir)
    kw = dict(device=torch.device("cpu"), batch_size=8, max_tokens=4096, pad_multiple=PAD_MULTIPLE)
    two = float(_score_variant_llr(
        model, tok, [{"reference": REF, "mutations": [[2, "G", "A"], [5, "C", "T"]]}], **kw
    )["scores"][0])
    s1 = float(_score_variant_llr(
        model, tok, [{"reference": REF, "mutations": [[2, "G", "A"]]}], **kw)["scores"][0])
    s2 = float(_score_variant_llr(
        model, tok, [{"reference": REF, "mutations": [[5, "C", "T"]]}], **kw)["scores"][0])
    assert two == pytest.approx(s1 + s2, abs=1e-4)


def test_llr_favored_base_is_positive(hf_dir):
    from glmbench.runners.ntv3_runner import _encode_ids, _pad_len, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir)
    ids = _encode_ids(tok, [REF], max_tokens=4096)[0]
    plen = _pad_len(len(ids), PAD_MULTIPLE)
    t = 2
    ids_m = list(ids) + [int(tok.pad_token_id)] * (plen - len(ids))
    ids_m[t] = int(tok.mask_token_id)
    inp = torch.tensor([ids_m], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp).logits[0, t].float()
    nt_ids = {b: int(tok.convert_tokens_to_ids(b)) for b in "ACGT"}
    fav = max(nt_ids, key=lambda b: float(logits[nt_ids[b]]))
    least = min(nt_ids, key=lambda b: float(logits[nt_ids[b]]))
    if fav == least:
        pytest.skip("degenerate logits (all nucleotide tokens equal)")
    s = float(_score_variant_llr(
        model, tok, [{"reference": REF, "mutations": [[2, least, fav]]}],
        device=torch.device("cpu"), batch_size=8, max_tokens=4096, pad_multiple=PAD_MULTIPLE,
    )["scores"][0])
    assert s > 0.0


def test_llr_caching_matches_naive(hf_dir):
    from glmbench.runners.ntv3_runner import _encode_ids, _pad_len, _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir)
    items = [
        {"reference": REF, "mutations": [[2, "G", "A"]]},
        {"reference": REF, "mutations": [[2, "G", "T"], [5, "C", "A"]]},
        {"reference": REF, "mutations": [[8, "A", "C"]]},
    ]
    cached = _score_variant_llr(
        model, tok, items, device=torch.device("cpu"), batch_size=8, max_tokens=4096,
        pad_multiple=PAD_MULTIPLE,
    )["scores"]

    def _naive(item):
        ids = _encode_ids(tok, [item["reference"]], max_tokens=4096)[0]
        plen = _pad_len(len(ids), PAD_MULTIPLE)
        total = 0.0
        for pos0, wt, mut in item["mutations"]:
            ids_m = list(ids) + [int(tok.pad_token_id)] * (plen - len(ids))
            ids_m[int(pos0)] = int(tok.mask_token_id)
            inp = torch.tensor([ids_m], dtype=torch.long)
            with torch.inference_mode():
                row = model(input_ids=inp).logits[0, int(pos0)].float()
            total += float(row[int(tok.convert_tokens_to_ids(mut))]
                           - row[int(tok.convert_tokens_to_ids(wt))])
        return total

    naive = np.array([_naive(it) for it in items], dtype=np.float64)
    np.testing.assert_allclose(cached, naive, rtol=0, atol=1e-4)


def test_llr_deterministic(adapter):
    items = [{"reference": REF, "mutations": [[2, "G", "A"], [8, "A", "C"]]}]
    assert adapter.score_variant_llr(items) == adapter.score_variant_llr(items)


def test_llr_no_mask_token_fails_loud(hf_dir):
    from glmbench.runners.ntv3_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir)

    class _NoMask:
        def __getattr__(self, k):
            return getattr(tok, k)

        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        _score_variant_llr(
            model, _NoMask(), [{"reference": REF, "mutations": [[2, "G", "A"]]}],
            device=torch.device("cpu"), batch_size=8, max_tokens=4096, pad_multiple=PAD_MULTIPLE,
        )
