"""GenomeOceanAdapter + genomeocean_runner — the stock-HF BPE Mistral path.

CPU gate (no GPU — a tiny random ``MistralForCausalLM`` with a trained DNA **BPE**
tokenizer stands in for GenomeOcean-100M/500M). The point of this suite is the BPE-specific
behavior that differs from the 1-char/token causal-LM adapters (e.g. ``loam-hf``):

- capabilities are ``{TOKENIZE, SEQUENCE_LOGLIKELIHOOD, PER_TOKEN_LOGPROBS, EMBEDDING}`` and
  ``LOGPROB_EMBEDDING`` is **N/A** (BPE ⇒ no 1 log-prob per nucleotide) — fail-closed;
- ``model_hash`` deterministic + sensitive, torch-free;
- ``LocalSubprocessRunner`` round-trip for tokenize / score / per_token_logprobs / embed;
- **BPE ``token_spans`` cover the sequence with variable widths** (the core BPE gate);
- ``per_token_logprobs`` is per **BPE token**, and ``mean(per_token) == score(mean)``;
- scoring is **padding-invariant** across batch sizes (right-padding + special masking);
- ``embed`` pooling incl/excl special tokens, layers="all" in one pass, NaN guards.

Torch-free CI skips the whole module (no torch/transformers). The real wire boundary is
exercised with ``python_exe = sys.executable`` (this same env).
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
from glmbench.adapters.genomeocean import GenomeOceanAdapter  # noqa: E402
from glmbench.config.model_spec import ModelSpec  # noqa: E402

# Long-ish sequences with repeated motifs so the trained BPE forms multi-nt tokens.
SEQS = [
    "ACGTACGTACGTACGTTTGGCCAAACGTACGT",
    "TTGGCCAATTGGCCAAACGTACGTTTGGCCAA",
    "ACGTTTGGACGTACGTCCAAACGTACGTACGT",
]


def _build_tiny_bpe_hf_dir(dir_path: Path, *, n_layers: int = 3, seed: int = 0) -> Path:
    """A tiny random MistralForCausalLM + a trained DNA **BPE** tokenizer (variable nt/token).

    Mirrors GenomeOcean's contract: BERT-style specials used as BOS/EOS
    ([UNK]=0, [CLS]=1, [SEP]=2, [PAD]=3, [MASK]=4), sequences wrapped ``[CLS] … [SEP]``, and
    a fast tokenizer that yields ``special_tokens_mask`` + offset mappings.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    # --- train a tiny DNA BPE so tokens span variable numbers of nucleotides ---
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()  # one "word" per contiguous run → merges within DNA
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=80, special_tokens=specials, min_frequency=1)
    corpus = [
        "ACGT" * 64,
        "TTGGCCAA" * 48,
        "ACGTTTGG" * 48,
        "CCAAACGT" * 48,
        "ACGTACGTTTGGCCAA" * 32,
    ]
    tk.train_from_iterator(corpus, trainer)
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    vocab_size = tok.vocab_size

    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    cfg.architectures = ["MistralForCausalLM"]
    model = MistralForCausalLM(cfg)
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    return dir_path


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory):
    return _build_tiny_bpe_hf_dir(tmp_path_factory.mktemp("go") / "genomeocean-tiny")


@pytest.fixture
def adapter(hf_dir):
    return GenomeOceanAdapter(
        checkpoint=str(hf_dir),
        python_exe=sys.executable,
        device="cpu",
        dtype="float32",
        weights_digest_strategy="file_sha256",
    )


# --------------------------------------------------------------------------- #
# capability contract (fail-closed) — LOGPROB_EMBEDDING is N/A for BPE
# --------------------------------------------------------------------------- #


def test_capabilities_exclude_logprob_embedding():
    assert frozenset(
        {
            Capability.TOKENIZE,
            Capability.SEQUENCE_LOGLIKELIHOOD,
            Capability.PER_TOKEN_LOGPROBS,
            Capability.EMBEDDING,
        }
    ) == GenomeOceanAdapter.CAPABILITIES
    assert Capability.LOGPROB_EMBEDDING not in GenomeOceanAdapter.CAPABILITIES
    assert missing_capability_methods(GenomeOceanAdapter) == []


def test_logprob_embedding_is_na_fail_closed(adapter):
    """BPE ⇒ no per-nucleotide log-probs: calling it raises (clean first-class N/A)."""
    with pytest.raises(CapabilityNotImplemented):
        adapter.logprob_embedding(SEQS)


def test_unknown_backend_raises(hf_dir):
    with pytest.raises(ValueError, match="local' or 'docker'"):
        GenomeOceanAdapter(checkpoint=str(hf_dir), backend="slurm")


# --------------------------------------------------------------------------- #
# model hash — torch-free, stable, sensitive
# --------------------------------------------------------------------------- #


def test_model_hash_no_forward_stable_and_sensitive(tmp_path):
    a_dir = _build_tiny_bpe_hf_dir(tmp_path / "a", seed=1)
    a = GenomeOceanAdapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    )
    h1, h2 = a.model_hash(), a.model_hash()
    assert h1 == h2 and h1.startswith("glmb:")

    b_dir = _build_tiny_bpe_hf_dir(tmp_path / "b", seed=2)
    hb = GenomeOceanAdapter(
        checkpoint=str(b_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hb != h1

    cfg_path = a_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    data["num_hidden_layers"] = data["num_hidden_layers"] + 1
    cfg_path.write_text(json.dumps(data))
    hc = GenomeOceanAdapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hc != h1


def test_model_hash_hf_revision_no_path():
    """hf_revision keys the hash off the pinned commit — no checkpoint read needed."""
    a = GenomeOceanAdapter(
        checkpoint="pGenomeOcean/GenomeOcean-100M",
        revision="deadbeef",
        weights_digest_strategy="hf_revision",
        python_exe=sys.executable,
    )
    h = a.model_hash()
    assert h.startswith("glmb:")
    # a different revision → different hash
    b = GenomeOceanAdapter(
        checkpoint="pGenomeOcean/GenomeOcean-100M",
        revision="feedface",
        weights_digest_strategy="hf_revision",
        python_exe=sys.executable,
    )
    assert b.model_hash() != h


def test_describe_echoes_config(adapter):
    d = adapter.describe()
    assert d["name"] == "genomeocean"
    assert d["capabilities"] == [
        "embedding", "per_token_logprobs", "sequence_loglikelihood", "tokenize"
    ]
    assert d["backend"] == "local"
    assert d["embedding_dim"] == 32
    assert d["n_layers"] == 3
    assert d["max_context"] == 5000  # default nt bound (NOT max_position_embeddings)
    assert d["max_tokens"] == 1024
    assert d["pool_include_special"] is True
    assert d["arch"]["family"] == "genomeocean"
    assert d["arch"]["model_type"] == "mistral"


# --------------------------------------------------------------------------- #
# spec → adapter + dry-run command
# --------------------------------------------------------------------------- #


def test_from_spec_builds_local_adapter(hf_dir):
    spec = ModelSpec.model_validate(
        {
            "adapter": "genomeocean",
            "adapter_version": "0.2.0",
            "model": {"checkpoint": str(hf_dir), "dtype": "float32", "max_context": 4096},
            "weights_digest": {"strategy": "file_sha256"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 4}},
        }
    )
    a = GenomeOceanAdapter.from_spec(spec)
    assert a.backend == "local" and a.device == "cpu" and a.batch_size == 4
    assert a.max_context == 4096
    scores = a.score_sequences(SEQS)
    assert len(scores) == len(SEQS)


def test_from_spec_requires_checkpoint():
    spec = ModelSpec.model_validate(
        {"adapter": "genomeocean", "adapter_version": "0.2.0", "model": {}}
    )
    with pytest.raises(ValueError, match="must set 'checkpoint'"):
        GenomeOceanAdapter.from_spec(spec)


def test_dry_run_emits_local_command(hf_dir, capsys):
    a = GenomeOceanAdapter(checkpoint=str(hf_dir), python_exe="/envs/loam/bin/python", dry_run=True)
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.score_sequences(SEQS)
    out = capsys.readouterr().out
    assert "/envs/loam/bin/python -m glmbench.runners.genomeocean_runner" in out


# --------------------------------------------------------------------------- #
# tokenize — BPE ids (no special tokens)
# --------------------------------------------------------------------------- #


def test_tokenize_returns_bpe_ids(adapter):
    toks = adapter.tokenize(SEQS)
    assert len(toks) == len(SEQS)
    for seq, t in zip(SEQS, toks, strict=True):
        assert all(isinstance(x, int) for x in t)
        # BPE compresses: fewer tokens than nucleotides for repetitive DNA
        assert 0 < len(t) < len(seq)


# --------------------------------------------------------------------------- #
# scoring — round-trip, reduction, padding-invariance, NaN guard
# --------------------------------------------------------------------------- #


def test_score_sequences_finite_and_len(adapter):
    scores = adapter.score_sequences(SEQS)
    assert len(scores) == len(SEQS)
    assert all(np.isfinite(s) and s <= 0.0 for s in scores)


def test_score_reduction_mean_and_sum(adapter):
    mean = adapter.score_sequences(SEQS, reduction="mean")
    total = adapter.score_sequences(SEQS, reduction="sum")
    for m, s in zip(mean, total, strict=True):
        assert abs(s) >= abs(m) - 1e-6


def test_score_padding_invariant(hf_dir):
    """Right-padding + special masking ⇒ scores independent of batch size."""
    a1 = GenomeOceanAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                            dtype="float32", weights_digest_strategy="file_sha256", batch_size=1)
    aN = GenomeOceanAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                            dtype="float32", weights_digest_strategy="file_sha256", batch_size=8)
    s1 = a1.score_sequences(SEQS)
    sN = aN.score_sequences(SEQS)
    np.testing.assert_allclose(s1, sN, rtol=0, atol=1e-5)


def test_score_rejects_unsupported_reduction(adapter):
    with pytest.raises(ValueError, match="mean'|sum'"):
        adapter.score_sequences(SEQS, reduction="max")


def test_score_nan_guard_trips(adapter):
    from glmbench.adapters.runner_backend import RunResult
    from glmbench.adapters.wire import Response, WireProtocolError

    class _NaNBackend:
        def execute(self, op, params, sequences, ids=None, *, output_name="output.npz"):
            return RunResult(
                response=Response(op=op, status="ok", output_path="x.npz"),
                payload={"scores": np.full(len(sequences), np.nan, dtype=np.float64)},
                scratch_dir="x",
            )

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.score_sequences(SEQS)


# --------------------------------------------------------------------------- #
# per_token_logprobs — per BPE token; mean == score(mean)
# --------------------------------------------------------------------------- #


def test_per_token_logprobs_per_bpe_token(adapter):
    z = adapter.per_token_logprobs(SEQS)
    toks = adapter.tokenize(SEQS)
    assert len(z) == len(SEQS)
    for seq, arr, t in zip(SEQS, z, toks, strict=True):
        assert np.isfinite(arr).all() and np.all(arr <= 0.0)
        assert arr.shape == (len(t),)  # one log-prob per BPE token (NOT per nt)
        assert arr.shape[0] < len(seq)  # fewer tokens than nucleotides


def test_per_token_mean_equals_score_mean(adapter):
    """Consistency: the per-token vector is exactly what scoring reduces."""
    z = adapter.per_token_logprobs(SEQS)
    mean_scores = adapter.score_sequences(SEQS, reduction="mean")
    for arr, m in zip(z, mean_scores, strict=True):
        np.testing.assert_allclose(float(arr.mean()), m, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------- #
# embeddings — BPE token_spans, layers, pooling, NaN guard
# --------------------------------------------------------------------------- #


def test_embed_pool_mean_shape(adapter):
    res = adapter.embed(SEQS, layers="last", pool="mean")
    assert res.pool == "mean" and res.token_spans is None
    assert res.layers == [3] and res.embedding_dim == 32
    for arr in res.arrays:
        assert arr.shape == (1, 32) and np.isfinite(arr).all()


def test_embed_layers_all_one_pass_and_distinct(adapter):
    res = adapter.embed(SEQS, layers="all", pool="mean")
    assert res.layers == [0, 1, 2, 3]
    for arr in res.arrays:
        assert arr.shape == (4, 32)
        assert not np.allclose(arr[0], arr[3]), "layer-0 and last layer identical"


def test_embed_pool_none_bpe_spans_cover_sequence(adapter):
    """The core BPE gate: spans are contiguous, cover exactly len(seq), variable width."""
    res = adapter.embed(SEQS, layers="all", pool="none")
    assert res.pool == "none" and res.token_spans is not None
    for seq, arr, spans in zip(SEQS, res.arrays, res.token_spans, strict=True):
        n_tok = len(spans)
        assert arr.shape == (n_tok, 4, 32)
        # contiguous, covering the whole sequence
        assert spans[0][0] == 0
        assert spans[-1][1] == len(seq)
        for prev, nxt in zip(spans, spans[1:], strict=False):
            assert nxt[0] == prev[1]  # no gaps / overlaps
        widths = [b - a for a, b in spans]
        assert sum(widths) == len(seq)
        assert max(widths) > 1  # BPE made at least one multi-nt token (variable width)


def test_embed_pool_include_special_changes_mean(hf_dir):
    """Pooling with vs without [CLS]/[SEP] gives different mean embeddings."""
    incl = GenomeOceanAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                              dtype="float32", weights_digest_strategy="file_sha256",
                              pool_include_special=True)
    excl = GenomeOceanAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                              dtype="float32", weights_digest_strategy="file_sha256",
                              pool_include_special=False)
    a = incl.embed(SEQS, layers="last", pool="mean")
    b = excl.embed(SEQS, layers="last", pool="mean")
    assert not np.allclose(a.arrays[0], b.arrays[0])


def test_embed_hidden_states_match_causal_lm(adapter, hf_dir):
    """GATE: the head-less ``AutoModel`` embed path returns the CausalLM's LAST-BLOCK output.

    ``_load(for_logits=False)`` drops the LM head for ``embed``/``tokenize`` (the
    [B, S, 4096] logit matmul nothing read). This gate's job is to prove that dropping the
    head did not change which weights are read: the two loads must agree.

    **The reference is the last block's raw output, not ``hidden_states[-1]``.** On this
    pre-LN Mistral stack ``hidden_states[-1]`` is ``norm(last_block_output)`` — a different
    tensor. The readout convention names the block output, so the reference here is the
    CausalLM's hooked last block, and the test additionally asserts that
    ``hidden_states[-1]`` does NOT match. Without that second half, a regression that quietly
    restored ``hidden_states[-1]`` on both sides would still pass.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = adapter.embed(SEQS, layers="last", pool="none")
    tok = AutoTokenizer.from_pretrained(str(hf_dir), padding_side="right")
    clm = AutoModelForCausalLM.from_pretrained(str(hf_dir)).eval()
    for seq, arr in zip(SEQS, res.arrays, strict=True):
        enc = tok(
            seq,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        cap: dict = {}
        handle = clm.model.layers[-1].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__(
                "o", (o[0] if isinstance(o, tuple) else o).detach()
            )
        )
        try:
            with torch.no_grad():
                out = clm(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                )
        finally:
            handle.remove()
        real = enc["special_tokens_mask"][0] == 0
        ref = cap["o"][0][real].float().numpy()
        assert arr.shape == (ref.shape[0], 1, ref.shape[1])
        np.testing.assert_allclose(arr[:, 0, :], ref, rtol=0, atol=1e-4)

        # …and it is genuinely NOT the final-normed tensor.
        old = out.hidden_states[-1][0][real].float().numpy()
        rms = float(np.sqrt((ref**2).mean()))
        assert float(np.abs(ref - old).max()) / rms > 1e-2, (
            "the block output and norm(block output) are indistinguishable on this fixture, "
            "so this test could not tell the two readouts apart."
        )


def test_embed_and_score_paths_agree_on_the_same_model(adapter, hf_dir):
    """GATE: dropping the LM head for embed did not desync the two paths.

    The score path keeps ``AutoModelForCausalLM``; the embed path loads ``AutoModel``. Both
    must be reading the *same* weights: re-deriving the scores from the CausalLM's logits and
    the embeddings from the same checkpoint's hidden states has to reproduce both runner
    outputs at once.
    """
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scores = adapter.score_sequences(SEQS, reduction="mean")
    emb = adapter.embed(SEQS, layers="last", pool="mean")  # pool_include_special=True default

    tok = AutoTokenizer.from_pretrained(str(hf_dir), padding_side="right")
    clm = AutoModelForCausalLM.from_pretrained(str(hf_dir)).eval()
    for seq, score, arr in zip(SEQS, scores, emb.arrays, strict=True):
        enc = tok(
            seq,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        cap: dict = {}
        handle = clm.model.layers[-1].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__(
                "o", (o[0] if isinstance(o, tuple) else o).detach()
            )
        )
        try:
            with torch.no_grad():
                out = clm(
                    input_ids=ids,
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                )
        finally:
            handle.remove()
        lp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        tok_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        real_targets = enc["special_tokens_mask"][0, 1:] == 0
        np.testing.assert_allclose(
            float(tok_lp[real_targets].mean()), score, rtol=0, atol=1e-5
        )
        # mean-pooled over ALL non-pad positions (specials included = vendor convention),
        # off the LAST BLOCK's raw output (the readout convention), not
        # `out.hidden_states[-1]` (which is norm(block_output) on this pre-LN stack).
        ref = cap["o"][0].float().mean(dim=0).numpy()
        np.testing.assert_allclose(arr[0], ref, rtol=0, atol=1e-4)


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
                    "layer_ids": np.asarray([3], dtype=np.int64),
                    "embedding_dim": np.asarray(32, dtype=np.int64),
                },
                scratch_dir="x",
            )

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.embed(SEQS, layers="last", pool="mean")


# --------------------------------------------------------------------------- #
# bucketing order gate
# --------------------------------------------------------------------------- #
#
# The runner length-buckets its batches (glmbench.runners._batching) and scatters results
# back by INPUT index. A bad scatter is SILENT: every metric still computes, just on rows
# belonging to other sequences. These pin the invariant with a deliberately wide length
# spread, which is what makes the bucketer reorder aggressively.

_WIDE = [
    "ACGT",
    "ACGTACGTTTGGCCAA" * 12,
    "TTGA" * 2,
    "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
    "TTGGCCAA" * 25,
    "ACG",
]


def test_bucketing_preserves_input_order_for_scores(adapter):
    """GATE: per-sequence scores stay keyed to their INPUT position under bucketing."""
    batched = adapter.score_sequences(_WIDE, reduction="mean")
    solo = [adapter.score_sequences([s], reduction="mean")[0] for s in _WIDE]
    np.testing.assert_allclose(batched, solo, rtol=0, atol=1e-4)


def test_bucketing_preserves_input_order_for_per_token_logprobs(adapter):
    """GATE: per-token vector i must be sequence i's — length AND values, one at a time."""
    batched = adapter.per_token_logprobs(_WIDE)
    solo = [adapter.per_token_logprobs([s])[0] for s in _WIDE]
    toks = adapter.tokenize(_WIDE)
    assert [len(z) for z in batched] == [len(t) for t in toks]
    for b, s in zip(batched, solo, strict=True):
        np.testing.assert_allclose(b, s, rtol=0, atol=1e-4)


def test_bucketing_preserves_input_order_for_embeddings(adapter):
    """GATE: pooled embeddings under bucketing == embeddings computed one at a time."""
    batched = np.stack(adapter.embed(_WIDE, layers="last", pool="mean").arrays, axis=0)
    solo = np.concatenate(
        [adapter.embed([s], layers="last", pool="mean").arrays for s in _WIDE], axis=0
    )
    np.testing.assert_allclose(batched, solo, rtol=0, atol=1e-3)


def test_bucketing_preserves_input_order_for_pool_none_spans(adapter):
    """GATE: pool='none' arrays + BPE spans stay aligned to their own sequence.

    A mis-scatter here swaps sequence A's token vectors under sequence B's spans — spans
    that no longer sum to len(seq_B) are the loud tell.
    """
    res = adapter.embed(_WIDE, layers="last", pool="none")
    for seq, arr, spans in zip(_WIDE, res.arrays, res.token_spans, strict=True):
        assert spans[0][0] == 0 and spans[-1][1] == len(seq)
        assert arr.shape[0] == len(spans)
