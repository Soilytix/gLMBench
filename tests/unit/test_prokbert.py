"""ProkBertAdapter + prokbert_runner — the BERT-style masked-LM ENCODER path with an
OVERLAPPING-k-mer LCA tokenizer (embeddings + masked-marginal LLR).

CPU gate (no GPU). A **tiny real ProkBERT** (`num_hidden_layers=2, hidden_size=32,
heads=2`) is built from the cached remote `ProkBertConfig` + `ProkBertModel` / the real
`LCATokenizer` (`AutoModel`, `trust_remote_code=True`), `save_pretrained` to a temp dir, and
driven through `LocalSubprocessRunner` with `python_exe=sys.executable`. Needs the Hub for
the remote code: those tests are marked ``network`` (see ``conftest.py``).

The point of this suite is the ProkBERT-specific behavior that differs from gLM2 (the 1-nt
encoder it was copied from):

- capabilities are `{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}`; the causal caps are N/A;
- **[CLS]…[SEP] wrapping** — tokenize length is `n_content + 2`; content pooling excludes both;
- **overlapping k-mer token_spans** — content token `j` → window `(j·shift, j·shift+k)`; the
  spans overlap (shift<k) and their **union** covers `len(seq)` (no partition);
- **k-mer-aware masked-marginal LLR** — a single-nt substitution masks up to `k` overlapping
  tokens; multi-site == sum of independent single-site; caching == naive; fails loud w/o [MASK];
- robust layer indexing against the actual `hidden_states` length; batch-size invariance; NaN guard.
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
from glmbench.adapters.prokbert import ProkBertAdapter  # noqa: E402
from glmbench.config.model_spec import ModelSpec  # noqa: E402

REPO = "neuralbioinfo/prokbert-mini"  # source of the real config + LCA tokenizer (downsized)
SEQS = [
    "ATGTCCGCGGGACCTAAAGGGTTTCCCATGATG",
    "TTGGCCAATTGGCCAAACGTACGTTTGGCCAATT",
    "ACGTTTGGACGTACGTCCAAACGTACGTACGTAC",
]
# Probe-confirmed special-token ids (stable across the downsizing — only depth/dim/heads change).
CLS_ID, SEP_ID, MASK_ID, PAD_ID, UNK_ID = 2, 3, 4, 0, 1


def _build_tiny_prokbert_dir(dir_path: Path, *, seed: int = 0, mlm: bool = False) -> Path:
    """A tiny random ProkBERT (`AutoModel` or `AutoModelForMaskedLM`) + the real LCATokenizer.

    Pulls the real `ProkBertConfig` (carrying `auto_map` to the bundled modeling code) + the
    real tokenizer, downsizes the arch, random-inits, and `save_pretrained`. Skips if the
    remote code can't be fetched (offline + cold cache).
    """
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

    try:
        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - offline / hub down → skip, not fail
        pytest.skip(f"ProkBERT remote code/tokenizer unavailable ({type(e).__name__}: {e})")
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 32
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 64
    torch.manual_seed(seed)
    loader = AutoModelForMaskedLM if mlm else AutoModel
    try:
        model = loader.from_config(cfg, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - no such auto_map → skip
        pytest.skip(f"ProkBERT {loader.__name__} unavailable ({type(e).__name__}: {e})")
    dir_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dir_path))
    tok.save_pretrained(str(dir_path))
    _copy_remote_code(REPO, dir_path)
    return dir_path


def _copy_remote_code(repo: str, dir_path: Path) -> None:
    """Copy the cached remote `*.py` next to the checkpoint so the subprocess reload is offline-safe."""
    import shutil

    from transformers.utils import HF_MODULES_CACHE

    # ProkBERT's code lives in the separate nbrg-transformers repo (cross-repo auto_map).
    for repo_key in (repo, "neuralbioinfo/nbrg-transformers", "neuralbioinfo/nbrg_transformers"):
        src = Path(HF_MODULES_CACHE) / "transformers_modules" / repo_key.replace("/", "--")
        if src.exists():
            for p in src.rglob("*.py"):
                shutil.copy(p, dir_path / p.name)


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory, hub_access):
    return _build_tiny_prokbert_dir(tmp_path_factory.mktemp("prokbert") / "prokbert-tiny")


@pytest.fixture
def adapter(hf_dir):
    return ProkBertAdapter(
        checkpoint=str(hf_dir),
        python_exe=sys.executable,
        device="cpu",
        dtype="float32",
        weights_digest_strategy="file_sha256",
    )


# --------------------------------------------------------------------------- #
# capability contract (fail-closed) — encoder ⇒ causal caps are N/A
# --------------------------------------------------------------------------- #


def test_capabilities_are_tokenize_embedding_and_llr():
    assert (
        frozenset(
            {Capability.TOKENIZE, Capability.EMBEDDING, Capability.MASKED_MARGINAL_LLR}
        )
        == ProkBertAdapter.CAPABILITIES
    )
    for absent in (
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.LOGPROB_EMBEDDING,
    ):
        assert absent not in ProkBertAdapter.CAPABILITIES
    assert missing_capability_methods(ProkBertAdapter) == []


def test_scoring_capabilities_are_na_fail_closed(adapter):
    with pytest.raises(CapabilityNotImplemented):
        adapter.score_sequences(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.per_token_logprobs(SEQS)
    with pytest.raises(CapabilityNotImplemented):
        adapter.logprob_embedding(SEQS)


def test_unknown_backend_raises(hf_dir):
    with pytest.raises(ValueError, match="local' or 'docker'"):
        ProkBertAdapter(checkpoint=str(hf_dir), backend="slurm")


# --------------------------------------------------------------------------- #
# model hash — torch-free, stable, sensitive
# --------------------------------------------------------------------------- #


def test_model_hash_no_forward_stable_and_sensitive(tmp_path):
    a_dir = _build_tiny_prokbert_dir(tmp_path / "a", seed=1)
    a = ProkBertAdapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    )
    h1, h2 = a.model_hash(), a.model_hash()
    assert h1 == h2 and h1.startswith("glmb:")

    b_dir = _build_tiny_prokbert_dir(tmp_path / "b", seed=2)
    hb = ProkBertAdapter(
        checkpoint=str(b_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hb != h1  # different weights → different hash

    cfg_path = a_dir / "config.json"
    data = json.loads(cfg_path.read_text())
    data["num_hidden_layers"] = int(data["num_hidden_layers"]) + 1
    cfg_path.write_text(json.dumps(data))
    hc = ProkBertAdapter(
        checkpoint=str(a_dir), python_exe=sys.executable, device="cpu",
        weights_digest_strategy="file_sha256",
    ).model_hash()
    assert hc != h1  # arch field feeds the hash


def test_model_hash_hf_revision_no_path():
    a = ProkBertAdapter(
        checkpoint="neuralbioinfo/prokbert-mini", revision="deadbeef",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    h = a.model_hash()
    assert h.startswith("glmb:")
    b = ProkBertAdapter(
        checkpoint="neuralbioinfo/prokbert-mini", revision="feedface",
        weights_digest_strategy="hf_revision", python_exe=sys.executable,
    )
    assert b.model_hash() != h


def test_describe_echoes_config(adapter):
    d = adapter.describe()
    assert d["name"] == "prokbert"
    assert d["capabilities"] == ["embedding", "masked_marginal_llr", "tokenize"]
    assert d["backend"] == "local"
    assert d["embedding_dim"] == 32 and d["n_layers"] == 2
    assert d["max_context"] == 1020 and d["max_tokens"] == 1024
    assert d["pool_include_special"] is False
    assert d["trust_remote_code"] is True
    assert d["arch"]["family"] == "prokbert" and d["arch"]["model_type"] == "prokbert"


# --------------------------------------------------------------------------- #
# spec → adapter + dry-run command
# --------------------------------------------------------------------------- #


def test_from_spec_builds_local_adapter(hf_dir):
    spec = ModelSpec.model_validate(
        {
            "adapter": "prokbert",
            "adapter_version": "0.2.0",
            "model": {
                "checkpoint": str(hf_dir),
                "dtype": "float32",
                "max_context": 512,
                "code_revision": "abc123",
            },
            "weights_digest": {"strategy": "file_sha256"},
            "runner": {"backend": "local", "python_exe": sys.executable, "gpus": "cpu",
                       "extra": {"batch_size": 2}},
        }
    )
    a = ProkBertAdapter.from_spec(spec)
    assert a.backend == "local" and a.device == "cpu" and a.batch_size == 2
    assert a.max_context == 512 and a.trust_remote_code is True
    assert a.code_revision == "abc123"
    res = a.embed(SEQS, layers="last", pool="mean")
    assert len(res.arrays) == len(SEQS)


def test_from_spec_requires_checkpoint():
    spec = ModelSpec.model_validate(
        {"adapter": "prokbert", "adapter_version": "0.2.0", "model": {}}
    )
    with pytest.raises(ValueError, match="must set 'checkpoint'"):
        ProkBertAdapter.from_spec(spec)


def test_dry_run_emits_local_command(hf_dir, capsys):
    a = ProkBertAdapter(checkpoint=str(hf_dir), python_exe="/envs/loam/bin/python", dry_run=True)
    with pytest.raises(Exception):  # noqa: B017 - empty dry-run payload; we want the print
        a.embed(SEQS, layers="last", pool="mean")
    out = capsys.readouterr().out
    assert "/envs/loam/bin/python -m glmbench.runners.prokbert_runner" in out


# --------------------------------------------------------------------------- #
# tokenize — [CLS]…[SEP] wrapping + overlapping k-mer count
# --------------------------------------------------------------------------- #


def test_tokenize_wraps_cls_sep(adapter):
    toks = adapter.tokenize(["ATGTCCGCGGGACCT"])  # L=15, k=6,s=1 ⇒ 10 content + CLS + SEP
    assert toks[0][0] == CLS_ID and toks[0][-1] == SEP_ID
    assert len(toks[0]) == (15 - 6 + 1) + 2  # L−k+1 content tokens + [CLS] + [SEP]


def test_tokenize_case_invariant(adapter):
    assert adapter.tokenize(["ACGTACGTACGT"]) == adapter.tokenize(["acgtacgtacgt"])


# --------------------------------------------------------------------------- #
# embeddings — shapes, layers, spans (overlapping), pooling, batch-invariance, NaN
# --------------------------------------------------------------------------- #


def test_embed_pool_mean_shape(adapter):
    res = adapter.embed(SEQS, layers="last", pool="mean")
    assert res.pool == "mean" and res.token_spans is None
    assert res.layers == [2] and res.embedding_dim == 32  # n_layers+1=3 states ⇒ last index 2
    for arr in res.arrays:
        assert arr.shape == (1, 32) and np.isfinite(arr).all()


def test_embed_layers_all_one_pass_and_distinct(adapter):
    res = adapter.embed(SEQS, layers="all", pool="mean")
    assert res.layers == [0, 1, 2]  # n_layers+1 = 3 hidden states (standard BERT layout)
    for arr in res.arrays:
        assert arr.shape == (3, 32)
        assert not np.allclose(arr[0], arr[-1]), "all layers identical"


def test_embed_pool_none_spans_are_kmer_windows_union_covers_seq(adapter):
    """Overlapping k-mer spans: content token j → (j, j+6); union covers exactly len(seq)."""
    res = adapter.embed(SEQS, layers="all", pool="none")
    assert res.pool == "none" and res.token_spans is not None
    for seq, arr, spans in zip(SEQS, res.arrays, res.token_spans, strict=True):
        n_content = len(seq) - 6 + 1  # k=6, s=1
        assert len(spans) == n_content  # [CLS]/[SEP] excluded → one row per k-mer window
        assert arr.shape == (n_content, 3, 32)
        assert spans[0] == (0, 6) and spans[-1] == (n_content - 1, len(seq))
        # windows overlap (stride 1 < k 6) but their union is the whole sequence
        covered = set()
        for a, b in spans:
            covered.update(range(a, b))
        assert covered == set(range(len(seq)))


def test_embed_pool_cls_reads_first_position(adapter):
    res = adapter.embed(SEQS, layers="last", pool="cls")
    assert res.pool == "cls"
    for arr in res.arrays:
        assert arr.shape == (1, 32) and np.isfinite(arr).all()


def test_embed_pool_include_special_changes_mean(hf_dir):
    incl = ProkBertAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                           dtype="float32", weights_digest_strategy="file_sha256",
                           pool_include_special=True)
    excl = ProkBertAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                           dtype="float32", weights_digest_strategy="file_sha256",
                           pool_include_special=False)
    a = incl.embed(SEQS, layers="last", pool="mean")
    b = excl.embed(SEQS, layers="last", pool="mean")
    assert not np.allclose(a.arrays[0], b.arrays[0])


def test_embed_batch_size_invariant(hf_dir):
    a1 = ProkBertAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
                         dtype="float32", weights_digest_strategy="file_sha256", batch_size=1)
    aN = ProkBertAdapter(checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu",
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
                    "layer_ids": np.asarray([2], dtype=np.int64),
                    "embedding_dim": np.asarray(32, dtype=np.int64),
                },
                scratch_dir="x",
            )

    adapter._backend = _NaNBackend()
    with pytest.raises(WireProtocolError, match="NaN"):
        adapter.embed(SEQS, layers="last", pool="mean")


# --------------------------------------------------------------------------- #
# masked-marginal LLR (MASKED_MARGINAL_LLR) — the k-mer-generalized variant path
# --------------------------------------------------------------------------- #

REF = "ACGTACGTACGTACGTACGTACGT"  # L=24, k=6/s=1 ⇒ 19 content tokens


@pytest.fixture(scope="module")
def hf_dir_mlm(tmp_path_factory, hub_access):
    return _build_tiny_prokbert_dir(tmp_path_factory.mktemp("prokbertmlm") / "pb-mlm-tiny", mlm=True)


@pytest.fixture
def adapter_llr(hf_dir_mlm):
    return ProkBertAdapter(
        checkpoint=str(hf_dir_mlm), python_exe=sys.executable, device="cpu",
        dtype="float32", weights_digest_strategy="file_sha256",
    )


def _load_mlm_in_process(hf_dir_mlm: Path):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(str(hf_dir_mlm), trust_remote_code=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(hf_dir_mlm), trust_remote_code=True, padding_side="right")
    return model, tok


def test_affected_tokens_match_tokenizer_diff(hf_dir_mlm):
    """The analytic affected-window set == the tokens that actually change under substitution."""
    from glmbench.runners.prokbert_runner import _affected_content_tokens, _encode_full_ids

    _model, tok = _load_mlm_in_process(hf_dir_mlm)
    wt = _encode_full_ids(tok, [REF], max_tokens=1024)[0]
    n_content = len(wt) - 2
    for p in (0, 1, 5, 8, 12, len(REF) - 1):
        mut = "T" if REF[p] != "T" else "A"
        mref = REF[:p] + mut + REF[p + 1 :]
        mt = _encode_full_ids(tok, [mref], max_tokens=1024)[0]
        diff = [j for j in range(n_content) if wt[j + 1] != mt[j + 1]]
        assert _affected_content_tokens(p, 6, 1, n_content) == diff


def test_llr_roundtrip_finite_and_1to1(adapter_llr):
    items = [
        {"reference": REF, "mutations": [[8, "G", "A"]]},
        {"reference": REF, "mutations": [[8, "G", "T"], [12, "A", "C"]]},
    ]
    scores = adapter_llr.score_variant_llr(items)
    assert len(scores) == len(items) and all(np.isfinite(s) for s in scores)


def test_llr_equals_sum_over_affected_kmer_tokens(hf_dir_mlm):
    """White-box: LLR == Σ over affected k-mer tokens of (logit[mut_kmer] − logit[wt_kmer])."""
    from glmbench.runners.prokbert_runner import (
        _affected_content_tokens,
        _encode_full_ids,
        _score_variant_llr,
    )

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    p, mut = 8, "A"
    llr = float(
        _score_variant_llr(
            model, tok, [{"reference": REF, "mutations": [[p, REF[p], mut]]}],
            device=device, batch_size=8, max_tokens=1024,
        )["scores"][0]
    )
    wt_ids = _encode_full_ids(tok, [REF], max_tokens=1024)[0]
    mref = REF[:p] + mut + REF[p + 1 :]
    mut_ids = _encode_full_ids(tok, [mref], max_tokens=1024)[0]
    n_content = len(wt_ids) - 2
    aff = _affected_content_tokens(p, 6, 1, n_content)
    ids_masked = list(wt_ids)
    for j in aff:
        ids_masked[j + 1] = int(tok.mask_token_id)
    inp = torch.tensor([ids_masked], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0].float()
    expect = sum(
        float(logits[j + 1, int(mut_ids[j + 1])] - logits[j + 1, int(wt_ids[j + 1])]) for j in aff
    )
    assert llr == pytest.approx(expect, abs=1e-4)


def test_llr_multisite_equals_sum_of_single_sites(hf_dir_mlm):
    from glmbench.runners.prokbert_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    kw = dict(device=torch.device("cpu"), batch_size=8, max_tokens=1024)
    two = float(
        _score_variant_llr(
            model, tok, [{"reference": REF, "mutations": [[8, "G", "A"], [15, "A", "T"]]}], **kw
        )["scores"][0]
    )
    s1 = float(_score_variant_llr(model, tok, [{"reference": REF, "mutations": [[8, "G", "A"]]}], **kw)["scores"][0])
    s2 = float(_score_variant_llr(model, tok, [{"reference": REF, "mutations": [[15, "A", "T"]]}], **kw)["scores"][0])
    assert two == pytest.approx(s1 + s2, abs=1e-4)


def test_llr_caching_matches_naive_per_variant(hf_dir_mlm):
    from glmbench.runners.prokbert_runner import (
        _affected_content_tokens,
        _encode_full_ids,
        _score_variant_llr,
    )

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    items = [
        {"reference": REF, "mutations": [[8, "G", "A"]]},
        {"reference": REF, "mutations": [[8, "G", "T"], [15, "A", "C"]]},
        {"reference": REF, "mutations": [[3, "T", "C"]]},
    ]
    cached = _score_variant_llr(model, tok, items, device=device, batch_size=8, max_tokens=1024)["scores"]

    def _naive(item):
        wt = _encode_full_ids(tok, [item["reference"].upper()], max_tokens=1024)[0]
        n_content = len(wt) - 2
        total = 0.0
        for pos0, _w, mut in item["mutations"]:
            p = int(pos0)
            aff = _affected_content_tokens(p, 6, 1, n_content)
            ids_m = list(wt)
            for j in aff:
                ids_m[j + 1] = int(tok.mask_token_id)
            inp = torch.tensor([ids_m], dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0].float()
            mref = item["reference"].upper()
            mref = mref[:p] + mut.upper() + mref[p + 1 :]
            mut_ids = _encode_full_ids(tok, [mref], max_tokens=1024)[0]
            total += sum(
                float(logits[j + 1, int(mut_ids[j + 1])] - logits[j + 1, int(wt[j + 1])]) for j in aff
            )
        return total

    naive = np.array([_naive(it) for it in items], dtype=np.float64)
    np.testing.assert_allclose(cached, naive, rtol=0, atol=1e-4)


def test_llr_deterministic(adapter_llr):
    items = [{"reference": REF, "mutations": [[8, "G", "A"], [15, "A", "C"]]}]
    assert adapter_llr.score_variant_llr(items) == adapter_llr.score_variant_llr(items)


def test_llr_no_mask_token_fails_loud(hf_dir_mlm):
    from glmbench.runners.prokbert_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)

    class _NoMask:
        def __getattr__(self, k):
            return getattr(tok, k)

        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        _score_variant_llr(
            model, _NoMask(), [{"reference": REF, "mutations": [[8, "G", "A"]]}],
            device=torch.device("cpu"), batch_size=8, max_tokens=1024,
        )


# --------------------------------------------------------------------------- #
# bucketing order gate
# --------------------------------------------------------------------------- #
#
# The runner length-buckets its batches (`_batching.bucketed_batches`) and scatters every
# result back by INPUT index. A bad scatter is SILENT: every metric still computes, just on
# rows belonging to other sequences. These pin the invariant with a deliberately wide length
# spread, which is what makes the bucketer reorder aggressively — with a uniform-length input
# the identity permutation would hide the bug.

_WIDE = [
    "ACGTACGTACGT",  # 12
    "ATGCGATTCA" * 18,  # 180
    "TTGGCCAATTGGCCAAACGTACGTTTGGCCAATT",  # 34
    "GCGC" * 4,  # 16
    "ACGTTTGGACGTACGTCCAAACGTACGTACGTAC" * 6,  # 204
    "ACGTAC",  # 6 — exactly one content k-mer
    "AACCGGTTAACCGGTTAACC",  # 20
]


def _wide_adapter(hf_dir, **kw):
    return ProkBertAdapter(
        checkpoint=str(hf_dir), python_exe=sys.executable, device="cpu", dtype="float32",
        weights_digest_strategy="file_sha256", **kw,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_bucketing_preserves_input_order_for_embeddings(hf_dir, batch_size):
    """GATE: pooled embeddings under bucketing == embeddings computed one sequence at a time."""
    a = _wide_adapter(hf_dir, batch_size=batch_size)
    batched = a.embed(_WIDE, layers="all", pool="mean").arrays
    solo = [a.embed([s], layers="all", pool="mean").arrays[0] for s in _WIDE]
    assert len(batched) == len(_WIDE)
    for got, want in zip(batched, solo, strict=True):
        assert got.shape == want.shape == (3, 32)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)


def test_bucketing_preserves_input_order_for_pool_none(hf_dir):
    """GATE: per-token array i must have len(seq_i)−k+1 rows — a mis-scatter is a length swap."""
    a = _wide_adapter(hf_dir, batch_size=2)
    res = a.embed(_WIDE, layers="last", pool="none")
    assert res.token_spans is not None
    for seq, arr, spans in zip(_WIDE, res.arrays, res.token_spans, strict=True):
        n_content = len(seq) - 6 + 1  # k=6, shift=1
        assert arr.shape == (n_content, 1, 32)
        assert len(spans) == n_content
        assert spans[-1] == (n_content - 1, len(seq))
    # ...and the per-token rows themselves match the one-at-a-time computation.
    for i, seq in enumerate(_WIDE):
        solo = a.embed([seq], layers="last", pool="none").arrays[0]
        np.testing.assert_allclose(res.arrays[i], solo, rtol=0, atol=1e-4)


# LLR bucketing: the masked-forward jobs are (reference, position) pairs, so a wide spread of
# REFERENCE lengths is what makes the bucketer regroup them.
_REFS_WIDE = [
    "ACGTACGTACGTACGTACGTACGT",  # 24
    "ATGCGATTCA" * 15,  # 150
    "TTGGCCAATTGGCCAAACGTACGT",  # 24 (distinct content from the first)
    "ACGTTTGGACGTACGTCCAAACGTACGTACGTAC" * 5,  # 170
    "AACCGGTTAACCGGTTAA",  # 18
]
_ITEMS_WIDE = [
    {"reference": _REFS_WIDE[0], "mutations": [[8, _REFS_WIDE[0][8], "A"]]},
    {"reference": _REFS_WIDE[1], "mutations": [[100, _REFS_WIDE[1][100], "T"]]},
    {"reference": _REFS_WIDE[2], "mutations": [[0, _REFS_WIDE[2][0], "C"]]},
    {
        "reference": _REFS_WIDE[3],
        "mutations": [[9, _REFS_WIDE[3][9], "G"], [150, _REFS_WIDE[3][150], "T"]],
    },
    {"reference": _REFS_WIDE[4], "mutations": [[17, _REFS_WIDE[4][17], "C"]]},  # last nt
    {"reference": _REFS_WIDE[1], "mutations": [[3, _REFS_WIDE[1][3], "G"]]},  # ref reuse
]


@pytest.mark.parametrize("batch_size,token_budget", [(1, None), (2, None), (16, None), (16, 64)])
def test_bucketing_preserves_input_order_for_llr(hf_dir_mlm, batch_size, token_budget):
    """GATE: bucketed LLR scores == scores computed one variant at a time, in INPUT order."""
    from glmbench.runners.prokbert_runner import _score_variant_llr

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    batched = _score_variant_llr(
        model, tok, _ITEMS_WIDE, device=device, batch_size=batch_size,
        max_tokens=1024, token_budget=token_budget,
    )["scores"]
    solo = np.array(
        [
            float(
                _score_variant_llr(
                    model, tok, [it], device=device, batch_size=1, max_tokens=1024
                )["scores"][0]
            )
            for it in _ITEMS_WIDE
        ],
        dtype=np.float64,
    )
    assert len(batched) == len(_ITEMS_WIDE)
    np.testing.assert_allclose(batched, solo, rtol=0, atol=1e-4)


def test_llr_windowed_mutant_tokenization_matches_full_retokenization(hf_dir_mlm):
    """GATE: the O(L) window-splice of the mutant k-mer ids == the O(L²) full re-tokenization.

    The runner no longer re-tokenizes the whole mutated reference per (ref, pos, base); it
    tokenizes only the affected window and splices. This pins that shortcut against the naive
    path it replaced, on the WIDE (long) references where the shortcut actually bites.
    """
    from glmbench.runners.prokbert_runner import (
        _affected_content_tokens,
        _encode_full_ids,
        _score_variant_llr,
    )

    model, tok = _load_mlm_in_process(hf_dir_mlm)
    device = torch.device("cpu")
    got = _score_variant_llr(
        model, tok, _ITEMS_WIDE, device=device, batch_size=4, max_tokens=1024
    )["scores"]

    def _naive(item):  # independent impl: full re-tokenization, one forward per site
        wt = _encode_full_ids(tok, [item["reference"].upper()], max_tokens=1024)[0]
        n_content = len(wt) - 2
        total = 0.0
        for pos0, _w, mut in item["mutations"]:
            p = int(pos0)
            aff = _affected_content_tokens(p, 6, 1, n_content)
            ids_m = list(wt)
            for j in aff:
                ids_m[j + 1] = int(tok.mask_token_id)
            inp = torch.tensor([ids_m], dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits[0].float()
            mref = item["reference"].upper()
            mref = mref[:p] + mut.upper() + mref[p + 1 :]
            mut_ids = _encode_full_ids(tok, [mref], max_tokens=1024)[0]
            total += sum(
                float(logits[j + 1, int(mut_ids[j + 1])] - logits[j + 1, int(wt[j + 1])])
                for j in aff
            )
        return total

    naive = np.array([_naive(it) for it in _ITEMS_WIDE], dtype=np.float64)
    np.testing.assert_allclose(got, naive, rtol=0, atol=1e-4)


def test_rnagym_dms_fallback_end_to_end(adapter_llr):
    """RnagymDmsTask scored with ProkBERT falls back to the LLR path → OK + method recorded."""
    from glmbench.tasks.base import ResultStatus
    from glmbench.tasks.rnagym_dms import RnagymDmsTask

    tiny_manifest = (
        Path(__file__).resolve().parents[1] / "data" / "rnagym_tiny" / "tiny_manifest.yaml"
    )
    result = RnagymDmsTask(tiny_manifest).evaluate(adapter_llr)
    assert result.status is ResultStatus.OK
    assert result.metadata["scoring_method"] == "masked_marginal_llr"
    assert np.isfinite(result.metrics["macro_spearman"])
