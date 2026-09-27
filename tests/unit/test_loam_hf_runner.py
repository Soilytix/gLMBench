"""Runner tests for the ``loam-hf`` adapter: a tiny LOAM served end to end on CPU.

A 2-block, width-32 LOAM with random weights is built from the release's own
``modeling_loam.py`` and served through the subprocess runner; every op is checked against a
direct computation with the same model. Needs torch + transformers. The modeling code and
tokenizer come from ``tests/data/loam_hf_tiny`` (the ``Soilytix/LOAM-*`` release files, with
the modeling code's comments trimmed but its code unchanged, next to a tiny ``config.json``),
or from any LOAM export directory named by ``$GLMBENCH_LOAM_HF_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from glmbench.adapters.loam_hf import LOAMHFAdapter

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[2]
RELEASE_DIR = Path(
    os.environ.get("GLMBENCH_LOAM_HF_DIR") or REPO / "tests" / "data" / "loam_hf_tiny"
)


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-block, width-32 LOAM with random weights, using the release's own modeling code."""
    d = tmp_path_factory.mktemp("tiny_loam")
    for name in ("configuration_loam.py", "modeling_loam.py", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(RELEASE_DIR / name, d / name)
    cfg = json.loads((RELEASE_DIR / "config.json").read_text())
    cfg.update(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
               head_dim=16, intermediate_size=64, max_position_embeddings=128)
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(
        transformers.AutoConfig.from_pretrained(d, trust_remote_code=True), trust_remote_code=True
    )
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    return d


@pytest.fixture(scope="module")
def adapter(tiny_export: Path) -> LOAMHFAdapter:
    return LOAMHFAdapter(checkpoint=str(tiny_export), python_exe=sys.executable, device="cpu")


@pytest.fixture(scope="module")
def reference(tiny_export: Path):  # noqa: ANN201
    model = transformers.AutoModelForCausalLM.from_pretrained(
        tiny_export, trust_remote_code=True, dtype=torch.float32
    ).eval()
    tok = transformers.AutoTokenizer.from_pretrained(tiny_export, trust_remote_code=True)
    return model, tok


SEQS = ["ACGTACGTTGCA", "GGGCCCAAATTTACG", "ATG", "TTTTACGATCGATCGATGCTAGCTAGTCGAT"]


def _direct(reference, seq: str):  # noqa: ANN001, ANN202
    model, tok = reference
    ids = torch.tensor([[model.config.bos_token_id] + tok(seq, add_special_tokens=False)["input_ids"]])
    taps: list = []
    hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: taps.append(o))]
    hooks += [b.register_forward_hook(lambda m, i, o: taps.append(o)) for b in model.model.layers]
    with torch.inference_mode():
        logits = model(input_ids=ids).logits.float()
    for h in hooks:
        h.remove()
    logp = torch.log_softmax(logits[0, :-1], -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
    layers = torch.stack(taps)[:, 0, 1:]  # [L+1, n, H], BOS dropped
    return layers.numpy(), logp.numpy()


def test_embed_all_layers_matches_direct_hooks(adapter: LOAMHFAdapter, reference) -> None:  # noqa: ANN001
    res = adapter.embed(SEQS, layers="all", pool="mean")
    assert res.layers == [0, 1, 2]
    for seq, arr in zip(SEQS, res.arrays, strict=True):
        layers, _ = _direct(reference, seq)
        np.testing.assert_allclose(arr, layers.mean(axis=1), rtol=1e-5, atol=1e-6)


def test_last_is_exactly_a_slice_of_all(adapter: LOAMHFAdapter) -> None:
    last = adapter.embed(SEQS, layers="last", pool="mean")
    full = adapter.embed(SEQS, layers="all", pool="mean")
    assert last.layers == [2]
    for a, b in zip(last.arrays, full.arrays, strict=True):
        np.testing.assert_array_equal(a[0], b[-1])


def test_last_layer_is_the_raw_block_output_not_the_final_norm(adapter: LOAMHFAdapter, reference) -> None:  # noqa: ANN001
    model, tok = reference
    seq = SEQS[3]
    ids = torch.tensor([[model.config.bos_token_id] + tok(seq, add_special_tokens=False)["input_ids"]])
    with torch.inference_mode():
        normed = model.model(input_ids=ids).last_hidden_state[0, 1:].mean(0).numpy()
    ours = adapter.embed([seq], layers="last", pool="mean").arrays[0][0]
    assert not np.allclose(ours, normed, rtol=1e-3), "embed returned the final-normed state"


def test_pool_none_returns_one_nucleotide_span_per_token(adapter: LOAMHFAdapter) -> None:
    res = adapter.embed(SEQS[:2], layers=[0, -1], pool="none")
    for seq, arr, spans in zip(SEQS[:2], res.arrays, res.token_spans, strict=True):  # type: ignore[arg-type]
        assert arr.shape == (len(seq), 2, 32)
        assert spans == [(i, i + 1) for i in range(len(seq))]


def test_scores_and_logprobs_match_direct_computation(adapter: LOAMHFAdapter, reference) -> None:  # noqa: ANN001
    scores = adapter.score_sequences(SEQS)
    zs = adapter.logprob_embedding(SEQS)
    for seq, s, z in zip(SEQS, scores, zs, strict=True):
        _, logp = _direct(reference, seq)
        np.testing.assert_allclose(z, logp, rtol=1e-5, atol=1e-6)
        assert s == pytest.approx(float(logp.mean()), abs=1e-5)
        assert float(np.mean(z)) == pytest.approx(s, abs=1e-5)


def test_fused_embed_logprob_equals_the_separate_ops(adapter: LOAMHFAdapter) -> None:
    emb, zs = adapter.embed_with_logprob(SEQS, layers="all", pool="mean")
    emb2 = adapter.embed(SEQS, layers="all", pool="mean")
    zs2 = adapter.logprob_embedding(SEQS)
    for a, b in zip(emb.arrays, emb2.arrays, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(zs, zs2, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_tokenize_is_one_token_per_nucleotide(adapter: LOAMHFAdapter) -> None:
    for seq, ids in zip(SEQS, adapter.tokenize(SEQS), strict=True):
        assert len(ids) == len(seq)
