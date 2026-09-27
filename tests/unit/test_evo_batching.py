"""Order gates for the evo runner's bucketed batching.

`test_evo.py` drives the ADAPTER through fake backends and never imports torch, so it does
not execute `evo_runner`'s batch loops at all — and Evo 1.5 is a 7B StripedHyena needing
flash-attn, so we cannot load the real thing in a unit test either. That would leave the
bucketing in `_per_token` / `_embed` with no execution coverage.

So: drive the real runner functions against a **fake model shaped like StripedHyena**
(`model.backbone.blocks`, `model.config.hidden_size`, logits from `__call__`). That exercises
the exact code under test — the index scatter, the hook capture, the pooling — while keeping
the model trivial and deterministic.

The property under test is ORDER: bucketing reorders sequences into batches and scatters
results back by index. A bad scatter returns sequence A's vector under sequence B's label and
NOTHING fails loudly. Every assertion here compares batched output against one-at-a-time
output, with a deliberately wide length spread so the bucketer reorders aggressively.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from glmbench.runners import evo_runner  # noqa: E402

H = 8
VOCAB = 512


class _Block(nn.Module):
    """Pointwise, deterministic, token-dependent — so a mis-scatter changes the values."""

    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, x, *args, **kwargs):
        return x * self.k + 1.0


class _Backbone(nn.Module):
    def __init__(self, n_blocks: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(VOCAB, H)
        self.blocks = nn.ModuleList([_Block(1.0 + 0.1 * i) for i in range(n_blocks)])


class _Cfg:
    hidden_size = H


class _FakeEvo(nn.Module):
    """Minimal stand-in: same attribute paths the runner hooks into."""

    def __init__(self, n_blocks: int = 3):
        super().__init__()
        self.backbone = _Backbone(n_blocks)
        self.config = _Cfg()
        self.lm_head = nn.Linear(H, VOCAB, bias=False)

    def forward(self, input_ids=None, **kwargs):
        x = self.backbone.embedding_layer(input_ids)
        for b in self.backbone.blocks:
            x = b(x)

        class _Out:
            pass

        o = _Out()
        o.logits = self.lm_head(x)
        return o


@pytest.fixture(scope="module")
def fake():
    torch.manual_seed(0)
    m = _FakeEvo().eval()
    # Wide, shuffled length spread — this is what makes bucketing reorder aggressively.
    seqs = ["ACGT" * 1, "ACGT" * 40, "TT", "GCGC" * 7, "A" * 150, "ACG", "TTTT" * 25]
    return m, seqs


@pytest.mark.parametrize("batch_size,token_budget", [(4, None), (2, None), (8, 64)])
def test_gate_per_token_results_stay_in_input_order(fake, batch_size, token_budget):
    """GATE: z[i] belongs to seqs[i] — length is the tell, values are the proof."""
    model, seqs = fake
    batched = evo_runner._per_token(
        model, seqs, device=torch.device("cpu"), score_prefix="eos", eod_token_id=0,
        max_tokens=None, batch_size=batch_size, token_budget=token_budget,
    )
    solo = [
        evo_runner._per_token(
            model, [s], device=torch.device("cpu"), score_prefix="eos", eod_token_id=0,
            max_tokens=None, batch_size=1,
        )[0]
        for s in seqs
    ]
    assert [len(z) for z in batched] == [len(s) for s in seqs], "results permuted (length swap)"
    for i, (b, s) in enumerate(zip(batched, solo)):
        np.testing.assert_allclose(b, s, atol=1e-5, err_msg=f"seq {i} value mismatch")


@pytest.mark.parametrize("pool", ["mean", "max", "last"])
@pytest.mark.parametrize("layers", ["last", "all"])
def test_gate_embed_results_stay_in_input_order(fake, pool, layers):
    """GATE: pooled embeddings under bucketing == one-at-a-time, row for row."""
    model, seqs = fake
    batched = evo_runner._embed(
        model, seqs, device=torch.device("cpu"), layers_req=layers, pool=pool,
        max_tokens=None, batch_size=4,
    )["arrays"]
    solo = np.concatenate(
        [
            evo_runner._embed(
                model, [s], device=torch.device("cpu"), layers_req=layers, pool=pool,
                max_tokens=None, batch_size=1,
            )["arrays"]
            for s in seqs
        ],
        axis=0,
    )
    assert batched.shape == solo.shape
    np.testing.assert_allclose(batched, solo, atol=1e-5)


def test_gate_embed_pool_none_keeps_ragged_rows_keyed_to_input(fake):
    """GATE: ragged pool='none' arrays stay keyed to the INPUT index."""
    model, seqs = fake
    payload = evo_runner._embed(
        model, seqs, device=torch.device("cpu"), layers_req="last", pool="none",
        max_tokens=None, batch_size=4,
    )
    assert int(payload["n"]) == len(seqs)
    for i, s in enumerate(seqs):
        assert payload[f"arr_{i}"].shape[0] == len(s), f"seq {i}: row length != len(seq) — permuted"
