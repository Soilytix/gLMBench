"""Gates for the shared length-bucketed batching helper (`runners/_batching.py`).

The load-bearing property is **permutation**: runners scatter results back to input order by
index, so if `bucketed_batches` ever dropped, duplicated, or invented an index, a runner
would silently return one sequence's embedding under another sequence's label — a
wrong-answer bug that no downstream metric would flag. Every other property here is a
performance claim; this one is a correctness gate.

Pure CPU, no torch, no model — runs in the torch-free CI.
"""

from __future__ import annotations

import random

import pytest

from glmbench.runners._batching import bucketed_batches, padding_efficiency

# --------------------------------------------------------------------------- #
# the gate: every index exactly once
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 1001])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_budget": 4096},
        {"batch_size": 8},
        {"token_budget": 4096, "batch_size": 8},
        {"token_budget": 4096, "pad_multiple": 128},
    ],
)
def test_gate_is_a_permutation(n, kwargs):
    """GATE: the concatenated batches are exactly a permutation of range(n)."""
    rng = random.Random(0xC0FFEE)
    lengths = [rng.randint(1, 9000) for _ in range(n)]
    batches = bucketed_batches(lengths, **kwargs)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(n)), "indices dropped, duplicated, or invented"
    assert all(b for b in batches), "no empty batches"


def test_gate_permutation_with_degenerate_lengths():
    """GATE: zero-length and all-equal inputs still round-trip every index."""
    for lengths in ([0, 0, 0], [5] * 10, [0, 1, 0, 1]):
        batches = bucketed_batches(lengths, token_budget=8)
        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


# --------------------------------------------------------------------------- #
# the budget is respected (this is what stops the OOM cliff)
# --------------------------------------------------------------------------- #


def test_token_budget_is_respected():
    rng = random.Random(7)
    lengths = [rng.randint(50, 3000) for _ in range(500)]
    budget = 8192
    for b in bucketed_batches(lengths, token_budget=budget):
        width = max(lengths[i] for i in b)
        if len(b) > 1:  # a lone oversized sequence is allowed to exceed (see below)
            assert len(b) * width <= budget


def test_oversized_sequence_is_emitted_alone_not_dropped():
    """A sequence longer than the budget must still be scored — alone, never dropped."""
    lengths = [10, 10, 50_000, 10]
    batches = bucketed_batches(lengths, token_budget=1024)
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3]
    big = [b for b in batches if 2 in b]
    assert big == [[2]], "the oversized sequence must be alone in its own batch"


def test_batch_size_cap_is_respected():
    lengths = [10] * 100
    for b in bucketed_batches(lengths, batch_size=8):
        assert len(b) <= 8


def test_needs_a_bound():
    with pytest.raises(ValueError, match="bound its batches"):
        bucketed_batches([1, 2, 3])


def test_pad_multiple_rounds_up():
    """With pad_multiple, the budget accounts for the ROUNDED width (else we'd OOM)."""
    lengths = [1] * 10
    batches = bucketed_batches(lengths, token_budget=128, pad_multiple=128)
    # each row costs a full 128 slots -> exactly one row per batch
    assert all(len(b) == 1 for b in batches)


# --------------------------------------------------------------------------- #
# the performance claim: bucketing removes the padding waste
# --------------------------------------------------------------------------- #


def test_bucketing_beats_input_order_on_a_realistic_corpus():
    """The performance claim, as a test: ~63% -> ~100% token efficiency at batch 4.

    Length distribution mirrors the real bacbench-essentiality windows (median ~950 nt,
    p99 ~3.4 kb, a long tail to the 8192 context).
    """
    rng = random.Random(1234)
    lengths = [min(8192, max(60, int(rng.lognormvariate(6.9, 0.6)))) for _ in range(5000)]

    in_order = [list(range(i, min(i + 4, len(lengths)))) for i in range(0, len(lengths), 4)]
    bucketed = bucketed_batches(lengths, batch_size=4)

    eff_before = padding_efficiency(lengths, in_order)
    eff_after = padding_efficiency(lengths, bucketed)

    assert eff_before < 0.80, f"expected the unsorted corpus to waste tokens, got {eff_before}"
    assert eff_after > 0.97, f"bucketing should ~eliminate padding, got {eff_after}"
    assert eff_after > eff_before
