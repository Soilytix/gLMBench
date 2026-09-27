"""Length-bucketed, token-budget batching — shared by every runner.

**Why bucket.** Sequences reach a runner in input order (`read_sequences_jsonl` preserves it,
no task length-sorts), and every runner pads a batch to its longest member. A gene corpus
runs 188 → 8192 nt with a ~950 nt median, so a contiguous slice mixes a 200-nt window with
an 8-kb one and the whole batch pays 8 kb of compute. Measured on the real essentiality
corpus, only **63%** of the token slots at ``batch_size=4`` hold a real token (36% at 32).
On a compute-bound GPU every pad token is wasted FLOPs — sorting by length before batching
drives padding to ~0 and was worth **1.44×** on the LOAM logprob path.

**Why a token budget, not a sequence count.** Cost and memory scale with
``batch × padded_len``, not with the number of sequences: 16 × 900 nt and 16 × 8192 nt differ
~9× in both. That is why the shipped specs carry hand-tuned magic numbers (1, 4, 8, 16) that
are simultaneously too small for short windows and OOM-prone on long ones. A token budget is
a physical fact about the card and adapts on its own — short windows get big batches, long
windows get small ones.

**Contract.** These return **index** batches, never reordered data. The caller MUST scatter
results back to input order. ``bucketed_batches`` is always a permutation of ``range(n)`` —
every index appears exactly once — which is what makes the reordering safe and is asserted by
the unit gate (``test_batching.py``).

**Numerics.** Regrouping rows cannot change a real token's output under a causal mask with
right-padding: position *i* attends only to positions ≤ *i*, and pads sit to the right of
every real token, so no real position ever attends to one. Only GEMM tiling shifts, i.e.
float noise at the last bits (measured max |Δ| = 3.4e-5 vs the unbucketed path).
"""

from __future__ import annotations

from collections.abc import Sequence


def bucketed_batches(
    lengths: Sequence[int],
    *,
    token_budget: int | None = None,
    batch_size: int | None = None,
    pad_multiple: int = 1,
) -> list[list[int]]:
    """Group indices into length-sorted batches; return them as index lists.

    Args:
        lengths: per-sequence token length (excluding any BOS/EOS the caller adds — pass
            the padded-tensor width you will actually build if that differs).
        token_budget: cap on ``len(batch) * padded_len`` for each batch. The dominant knob:
            it bounds the real tensor, so it bounds both compute and memory.
        batch_size: optional hard cap on rows per batch, applied on top of the budget. Pass
            this alone to get "length-sorted, fixed batch size" (padding removed, batch
            count unchanged) — the drop-in upgrade for a runner that has a `batch_size` knob.
        pad_multiple: round each batch's padded length up to a multiple of this (for models
            whose torso requires it, e.g. a U-Net that downsamples 2^k×).

    Returns:
        A list of index batches. Concatenated, exactly a permutation of ``range(len(lengths))``.

    Raises:
        ValueError: if neither ``token_budget`` nor ``batch_size`` is given (a runner must
            bound its batches by something — silently defaulting is how OOM cliffs happen).

    Note:
        A single sequence longer than ``token_budget`` is still emitted, alone, in its own
        batch. Dropping it would silently lose a labeled item; the budget is a target, and
        one oversized sequence is the caller's (and the model's) problem, not ours.
    """
    if token_budget is None and batch_size is None:
        raise ValueError(
            "bucketed_batches needs a token_budget and/or a batch_size — a runner must "
            "bound its batches explicitly."
        )
    if token_budget is not None and token_budget < 1:
        raise ValueError(f"token_budget must be >= 1, got {token_budget}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if pad_multiple < 1:
        raise ValueError(f"pad_multiple must be >= 1, got {pad_multiple}")

    def _padded(n: int) -> int:
        if pad_multiple == 1:
            return n
        return ((n + pad_multiple - 1) // pad_multiple) * pad_multiple

    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0

    for i in order:
        width = _padded(lengths[i])
        new_max = max(cur_max, width)
        n_rows = len(cur) + 1
        over_budget = token_budget is not None and n_rows * new_max > token_budget
        over_rows = batch_size is not None and n_rows > batch_size
        if cur and (over_budget or over_rows):
            batches.append(cur)
            cur, cur_max = [i], width
        else:
            cur.append(i)
            cur_max = new_max

    if cur:
        batches.append(cur)
    return batches


def padding_efficiency(lengths: Sequence[int], batches: list[list[int]]) -> float:
    """Fraction of token slots that hold a real token — 1.0 means zero padding waste.

    Diagnostic for logs/tests: it is what turns "the batching is bad" into a number.
    """
    real = padded = 0
    for b in batches:
        if not b:
            continue
        width = max(lengths[i] for i in b)
        real += sum(lengths[i] for i in b)
        padded += width * len(b)
    return (real / padded) if padded else 1.0
