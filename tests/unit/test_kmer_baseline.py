"""The model-free k-mer baseline (`glmbench.baselines.kmer`) and the floor it defines.

The floor is a line drawn across every bar on the board, so its failure modes are all quiet
ones: a vector that is not actually composition, a "floor" that silently varies with depth (it
must not — that identity is what lets a layer-sweep task inherit its twin's number), or a
baseline that answers a task it cannot serve. Each is pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from glmbench.adapters.base import Capability, missing_capability_methods
from glmbench.baselines.kmer import KmerCompositionAdapter, kmer_frequency_vector


def test_it_is_composition_and_nothing_else():
    """Shuffle the sequence and the vector is unchanged — the defining property.

    If this ever fails, the "floor" has started measuring order, and every headroom number
    computed against it is overstated.
    """
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=4000))
    shuffled = "".join(rng.permutation(list(seq)))
    # k=1 is exactly order-free; higher k is order-free only up to window overlap, so the
    # invariant is stated where it is exact.
    assert np.allclose(kmer_frequency_vector(seq, 1), kmer_frequency_vector(shuffled, 1))
    assert not np.allclose(kmer_frequency_vector(seq, 4), kmer_frequency_vector(shuffled, 4))


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_the_vector_is_a_normalized_frequency_over_4_to_the_k(k):
    v = kmer_frequency_vector("ACGT" * 500, k)
    assert v.shape == (4**k,)
    assert v.sum() == pytest.approx(1.0)
    assert (v >= 0).all()


def test_counts_are_the_obvious_ones():
    v = kmer_frequency_vector("AAAC", 2)  # AA, AA, AC
    assert v[0] == pytest.approx(2 / 3)  # AA
    assert v[1] == pytest.approx(1 / 3)  # AC
    assert v.sum() == pytest.approx(1.0)


def test_non_acgt_windows_are_dropped_never_bucketed():
    """A window that is partly unknown is not evidence about any k-mer.

    Folding `N` into (say) `A` would give an ambiguity-rich sequence a composition it does not
    have — and the essentiality corpus has plenty of them. `ACGTNNACGT` therefore yields only
    the six windows fully inside the two `ACGT` runs (AC, CG, GT twice each) — and NOT the `TA`
    that closing the gap would manufacture.
    """
    v = kmer_frequency_vector("ACGTNNACGT", 2)
    ac, cg, gt, ta = 0b0001, 0b0110, 0b1011, 0b1100
    assert v[[ac, cg, gt]] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert v[ta] == 0.0, "dropping the N windows must not splice the two runs together"
    assert v.sum() == pytest.approx(1.0)


def test_an_uncountable_sequence_is_zero_not_a_divide_by_zero():
    for seq, k in [("", 3), ("AC", 3), ("NNNNNN", 3)]:
        v = kmer_frequency_vector(seq, k)
        assert v.shape == (64,)
        assert not v.any()
        assert np.isfinite(v).all(), "a NaN here would poison a whole task's metric"


# --- the adapter contract ---------------------------------------------------------------


def test_it_declares_embedding_only_so_lm_head_tasks_return_na():
    """Likelihood tasks such as `rnagym-dms` must get a first-class N/A.

    A frequency vector is not a likelihood model. Declaring more than EMBEDDING would produce a
    number on those tasks, and a meaningless number on a leaderboard is worse than a gap.
    """
    assert set(KmerCompositionAdapter.CAPABILITIES) == {Capability.EMBEDDING}
    assert missing_capability_methods(KmerCompositionAdapter) == []
    with pytest.raises(NotImplementedError):
        KmerCompositionAdapter(k=3).score_sequences(["ACGT"])
    with pytest.raises(NotImplementedError):
        KmerCompositionAdapter(k=3).logprob_embedding(["ACGT"])


def test_the_baseline_is_flat_across_layers():
    """The identity that lets a layer-sweep task inherit its single-layer twin's floor.

    A sweep is a max over layers. There is no depth here, so every tap is the same vector and
    the max is the single value — `scripts/kmer_floor.py` leans on this rather than paying for
    a second (in the EC case, multi-hour) run.
    """
    ad = KmerCompositionAdapter(k=3, n_layers=5)
    res = ad.embed(["ACGTACGTAC" * 20], layers="all")
    assert res.layers == [0, 1, 2, 3, 4, 5], "layer 0 is the embedding output, then one per block"
    arr = res.arrays[0]
    assert arr.shape == (6, 64)
    assert np.allclose(arr, arr[0]), "every tap must be identical, or inherited floors are wrong"

    last = ad.embed(["ACGTACGTAC" * 20], layers="last")
    assert np.allclose(last.arrays[0][0], arr[0])


def test_batch_first_and_one_to_one_with_inputs():
    seqs = ["ACGT" * 30, "GGGG" * 30, "N" * 40]
    res = KmerCompositionAdapter(k=2).embed(seqs)
    assert len(res.arrays) == len(seqs)
    assert res.embedding_dim == 16
    assert not res.arrays[2].any(), "an all-N sequence embeds to zero, not to garbage"


def test_pool_none_fails_loud_rather_than_inventing_token_spans():
    with pytest.raises(ValueError, match="no tokenization"):
        KmerCompositionAdapter(k=2).embed(["ACGT"], pool="none")


def test_the_hash_separates_ks():
    assert KmerCompositionAdapter(k=3).model_hash() != KmerCompositionAdapter(k=4).model_hash()
    assert "k3" in KmerCompositionAdapter(k=3).model_hash()


def test_it_is_not_in_the_adapter_registry():
    """A floor is a line across the board, not a competitor on it.

    Registered, it could be run into `results/` by the ordinary CLI and would then appear as a
    model row — ranked against the models it exists to disqualify.
    """
    from glmbench import registry

    assert "kmer-composition" not in registry.list("adapter")
