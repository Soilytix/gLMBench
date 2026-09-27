"""Gates for cross-task forward-pass reuse (adapters/reuse.py).

The claim being gated: a single-layer task and its layer sweep read the same corpus, so they
can share ONE forward pass, and each gets exactly what it would have computed on its own. Two
ways that could go silently wrong, both pinned here:

1. **The slice is the wrong layer.** `embed(layers="last")` is served from a cached
   `layers="all"` result. Slice the wrong index and every downstream probe still fits — on
   the wrong representation.
2. **The cache serves the wrong corpus.** It is keyed on the sequences, so a task in the same
   group that reads different windows must miss and compute its own forward.

And the property that protects independent runs: with only ONE task in a group, nothing is
promoted, nothing is cached, and the adapter is called exactly as before.

Model-free and torch-free: the inner model is the echo double, whose layers all differ, so a
wrong row changes the numbers. That a real model's `"last"` IS a row of its `"all"`, and that
a fused `embed_with_logprob` equals the two separate ops, are properties of each adapter and
are gated with it (`test_loam_hf_runner.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

import glmbench.tasks  # noqa: F401  — import for the side effect: registers the tasks
from glmbench import registry
from glmbench.adapters.base import Capability, EmbeddingResult
from glmbench.adapters.echo import EchoAdapter
from glmbench.adapters.reuse import (
    GroupPlan,
    ReusingAdapter,
    _select_layers,
    corpus_key,
    plan_fusion,
)

SEQS = [
    "ATGACGTACGTACGTACGTA",
    "ATGCCCGGGTTTAAACCCGGGTTTAAAGGGCCCAT",
    "ATG" + "ACGTACGTGC" * 4 + "TAA",
    "ACGTTTGCA",
]


@pytest.fixture(scope="module")
def adapter() -> EchoAdapter:
    return EchoAdapter()


# --------------------------------------------------------------------------- #
# GATE 1 — the layer slice is the RIGHT layer
# --------------------------------------------------------------------------- #


def test_gate_select_layers_refuses_a_layer_it_never_computed():
    """GATE: asking for a layer the cached pass doesn't have must RAISE, not silently substitute."""
    res = EmbeddingResult(
        arrays=[np.zeros((2, 4), dtype=np.float32)], layers=[3, 4], pool="mean",
        token_spans=None, embedding_dim=4,
    )
    with pytest.raises(KeyError, match="refusing to serve a different layer"):
        _select_layers(res, [0])


def test_select_layers_handles_explicit_and_negative_indices():
    res = EmbeddingResult(
        arrays=[np.arange(3 * 4, dtype=np.float32).reshape(3, 4)],
        layers=[0, 1, 2], pool="mean", token_spans=None, embedding_dim=4,
    )
    np.testing.assert_allclose(_select_layers(res, [1]).arrays[0], res.arrays[0][[1]])
    np.testing.assert_allclose(_select_layers(res, [-1]).arrays[0], res.arrays[0][[2]])
    np.testing.assert_allclose(_select_layers(res, "last").arrays[0], res.arrays[0][[2]])
    assert _select_layers(res, "last").layers == [2]
    assert _select_layers(res, "all").layers == [0, 1, 2]


def test_select_layers_slices_the_layer_axis_of_per_token_arrays():
    """pool='none' arrays are [n_tok, n_layers, dim]: the slice is on axis 1, not axis 0."""
    arr = np.arange(5 * 3 * 2, dtype=np.float32).reshape(5, 3, 2)
    res = EmbeddingResult(
        arrays=[arr], layers=[0, 1, 2], pool="none",
        token_spans=[[(i, i + 1) for i in range(5)]], embedding_dim=2,
    )
    out = _select_layers(res, "last")
    np.testing.assert_array_equal(out.arrays[0], arr[:, [2]])
    assert out.token_spans == res.token_spans


# --------------------------------------------------------------------------- #
# GATE 2 — the reuse wrapper end to end: same answers, fewer forwards
# --------------------------------------------------------------------------- #


class _CountingAdapter:
    """Wraps the real adapter and counts how many forwards each op actually costs."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[str, object]] = []

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def capabilities(self):
        return self._inner.capabilities()

    def embed(self, sequences, *, layers="last", pool="mean"):
        self.calls.append(("embed", layers))
        return self._inner.embed(sequences, layers=layers, pool=pool)

    def logprob_embedding(self, sequences):
        self.calls.append(("logprob_embedding", None))
        return self._inner.logprob_embedding(sequences)


def _sweep_plan():
    """The `loam_paper` shape: a single-layer task + its sweep (embed all), one corpus."""
    return {
        "g": GroupPlan(
            name="g", needs_all_layers=True, needs_logprob=False,
            n_embed_consumers=2, n_logprob_consumers=0,
        )
    }


@pytest.mark.parametrize("sweep_first", [False, True])
def test_gate_base_and_sweep_share_one_forward_with_the_same_answers(adapter, sweep_first):
    """GATE: base + sweep over one corpus == ONE all-layer forward; base gets the LAST row."""
    want_all = adapter.embed(SEQS, layers="all", pool="mean")
    # The echo layers differ, so serving the first row instead of the last would show.
    assert not np.allclose(want_all.arrays[0][0], want_all.arrays[0][-1])

    counting = _CountingAdapter(adapter)
    reuse = ReusingAdapter(counting, _sweep_plan())
    got = {}
    for task in (["sweep", "base"] if sweep_first else ["base", "sweep"]):
        reuse.begin_task("g")
        got[task] = reuse.embed(SEQS, layers="all" if task == "sweep" else "last", pool="mean")

    assert counting.calls == [("embed", "all")], f"expected ONE all-layer forward: {counting.calls}"
    assert got["sweep"].layers == want_all.layers
    for a, b in zip(got["sweep"].arrays, want_all.arrays, strict=True):
        np.testing.assert_array_equal(a, b)
    assert got["base"].layers == [want_all.layers[-1]]
    for a, b in zip(got["base"].arrays, want_all.arrays, strict=True):
        np.testing.assert_array_equal(a, b[-1:])


def test_gate_a_different_corpus_in_the_same_group_misses_and_recomputes(adapter):
    """GATE: the cache is keyed on the SEQUENCES, so a corpus mismatch never serves stale rows."""
    counting = _CountingAdapter(adapter)
    reuse = ReusingAdapter(counting, _sweep_plan())

    reuse.begin_task("g")
    reuse.embed(SEQS, layers="all", pool="mean")
    reuse.begin_task("g")
    other = ["ACGTACGTAC", "TTTTGGGGCC"]  # same group label, DIFFERENT corpus
    got = reuse.embed(other, layers="all", pool="mean")

    assert counting.calls == [("embed", "all"), ("embed", "all")]
    direct = adapter.embed(other, layers="all", pool="mean")
    for a, b in zip(got.arrays, direct.arrays, strict=True):
        np.testing.assert_array_equal(a, b)
    assert corpus_key(SEQS) != corpus_key(other)


def test_gate_logprob_consumer_takes_both_heads_from_one_fused_forward(adapter):
    """GATE: when a group also wants log-probs and the adapter can fuse, ONE call serves all."""

    class _Fusing(_CountingAdapter):
        def embed_with_logprob(self, sequences, *, layers="last", pool="mean"):
            self.calls.append(("embed_with_logprob", layers))
            return (
                self._inner.embed(sequences, layers=layers, pool=pool),
                self._inner.logprob_embedding(sequences),
            )

    plan = {"g": GroupPlan("g", needs_all_layers=True, needs_logprob=True,
                           n_embed_consumers=2, n_logprob_consumers=1)}
    fusing = _Fusing(adapter)
    reuse = ReusingAdapter(fusing, plan)
    reuse.begin_task("g")
    reuse.embed(SEQS, layers="last", pool="mean")
    reuse.begin_task("g")
    z = reuse.logprob_embedding(SEQS)
    reuse.begin_task("g")
    reuse.embed(SEQS, layers="all", pool="mean")

    assert fusing.calls == [("embed_with_logprob", "all")]
    for a, b in zip(z, adapter.logprob_embedding(SEQS), strict=True):
        np.testing.assert_array_equal(a, b)


def test_gate_adapter_without_the_fused_op_falls_back_cleanly(adapter):
    """GATE: an adapter that can't fuse (echo has no `embed_with_logprob`) still works."""
    assert not hasattr(adapter, "embed_with_logprob")
    plan = {"g": GroupPlan("g", needs_all_layers=True, needs_logprob=True,
                           n_embed_consumers=1, n_logprob_consumers=1)}
    counting = _CountingAdapter(adapter)
    reuse = ReusingAdapter(counting, plan)
    reuse.begin_task("g")
    got = reuse.embed(SEQS, layers="all", pool="mean")
    reuse.begin_task("g")
    z = reuse.logprob_embedding(SEQS)

    # a separate log-prob forward — correct, just not fused
    assert counting.calls == [("embed", "all"), ("logprob_embedding", None)]
    assert len(got.arrays) == len(SEQS)
    assert [len(x) for x in z] == [len(s) for s in SEQS]


# --------------------------------------------------------------------------- #
# GATE 3 — the plan comes from the tasks being run; a task ALONE is unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base", "sweep", "group"),
    [
        ("bacbench-essentiality", "bacbench-essentiality-layer-sweep",
         "bacbench-essentiality-windows"),
        ("dgeb-ec-classification-dna", "dgeb-ec-classification-dna-layer-sweep",
         "dgeb-ec-windows"),
    ],
)
def test_gate_plan_reflects_the_tasks_actually_being_run(base, sweep, group):
    """GATE: the plan is built from THIS run's task list, not from the registry at large."""
    pair = plan_fusion([base, sweep, "rnagym-dms"], registry)
    assert set(pair) == {group}, "rnagym-dms reads no shared corpus and must not join a group"
    assert pair[group].needs_all_layers is True
    assert pair[group].needs_logprob is False
    assert pair[group].n_embed_consumers == 2

    # Drop the sweep: no all-layer consumer left, so no promotion.
    alone = plan_fusion([base], registry)[group]
    assert alone.needs_all_layers is False
    assert alone.n_embed_consumers == 1


def test_gate_single_task_run_is_not_promoted_or_cached(adapter):
    """GATE: one task on its own must behave exactly as before — this is the opt-out promise.

    A lone `bacbench-essentiality` has no all-layer consumer in its group, so the wrapper must
    NOT promote to layers='all' (which would cost extra) and must NOT cache.
    """
    plans = plan_fusion(["bacbench-essentiality"], registry)
    counting = _CountingAdapter(adapter)
    reuse = ReusingAdapter(counting, plans)
    reuse.begin_task("bacbench-essentiality-windows")
    got = reuse.embed(SEQS, layers="last", pool="mean")

    assert counting.calls == [("embed", "last")]
    direct = adapter.embed(SEQS, layers="last", pool="mean")
    assert got.layers == direct.layers
    for a, b in zip(got.arrays, direct.arrays, strict=True):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# GATE 4 — the wrapper is invisible in the record
# --------------------------------------------------------------------------- #


def test_capabilities_are_passed_through(adapter):
    reuse = ReusingAdapter(adapter, _sweep_plan())
    assert Capability.EMBEDDING in reuse.capabilities()
    assert reuse.model_hash() == adapter.model_hash()


def test_gate_identity_fields_are_forwarded_not_shadowed(adapter):
    """GATE: `name`/`adapter_version`/readout must reach the result record through the wrapper.

    Regression gate. These are declared as CLASS attributes on ModelAdapter (default "" or
    None), so ordinary attribute lookup finds the default on the subclass and `__getattr__`
    never fires. That shipped: the first fused run wrote `adapter_name: ""` and
    `adapter_version: ""` into its result JSON. The model_hash was still correct (it's an
    explicit method), so nothing failed loudly — the provenance was just silently blank.
    """
    reuse = ReusingAdapter(adapter, _sweep_plan())
    assert reuse.name == adapter.name == "echo"
    assert reuse.adapter_version == adapter.adapter_version
    assert reuse.adapter_version, "adapter_version came through empty — the shadowing is back"
    assert reuse.readout is adapter.readout is not None
    assert reuse.readout_declaration() == adapter.readout_declaration()

    # ... and the record builder (the thing that actually broke) sees them.
    assert reuse.describe()["name"] == "echo"
