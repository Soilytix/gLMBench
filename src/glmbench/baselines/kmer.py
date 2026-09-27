"""The k-mer composition baseline — a model-free adapter that counts words and stops there.

`embed()` returns a sequence's normalized k-mer frequency vector: shuffle the sequence and the
vector is unchanged, so it carries composition and nothing else — no order, no structure, no
grammar, no function. Whatever a task scores against it is the part of that task that needs no
language model at all. We call that number the task's **k-mer floor**.

Composition is not junk — it is textbook biology (tetranucleotide frequency is how metagenomic
binning assigns contigs to genomes; codon-usage bias *is* 3-mer composition, and it correlates
with essentiality). The point is only that you do not need a 7-billion-parameter model to get
it, so a leaderboard cell sitting at or below this line is not evidence about a language model.

This is a real :class:`~glmbench.adapters.base.ModelAdapter`, not a duck-typed stand-in, so it
goes through the same capability negotiation as any model and the embedding tasks run over it
byte-for-byte unmodified. It declares ``EMBEDDING`` **only**: a frequency vector is not a
likelihood model, so an LM-head task (``rnagym-dms``) correctly returns a first-class ``N/A``
rather than a number that would mean nothing.

It is deliberately NOT in the adapter registry. A baseline is a floor drawn across the board,
not a competitor on it, and registering it would let it be run into ``results/`` as an ordinary
model row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter, Readout

BASES = "ACGT"

#: Non-ACGT input (N, IUPAC ambiguity codes, soft-masked lowercase) maps to -1 and the windows
#: containing it are dropped, rather than folded into some arbitrary bucket. A window that is
#: partly unknown is not evidence about any k-mer.
_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i


def kmer_frequency_vector(seq: str, k: int) -> np.ndarray:
    """Normalized frequency of every canonical k-mer in *seq*, as a ``4**k`` float64 vector.

    Overlapping windows, lexicographic k-mer order (``AAA…A`` first, ``TTT…T`` last). Windows
    containing any non-ACGT character are dropped. A sequence shorter than *k*, or one with no
    fully-canonical window at all, returns the zero vector — a sequence we cannot count is not
    silently given the composition of the empty string plus a divide-by-zero.
    """
    out = np.zeros(4**k, dtype=np.float64)
    codes = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    if len(codes) < k:
        return out
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    win = win[(win >= 0).all(axis=1)]
    if win.size == 0:
        return out
    flat = np.zeros(len(win), dtype=np.int64)
    for j in range(k):
        flat = flat * 4 + win[:, j]
    np.add.at(out, flat, 1.0)
    total = out.sum()
    return out / total if total else out


class KmerCompositionAdapter(ModelAdapter):
    """A "model" whose entire hidden state is the k-mer frequency vector.

    Args:
        k: k-mer length. ``embedding_dim`` is ``4**k``, so k=6 is 4,096 dims.
        max_context: Echoed to the tasks that window long inputs, so the baseline sees the SAME
            windowing as the model fleet (8,192 nt for LOAM/Evo). It is not a real limit here —
            counting has no context — but matching it is what keeps the floor on the same axis
            as the cells it is drawn under.
        n_layers: How many transformer blocks to pretend to have. Only the layer-sweep tasks
            look: they request ``layers="all"`` and take the max over the returned depth axis.
            Every layer gets the SAME vector, so a sweep over this adapter is flat *by
            construction* and its max-over-layers equals the single-layer score exactly. That
            identity is the reason a sweep task's floor may be read off its single-layer twin;
            ``scripts/kmer_floor.py`` verifies it rather than assuming it.
    """

    name = "kmer-composition"
    adapter_version = "1.0.0"
    CAPABILITIES = frozenset({Capability.EMBEDDING})
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "by_construction"
    readout_note = (
        "There is no network here: every layer returns the SAME composition vector, and no "
        "norm exists anywhere for the distinction to be about. Declared rather than left "
        "blank, so the baseline carries the same readout declaration as every model."
    )

    def __init__(self, k: int = 4, max_context: int | None = 8192, n_layers: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.max_context = max_context
        self.n_layers = int(n_layers)

    # --- capability ------------------------------------------------------------------------

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        if pool == "none":
            # Would require token_spans, and there is no tokenization here to span. Fail loud
            # rather than invent one: no task in the suite asks for it.
            raise ValueError(
                "KmerCompositionAdapter has no tokenization, so pool='none' (which requires "
                "per-token spans) is not meaningful. Use a reducing pool."
            )
        ids = self._layer_ids(layers)
        dim = 4**self.k
        arrays = [np.tile(kmer_frequency_vector(s, self.k), (len(ids), 1)) for s in sequences]
        return EmbeddingResult(
            arrays=arrays, layers=ids, pool=pool, token_spans=None, embedding_dim=dim
        )

    def _layer_ids(self, layers: list[int] | str) -> list[int]:
        """Resolve the tasks' ``layers`` selector against a stack that has no depth."""
        if layers == "all":
            return list(range(self.n_layers + 1))  # 0 = embedding output, then one per block
        if layers in ("last", None):
            return [self.n_layers]
        if isinstance(layers, int):
            return [layers]
        return [int(i) for i in layers]

    # --- identity --------------------------------------------------------------------------

    def model_hash(self) -> str:
        return f"kmer-composition:k{self.k}:v{self.adapter_version}"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_version": self.adapter_version,
            "capabilities": sorted(c.value for c in self.capabilities()),
            **self.readout_declaration(),
            "config": {
                "family": "kmer-composition",
                "k": self.k,
                "embedding_dim": 4**self.k,
                "max_context": self.max_context,
                "n_layers": self.n_layers,
                "params": 0,
            },
        }
