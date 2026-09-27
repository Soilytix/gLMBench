"""The model-agnostic adapter spine — capabilities, the ABC, the embedding contract.

A ``ModelAdapter`` wraps one model family (LOAM, Evo2, gLM2, …) behind one
capability-gated, **batch-first** interface. The benchmark core never imports a
model: an adapter shells out to its own environment via a runner backend and
exchanges data over files. This module is the contract every adapter implements; it
is intentionally torch-free (the torch-free import gate) — only ``numpy`` (a core
dep) is referenced, for the embedding-result arrays.

Capability gating is explicit and loud: every capability method
has a default body that raises :class:`CapabilityNotImplemented`. An adapter
"has" a capability iff it overrides the method **and** lists it in the class
attribute ``CAPABILITIES``. The runner gates on the *declaration* only
(``capabilities()``), so a half-built adapter that overrode a method but forgot
to declare it is treated as *not* having the capability (fail-closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from glmbench.config.model_spec import ModelSpec


class Capability(str, Enum):  # noqa: UP042 - the (str, Enum) form is deliberate
    """The capability set an adapter can declare.

    Some may be unimplemented per model and that is fine — a task that needs a
    capability the adapter lacks returns a first-class ``N/A`` result.
    """

    TOKENIZE = "tokenize"
    SEQUENCE_LOGLIKELIHOOD = "sequence_loglikelihood"
    PER_TOKEN_LOGPROBS = "per_token_logprobs"
    EMBEDDING = "embedding"
    LOGPROB_EMBEDDING = "logprob_embedding"
    # Masked-marginal log-likelihood ratio for MLM encoders (the variant-effect scoring
    # path; see ModelAdapter.score_variant_llr).
    MASKED_MARGINAL_LLR = "masked_marginal_llr"


# Maps each capability to the adapter method that implements it. Used for the
# declared-vs-overridden consistency check and for capability gating.
CAPABILITY_METHODS: dict[Capability, str] = {
    Capability.TOKENIZE: "tokenize",
    Capability.SEQUENCE_LOGLIKELIHOOD: "score_sequences",
    Capability.PER_TOKEN_LOGPROBS: "per_token_logprobs",
    Capability.EMBEDDING: "embed",
    Capability.LOGPROB_EMBEDDING: "logprob_embedding",
    Capability.MASKED_MARGINAL_LLR: "score_variant_llr",
}


class Readout(str, Enum):  # noqa: UP042 - matches the (str, Enum) form used by Capability
    """**Which tensor** an adapter hands back when asked for a layer.

    Without a declared convention ``layers="last"`` names two different objects depending on
    the adapter, and the difference is not a rounding difference: taking the normalised
    final tensor instead of the raw one moved a DGEB score (``top_corr``) by −0.046 to
    −0.128 at the final layer, growing with depth. So the kind of tensor is declared, per
    adapter, in machine-readable form — an adapter whose architecture cannot supply the
    target convention says what it *does* supply rather than lying by omission.

    The four values are three different objects and one qualifier:

    * :attr:`BLOCK_OUTPUT` — the residual stream (trunk) after block *i*'s adds, **raw**.
      The target convention (see :class:`EmbeddingResult`).
    * :attr:`NORMED_BLOCK_OUTPUT` — that same trunk with a norm applied *per tap*
      (e.g. Evo 2's ``blocks.N.pre_norm``, which captures ``norm(x_{N-1})``). A residual
      read that is normalised where the target convention's is not: still a comparability
      gap, and naming it is the point.
    * :attr:`POST_FINAL_NORM` — the trunk with the model's *closing* norm applied, i.e.
      a pre-LN decoder's ``hidden_states[-1]``.
    * :attr:`BRANCH_TAP` — an attention/MLP branch output *before* the residual add
      (Evo 2 Arc's documented ``blocks.N.mlp.l3``). A **proposed edit**, not a state; a row
      carrying one is a labelled diagnostic and never a comparison row.
    """

    BLOCK_OUTPUT = "block_output"
    NORMED_BLOCK_OUTPUT = "normed_block_output"
    POST_FINAL_NORM = "post_final_norm"
    BRANCH_TAP = "branch_tap"


#: Readout kinds that may sit in a comparison column. ``BRANCH_TAP`` is excluded because it is
#: not a state of the network at all; the other three are states, and one column should
#: still not *mix* them.
COMPARABLE_READOUTS: frozenset[Readout] = frozenset(
    {Readout.BLOCK_OUTPUT, Readout.NORMED_BLOCK_OUTPUT, Readout.POST_FINAL_NORM}
)


class CapabilityNotImplemented(NotImplementedError):
    """Raised by a default capability method an adapter did not implement.

    Tasks negotiate capabilities *before* calling, so in normal operation
    this is never hit — it is the loud fail-closed backstop when something calls a
    capability the adapter did not declare.
    """

    def __init__(self, capability: Capability, adapter_name: str) -> None:
        self.capability = capability
        self.adapter_name = adapter_name
        super().__init__(
            f"Adapter '{adapter_name}' does not implement capability "
            f"'{capability.value}'. Declared capabilities only are honored; this "
            f"call should have been gated by the task's capability negotiation."
        )


@dataclass
class EmbeddingResult:
    """The explicit, model-agnostic embedding representation contract.

    Because tokenization granularity differs across models (LOAM and Evo2 single-nt =
    1 bp/token, ProkBERT overlapping 6-mers, GenomeOcean BPE = variable bp/token …),
    the contract makes the
    representation unambiguous so a downstream task can compare embeddings
    model-agnostically.

    Attributes:
        arrays: One array per input sequence. Shape depends on ``pool``:
            ``[len(layers), embedding_dim]`` for a reducing pool
            (``mean``/``max``/``bos``/``eos``/``last``), or
            ``[n_tokens, len(layers), embedding_dim]`` for ``pool == "none"``.
        layers: Which layers each array's layer axis corresponds to.

            **The layer convention (normative).** Layer ``i`` is the
            **output of transformer block i** — the residual stream after block ``i``'s
            adds — **pre any model-level final normalisation**. Layer 0 is the
            embedding-layer output (post token+positional embedding, pre block 0). So for
            an ``L``-block model the ids run ``0..L`` and ``layers="last"`` is ``L``: the
            raw output of the last block, *not* ``norm(last_block_output)``.

            An adapter whose architecture cannot supply that tensor does not silently
            substitute another one — it declares what it actually returns via
            :attr:`ModelAdapter.readout` (:class:`Readout`), which is echoed into
            ``describe()`` and into the stored result record. Every adapter in this module
            references this paragraph rather than restating it.
        pool: The pooling that was applied (echoed back, never inferred).
        token_spans: Per sequence, per token, a half-open ``(nt_start, nt_end)``
            span into the **original** nucleotide string. **Required** when
            ``pool == "none"`` so tasks can map tokens → nucleotides regardless of
            tokenizer granularity; ``None`` otherwise.
        embedding_dim: Width of the model's hidden representation.
    """

    arrays: list[np.ndarray]
    layers: list[int]
    pool: str
    token_spans: list[list[tuple[int, int]]] | None
    embedding_dim: int

    def __post_init__(self) -> None:
        if self.pool == "none" and self.token_spans is None:
            raise ValueError(
                "EmbeddingResult with pool='none' requires token_spans so token "
                "positions can be aligned to nucleotides (see EmbeddingResult)."
            )
        if self.token_spans is not None and len(self.token_spans) != len(self.arrays):
            raise ValueError(
                f"token_spans length ({len(self.token_spans)}) must match number of "
                f"sequences ({len(self.arrays)})."
            )


class ModelAdapter(ABC):
    """Base class for every model adapter.

    Subclasses declare the capabilities they provide in :attr:`CAPABILITIES`,
    override the matching capability methods, and implement :meth:`model_hash` and
    :meth:`describe`. All capability methods are **batch-first**: they take a list
    of sequences and return a list.
    """

    name: str = ""
    adapter_version: str = ""
    CAPABILITIES: frozenset[Capability] = frozenset()

    #: Which tensor this adapter's :meth:`embed` returns per layer. ``None`` means
    #: **undeclared** — every adapter owes an answer to the question. It participates in the
    #: model hash only via :attr:`adapter_version`: changing what ``embed`` returns is a
    #: semantics change, so the adapter version is bumped, the hash moves, and a NEW row is
    #: written. Do **not** add ``readout`` as a fresh hash input — that would drift every
    #: existing row for a field their runs already satisfy.
    readout: Readout | None = None

    #: Set ``True`` by an adapter whose readout is a property of the **spec** rather than of
    #: the code, so the class attribute is legitimately ``None`` and the value is assigned in
    #: ``__init__``. Only Evo 2 Arc needs it: its ``layer_names`` can name a branch, a
    #: normalised trunk or the trunk itself. The flag is not an answer on its own: a default
    #: instance must still declare a readout, so the escape hatch cannot become a way to skip
    #: the question.
    readout_is_instance_level: bool = False

    #: Free-text qualifier on the readout, or ``None``. Used where the declared kind is
    #: honest but incomplete on its own — e.g. Evo 2's residual tap is a *normalised* trunk
    #: read where LOAM's is raw. Rendered on the row; not hashed.
    readout_note: str | None = None

    #: **How we know** the :attr:`readout` declaration is true. "We didn't change it" is not
    #: evidence: an adapter that needed no code change still owes a measurement or an
    #: inspection. Three states, kept distinct as *measured / by inspection / never checked*:
    #:
    #: * ``"probed"`` — measured on this architecture: a forward hook on the last block
    #:   compared with the tensor the model returns (the verdict is in ``readout_note``);
    #: * ``"by_construction"`` — the runner *builds* the tensor itself (an explicit forward
    #:   hook, a named tap), so the claim is true by inspection of code we own, and the
    #:   inspection is cited in ``readout_note``;
    #: * ``"unverified"`` — declared and not yet established. It exists so that "nobody
    #:   checked" is a *visible* state rather than an absent one.
    readout_evidence: str = "unverified"

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> ModelAdapter:
        """Build an adapter instance from a :class:`ModelSpec`.

        The single entry point the runner uses to turn a spec into a live adapter,
        keeping all adapter-specific construction inside the adapter (one runner, all
        models). Subclasses must override this; the base raises so a
        not-yet-wired adapter fails loudly rather than silently misbuilding.
        """
        raise NotImplementedError(
            f"Adapter '{cls.__name__}' does not implement from_spec(); it cannot be "
            f"built from a ModelSpec yet."
        )

    def capabilities(self) -> frozenset[Capability]:
        """Return the capabilities this adapter declares.

        The runner gates on this declaration only — it does **not** introspect
        method overrides. Subclasses whose capability set is instance-configurable
        (e.g. test doubles) override this; by default it echoes the class
        attribute.
        """
        return type(self).CAPABILITIES

    @abstractmethod
    def model_hash(self) -> str:
        """Return the stable signature. Must not require running the model."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a human-readable echo of the adapter/model config + capabilities."""

    # --- capability-gated, batch-first; default bodies raise (fail-closed) ----

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        raise CapabilityNotImplemented(Capability.TOKENIZE, self.name)

    def score_sequences(
        self, sequences: list[str], *, reduction: str = "mean"
    ) -> list[float]:
        raise CapabilityNotImplemented(Capability.SEQUENCE_LOGLIKELIHOOD, self.name)

    def per_token_logprobs(self, sequences: list[str]) -> list[np.ndarray]:
        raise CapabilityNotImplemented(Capability.PER_TOKEN_LOGPROBS, self.name)

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        """Per-**nucleotide** conditional log-probability vector (the surprisal embedding).

        The reduction-free form of :meth:`score_sequences`: for a sequence of ``N``
        nucleotides, returns one 1-D ``float64`` array of length ``N`` giving
        ``log P(nᵢ | n₁…nᵢ₋₁)`` per realized base, in input order, 1:1 with inputs.
        Built from the **same** prefix/BOS/masking as scoring, so per sequence
        ``mean(z) == score_sequences(reduction="mean")`` to fp tolerance.
        Default body raises (fail-closed) — adapters that compute it override + declare it.
        """
        raise CapabilityNotImplemented(Capability.LOGPROB_EMBEDDING, self.name)

    def score_variant_llr(self, items: list[dict[str, Any]]) -> list[float]:
        """Masked-marginal LLR per variant (batch-first, 1:1 with *items*).

        The variant-effect scoring path for **masked-LM (MLM) encoders** (gLM2 and any
        future bidirectional model), which have no valid causal likelihood factorization
        and therefore cannot serve :meth:`score_sequences`. For a variant with mutated
        positions ``M`` against reference (WT) sequence ``r``::

            LLR(variant) = Σ_{i ∈ M} [ log p(x_i = mut_i | r_{\\i})
                                       − log p(x_i = wt_i  | r_{\\i}) ]

        where ``r_{\\i}`` is the reference with position ``i`` replaced by the model's
        ``[MASK]`` token, scored in a single forward pass. Because both terms share the
        same masked context, the softmax normalizer cancels and the contribution reduces
        to the raw logit difference ``logit_mut_i − logit_wt_i`` (vocab-restriction
        invariant). Multi-site variants use the additive single-site masked-marginal
        approximation (each position masked independently on the WT background; no joint
        masking). Higher LLR ⇒ model prefers the mutant (same sign convention as
        :meth:`score_sequences`).

        Args:
            items: One dict per variant, each::

                {"reference": <WT DNA str>,
                 "mutations": [[pos0, "WT_base", "MUT_base"], ...]}  # 0-based positions

        Returns:
            ``Σ_i (logit_mut_i − logit_wt_i)`` per variant, on the ``[MASK]``-at-i
            reference context, in input order. Default body raises (fail-closed) —
            adapters that compute it override + declare ``MASKED_MARGINAL_LLR``.
        """
        raise CapabilityNotImplemented(Capability.MASKED_MARGINAL_LLR, self.name)

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        """Hidden representations per sequence, per requested layer.

        ``layers`` follows the normative layer convention documented on
        :class:`EmbeddingResult` — layer ``i`` is the **raw output of block i**, pre any
        model-level final norm, and ``"last"`` is the final block's raw output. The kind of
        tensor this adapter actually returns is declared by :attr:`readout`.

        Two properties every implementation owes its callers:

        * ``embed(layers="last")`` is **exactly** ``embed(layers="all")`` sliced at the last
          layer id. :mod:`glmbench.adapters.reuse` serves a single-layer task from
          the sweep's cached forward on that basis, so a violation makes a fused run disagree
          with an unfused one.
        * ``layers`` ids are echoed back in :attr:`EmbeddingResult.layers`, never inferred.
        """
        raise CapabilityNotImplemented(Capability.EMBEDDING, self.name)

    # --- sweepability ---------------------------------------------------------

    #: Can this adapter reach a non-final layer at all? ``False`` means "not sweepable" is a
    #: property of the serving path, not an omission — a reader must be able to tell
    #: *cannot* from *did not*, because "not swept" and "swept and it was flat" are opposite
    #: claims.
    sweepable: bool = True

    #: Why not, when :attr:`sweepable` is ``False``. Rendered on the row.
    not_sweepable_reason: str | None = None

    @classmethod
    def expected_sweep_taps(cls, describe: dict[str, Any]) -> int | None:
        """How many taps a **full** sweep of this model yields, or ``None`` if not knowable.

        The denominator of "did we sweep everything?". It is a classmethod over a stored
        ``describe()`` dict rather than an instance property because a coverage check reads
        *records*, long after the adapter that wrote them is gone.

        The default is ``n_layers + 1`` — the layer convention, embedding output plus one tap
        per block. **Three adapters legitimately differ and each overrides**, which is exactly
        why this is a method and not a constant: gLM2's tuple omits the embedding output
        entirely (so full is ``n_layers``), NTv3's depth axis is a U-Net
        (``2·num_downsamples + num_layers`` mixed-resolution states), and Evo 2 Arc's is
        whatever its spec's ``layer_names`` lists. Hard-coding ``n_layers + 1`` would mark a
        complete gLM2 sweep as one tap short, forever.

        ``None`` when the depth is not recorded — the caller must then infer the denominator
        from what was observed and say so.
        """
        n = describe.get("n_layers")
        return int(n) + 1 if isinstance(n, int) and n > 0 else None

    # --- readout declaration --------------------------------------------------

    def readout_declaration(self) -> dict[str, Any]:
        """The machine-readable readout block, for ``describe()`` and the result record.

        One helper rather than twelve copies of the same three keys, so a new adapter
        declares the *value* and inherits the shape.
        """
        r = type(self).readout if self.readout is None else self.readout
        return {
            "readout": r.value if r is not None else None,
            "readout_note": self.readout_note,
            "readout_evidence": self.readout_evidence,
            # Recorded so a stored record can be asked "did the semantics change without a
            # version bump?" without re-deriving the pairing by hand.
            "readout_adapter_version": self.adapter_version,
        }


def missing_capability_methods(adapter_cls: type[ModelAdapter]) -> list[Capability]:
    """Return declared capabilities whose method is **not** overridden on *adapter_cls*.

    Used by the consistency test: for every adapter, every declared
    capability's method must actually be overridden. An empty list means the
    adapter is consistent; a non-empty list names the declared-but-unimplemented
    capabilities (the half-built mistake we catch loudly).
    """
    missing: list[Capability] = []
    for cap in adapter_cls.CAPABILITIES:
        method = CAPABILITY_METHODS[cap]
        if getattr(adapter_cls, method) is getattr(ModelAdapter, method):
            missing.append(cap)
    return sorted(missing)
