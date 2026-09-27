"""Evo2ArcAdapter — Evo2 through the official ArcInstitute ``evo2`` pip package.

The **one-env, one-forward** Evo2 route: it calls ``Evo2('evo2_7b_base')`` in a plain conda
env and serves **every** capability from the same forward pass.

**The layer tap.** The package runs the **whole** stack and taps named activations on the
way past::

    outputs, embeddings = model(input_ids, return_embeddings=True,
                                layer_names=['blocks.28'])

so ``outputs[0]`` is the full-depth logits **and** ``embeddings['blocks.28']`` is block 28's
output, from one pass. Consequences worth internalizing:

- The layer name affects **embeddings only**. Scoring (``rnagym-dms``) and per-nucleotide
  log-probs always read the final layer + LM head and are untouched by ``layer_names``.
- Arbitrary, intermediate and *multiple* layers come back in ONE pass, so the layer-sweep
  tasks work from a single forward.

**Layer naming vs. the integer layer contract.** :class:`~glmbench.adapters.base.EmbeddingResult`
keys layers by *int* ("0 = embedding output, i = block i"), and Arc's layer names
(``blocks.28``, or ``blocks.28.mlp.l3``, a submodule *inside* block 28) are module paths, not
integers in that space. Rather than widen the contract across every adapter and task, this
adapter exposes the **position of each name in ``model.layer_names``** as the integer index:
``layer_names=['blocks.20', 'blocks.28']`` ⇒ layer ids ``[0, 1]``. The name↔index map is
recorded in :meth:`describe` (and therefore in the result JSON), and ``layers='last'``
resolves to the **last configured name** — which is what a single-layer task reads, with no
task-side change.

**``layer_names`` is part of the model hash** (it is in ``_arch``). "Evo2 read at block 28"
is a genuinely different feature extractor from "Evo2 read at the final block", so the two
get distinct leaderboard rows instead of silently overwriting each other (the leaderboard is
idempotent per ``model_hash`` × task).

**Scoring convention.** The package has no ``score_sequences()``; the runner computes the
reduction itself: the mean per-token causal log-prob, with one prefix token prepended as
context and never scored. ``score_prefix`` is a knob (``bos``|``eos``|``none``), not a
hardcode. The package's ``prepend_bos=True`` actually prepends ``tokenizer.eod``, and for
Evo2's ``CharLevelTokenizer`` ``eod == eos == 0``, so ``bos`` and ``eos`` select the same
token and only ``none`` is a different setting. With ``bos`` this route matches an
independent implementation of the same convention on ``rnagym-dms`` to 0.0006 in macro
abs-Spearman.

Capabilities: ``{SEQUENCE_LOGLIKELIHOOD, EMBEDDING, LOGPROB_EMBEDDING}``. This module is
**core (torch-free)** — it assembles wire requests and validates responses; all torch/Evo2
work lives in :mod:`glmbench.runners.evo2_arc_runner`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from glmbench import registry

if TYPE_CHECKING:
    from glmbench.config.model_spec import ModelSpec

from .base import Capability, EmbeddingResult, ModelAdapter, Readout
from .hashing import compute_weights_digest, model_hash
from .runner_backend import LocalSubprocessRunner, RunnerBackend
from .wire import WireProtocolError

logger = logging.getLogger(__name__)

EVO2_ARC_RUNNER_MODULE = "glmbench.runners.evo2_arc_runner"

# Arc's own recommended embedding layer, per model. These strings are ARCH-SPECIFIC: the
# 7B stack has 32 blocks and Arc points at block 28; BacBench's Evo2Embedder default
# (blocks.24.mlp.l3) is for the 1B model. A wrong name does not fall back — the runner
# raises on a missing key rather than silently returning a different tensor.
#   https://github.com/arcinstitute/evo2#embeddings
MODEL2LAYER = {
    "evo2_7b": "blocks.28.mlp.l3",
    "evo2_7b_base": "blocks.28.mlp.l3",
    "evo2_1b_base": "blocks.24.mlp.l3",
    "evo2_40b": "blocks.28.mlp.l3",
    "evo2_40b_base": "blocks.28.mlp.l3",
}

# Published block counts / widths, used only for the hash arch + describe(). The runner
# reports the real embedding_dim from the tensor it actually produced; these never override it.
MODEL2ARCH = {
    "evo2_7b": {"n_layers": 32, "hidden_size": 4096},
    "evo2_7b_base": {"n_layers": 32, "hidden_size": 4096},
    "evo2_1b_base": {"n_layers": 25, "hidden_size": 1920},
    "evo2_40b": {"n_layers": 50, "hidden_size": 8192},
    "evo2_40b_base": {"n_layers": 50, "hidden_size": 8192},
}

_CAPS = frozenset(
    {
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.EMBEDDING,
        Capability.LOGPROB_EMBEDDING,
    }
)


def classify_taps(layer_names: list[str]) -> tuple[Readout, str]:
    """Map a list of Vortex module paths to the :class:`Readout` they actually produce.

    Evo 2 is the one model on the board whose readout is set by the **spec**, not the code,
    because ``layer_names`` names arbitrary submodules and Vortex hooks whatever it is given:

    * ``blocks.N``            → the block module's own output: the trunk after the residual
      add. ``block_output``, the target convention.
    * ``blocks.N.pre_norm``   → a hook on ``pre_norm`` captures that module's OUTPUT, i.e.
      the **normalised** trunk entering block N. ``normed_block_output`` — a residual read,
      but normalised where LOAM's is raw, and that gap is named rather than buried. It
      matters here more than almost anywhere: Evo 2's stream reaches RMS 89.3 at block 28,
      and dividing by a per-token RMS that a few huge dimensions own shrinks every other
      dimension — the reason LOAM is read raw.
    * ``blocks.N.mlp.l3`` / ``…attn…`` → a branch output BEFORE the residual add: Arc's own
      documented recommendation, and a **proposed edit** rather than a state of the network.
      ``branch_tap`` — kept as a labelled diagnostic row, never a comparison row.

    A mixed list is a hard error: one row cannot report two readouts, and picking the
    "dominant" one would put a branch tap into a comparison column under a residual label.
    """
    kinds: dict[Readout, list[str]] = {}
    for name in layer_names:
        tail = name.split(".")
        if len(tail) == 2 and tail[0] == "blocks":
            kind = Readout.BLOCK_OUTPUT
        elif name.endswith(".pre_norm") or name.endswith(".post_norm"):
            kind = Readout.NORMED_BLOCK_OUTPUT
        else:
            kind = Readout.BRANCH_TAP
        kinds.setdefault(kind, []).append(name)

    if len(kinds) > 1:
        raise ValueError(
            "Evo2ArcAdapter: model.layer_names mixes tap kinds "
            + "; ".join(f"{k.value}={v}" for k, v in kinds.items())
            + ". A row reports ONE readout, and averaging over kinds would put a branch tap "
            "into a comparison column under a residual-stream label. Split them into "
            "separate specs (the branch one becomes a diagnostic row)."
        )
    kind = next(iter(kinds))
    notes = {
        Readout.BLOCK_OUTPUT: (
            "taps the block modules themselves, so the captured tensor is the trunk after "
            "the residual add — like-for-like with LOAM."
        ),
        Readout.NORMED_BLOCK_OUTPUT: (
            "taps `pre_norm`, whose hook captures that module's OUTPUT: the NORMALISED trunk "
            "entering the block, not the raw stream. A residual read that carries a "
            "normalisation LOAM's does not — a real comparability gap, recorded on the row."
        ),
        Readout.BRANCH_TAP: (
            "taps an MLP/attention branch BEFORE the residual add (Arc's documented default, "
            "https://github.com/arcinstitute/evo2#embeddings). Not a state of the network — "
            "a proposed edit to it. Diagnostic row only; excluded from ranking."
        ),
    }
    return kind, f"layer_names={layer_names}: {notes[kind]}"


@registry.register("adapter", "evo2-arc")
class Evo2ArcAdapter(ModelAdapter):
    """Adapter for Evo2 served by the ArcInstitute ``evo2`` package (named-layer taps)."""

    name = "evo2-arc"
    adapter_version = "0.1.0"
    CAPABILITIES = _CAPS
    # Set per INSTANCE in __init__ from the configured tap names — unlike every other adapter,
    # this one's readout is a property of the spec, not of the code. `blocks.28.mlp.l3` is a
    # branch tap; `blocks.28.pre_norm` is a normalised trunk read; `blocks.28` is the trunk.
    # Declaring a class-level default would let a residual spec inherit "branch_tap" (or vice
    # versa) and mislabel the row.
    readout: Readout | None = None
    readout_is_instance_level = True
    readout_evidence = "by_construction"

    @classmethod
    def expected_sweep_taps(cls, describe: dict[str, Any]) -> int | None:
        """Evo 2 Arc's depth axis is whatever its spec's ``layer_names`` lists.

        There is no architectural "all layers" here — Vortex hooks the named modules and
        nothing else — so a full sweep is as full as the spec made it. Returning
        ``len(layer_names)`` means a one-name spec reports ``expected == observed == 1``,
        which is honest but useless on its own: a sweep of one tap must not be PRESENTED as
        a sweep.
        """
        names = describe.get("layer_names")
        return len(names) if isinstance(names, list) and names else None

    def __init__(
        self,
        *,
        model_name: str = "evo2_7b_base",
        layer_names: list[str] | None = None,
        dtype: str = "bfloat16",
        score_prefix: str = "bos",
        max_context: int = 8192,
        batch_size: int = 1,
        token_budget: int | None = None,
        device: str = "auto",
        backend: str = "local",
        python_exe: str = "python",
        n_layers: int | None = None,
        hidden_size: int | None = None,
        hf_home: str | None = None,
        weights_digest_strategy: str = "hf_revision",
        weights_digest_value: str | None = None,
        scratch_dir: str | None = None,
        keep_scratch: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.model_name = model_name
        # Default to Arc's recommended tap for this model rather than the final block:
        # the final layer of a causal LM has specialized for next-token prediction, which
        # is exactly the readout Arc steers users away from for downstream probes.
        if layer_names is None:
            default = MODEL2LAYER.get(model_name)
            if default is None:
                raise ValueError(
                    f"Evo2ArcAdapter: no default embedding layer known for model "
                    f"{model_name!r} (known: {sorted(MODEL2LAYER)}). Layer names are "
                    f"arch-specific — set model.layer_names explicitly in the spec."
                )
            layer_names = [default]
        if not layer_names or not all(isinstance(x, str) and x for x in layer_names):
            raise ValueError(
                f"Evo2ArcAdapter: model.layer_names must be a non-empty list of non-empty "
                f"strings (e.g. ['blocks.28.mlp.l3']); got {layer_names!r}."
            )
        if len(set(layer_names)) != len(layer_names):
            raise ValueError(
                f"Evo2ArcAdapter: model.layer_names contains duplicates ({layer_names!r}); "
                f"each name maps to one integer layer id, so duplicates are ambiguous."
            )
        self.layer_names = list(layer_names)
        self.readout, self.readout_note = classify_taps(self.layer_names)
        self.dtype = dtype
        self.score_prefix = score_prefix
        self.max_context = int(max_context)
        self.batch_size = int(batch_size)
        self.token_budget = int(token_budget) if token_budget else None
        self.device = device
        self.backend = backend
        self.python_exe = python_exe
        self.hf_home = hf_home
        self._weights_digest_strategy = weights_digest_strategy
        self._weights_digest_value = weights_digest_value

        arch_defaults = MODEL2ARCH.get(model_name, {})
        self.n_layers = int(
            n_layers if n_layers is not None else arch_defaults.get("n_layers", 0)
        )
        self.embedding_dim = int(
            hidden_size if hidden_size is not None else arch_defaults.get("hidden_size", 0)
        )

        # The hashed arch. layer_names IS in here on purpose: a different tap is a
        # different feature extractor and must key its own leaderboard row (module docstring).
        self._arch = {
            "model": {
                "family": "evo2-arc",
                "model_name": self.model_name,
                "layer_names": self.layer_names,
                "n_layers": self.n_layers,
                "hidden_size": self.embedding_dim,
                "max_context": self.max_context,
                "dtype": self.dtype,
                "score_prefix": self.score_prefix,
            }
        }

        if backend != "local":
            raise ValueError(
                f"Evo2ArcAdapter backend must be 'local' (the package runs in its own conda "
                f"env via LocalSubprocessRunner — that is the whole point of this route); "
                f"got {backend!r}."
            )
        self._backend: RunnerBackend = LocalSubprocessRunner(
            python_exe=python_exe,
            runner_module=EVO2_ARC_RUNNER_MODULE,
            scratch_dir=scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> Evo2ArcAdapter:
        """Build an Evo2ArcAdapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``model_name`` (the Arc model id the package downloads),
        optional ``layer_names`` (defaults to :data:`MODEL2LAYER` for the model),
        ``dtype``, ``score_prefix``, ``max_context``, ``n_layers``/``hidden_size`` (hash
        arch only). ``spec.runner`` selects ``python_exe`` (the evo2 env) and ``gpus``.
        """
        model = spec.model or {}
        extra = spec.runner.extra
        layer_names = model.get("layer_names")
        if isinstance(layer_names, str):  # a single name is a common, harmless spec typo
            layer_names = [layer_names]
        return cls(
            model_name=str(model.get("model_name", "evo2_7b_base")),
            layer_names=list(layer_names) if layer_names else None,
            dtype=str(model.get("dtype", "bfloat16")),
            score_prefix=str(model.get("score_prefix", "bos")),
            max_context=int(model.get("max_context", 8192)),
            batch_size=int(extra.get("batch_size", model.get("batch_size", 1))),
            token_budget=(
                int(tb)
                if (tb := extra.get("token_budget", model.get("token_budget")))
                else None
            ),
            device=_device_from_gpus(spec.runner.gpus),
            backend=spec.runner.backend,
            python_exe=spec.runner.python_exe or "python",
            n_layers=model.get("n_layers"),
            hidden_size=model.get("hidden_size"),
            hf_home=extra.get("hf_home", model.get("hf_home")),
            weights_digest_strategy=spec.weights_digest.strategy,
            weights_digest_value=spec.weights_digest.value,
            scratch_dir=spec.runner.scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    # --- metadata ----------------------------------------------------------

    def model_hash(self) -> str:
        digest = compute_weights_digest(
            self._weights_digest_strategy,
            path=None,
            value=self._weights_digest_value,
        )
        return model_hash(
            weights_digest=digest,
            config=self._arch,
            adapter_name=self.name,
            adapter_version=self.adapter_version,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_version": self.adapter_version,
            "capabilities": sorted(c.value for c in self.CAPABILITIES),
            **self.readout_declaration(),
            "model_name": self.model_name,
            "layer_names": list(self.layer_names),
            # The integer layer id ⇄ Arc name map. This is the record of WHICH tensor each
            # `layer_N_*` leaderboard column actually came from — without it the columns
            # are unattributable.
            "layer_id_to_name": {i: n for i, n in enumerate(self.layer_names)},
            "embedding_layer": self.layer_names[-1],  # what layers='last' resolves to
            "backend": self.backend,
            "python_exe": self.python_exe,
            "dtype": self.dtype,
            "score_prefix": self.score_prefix,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "max_context": self.max_context,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "arch": self._arch["model"],
            "real_tokens": (
                "scoring: full-depth logits + LM head (NOT the embedding tap — layer_names "
                "does not affect scoring). score_prefix prepended as causal context and "
                "masked from the loss; mean|sum per-token causal log-prob over real tokens."
            ),
            "embedding_real_tokens": (
                "embedding: mean/max over valid (non-pad) token positions of the named-layer "
                "activation captured via return_embeddings=True (no truncation — the full "
                "stack runs). Layer ids are POSITIONS IN layer_names, see layer_id_to_name; "
                "layers='last' = the last configured name."
            ),
            "logprob_embedding_real_tokens": (
                "logprob_embedding: the reduction-free score path — per-nucleotide "
                "log P(nᵢ|n₁…nᵢ₋₁) with the same score_prefix context. Evo2's tokenizer is "
                "byte-level single-nt, so 1 token ⇄ 1 nt and mean(z) == score(mean)."
            ),
        }

    # --- request plumbing --------------------------------------------------

    def _common_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            # The weights the model hash names: with an `hf_revision` digest the runner loads
            # exactly this commit, not whatever the Hub repo's `main` is today.
            "revision": (
                self._weights_digest_value
                if self._weights_digest_strategy == "hf_revision"
                else None
            ),
            "dtype": self.dtype,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "max_context": self.max_context,
            "hf_home": self.hf_home,
        }

    def _resolve_layers(self, layers: list[int] | str) -> list[int]:
        """Resolve ``'last'``/``'all'``/int/list to indices into :attr:`layer_names`.

        Unlike the block-index adapters, these ints index the **configured name list**, so
        ``'all'`` means "every layer this spec was told to tap" — bounded by the spec, not
        by the model's depth. That keeps a sweep honest: you cannot ask for a tensor the
        spec did not name (and therefore did not record in the model hash).
        """
        n = len(self.layer_names)
        if layers == "last":
            return [n - 1]
        if layers == "all":
            return list(range(n))
        if isinstance(layers, int):
            return [range(n)[layers]]  # negatives index from the end; IndexError if OOB
        if isinstance(layers, (list, tuple)) and layers and all(isinstance(x, int) for x in layers):
            return [range(n)[int(x)] for x in layers]
        raise ValueError(
            f"Evo2ArcAdapter.embed layers must be 'last', 'all', an int, or a non-empty "
            f"list of ints indexing model.layer_names ({self.layer_names!r}); got {layers!r}."
        )

    # --- capability-gated ops ----------------------------------------------

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        """Per-sequence causal log-likelihood from the full-depth logits.

        ``reduction='mean'`` is the convention described in the module docstring; ``'sum'``
        is also supported.
        """
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"Evo2ArcAdapter.score_sequences supports reduction 'mean'|'sum', got "
                f"{reduction!r}."
            )
        params = self._common_params()
        params["reduction"] = reduction
        params["score_prefix"] = self.score_prefix
        result = self._backend.execute(
            "score_sequences", params, sequences, output_name="output.npz"
        )
        scores = result.payload["scores"]
        if scores.shape[0] != len(sequences):
            raise WireProtocolError(
                f"score_sequences returned {scores.shape[0]} scores for {len(sequences)} inputs."
            )
        if np.isnan(scores).any():
            raise WireProtocolError("score_sequences returned NaN — refusing to poison metrics.")
        return [float(x) for x in scores]

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        """Per-nucleotide conditional log-prob vector ``z`` per sequence.

        The reduction-free body of :meth:`score_sequences` — same ``score_prefix`` context,
        same gather, no collapse. Evo2's tokenizer is byte-level (1 char/token), so ``z``
        has one value per nucleotide and ``mean(z) == score_sequences(mean)``.
        """
        params = self._common_params()
        params["score_prefix"] = self.score_prefix
        result = self._backend.execute(
            "logprob_embedding", params, sequences, output_name="output.npz"
        )
        payload = result.payload
        n = int(payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"logprob_embedding returned {n} arrays for {len(sequences)} inputs."
            )
        out: list[np.ndarray] = []
        for i, seq in enumerate(sequences):
            z = np.asarray(payload[f"arr_{i}"], dtype=np.float64)
            expected = min(len(seq), self.max_context)
            if z.shape != (expected,):
                raise WireProtocolError(
                    f"logprob_embedding[{i}] length {z.shape} != ({expected},) — expected one "
                    f"log-prob per nucleotide (Evo2 is 1 char/token), truncated at "
                    f"max_context={self.max_context}."
                )
            if not np.isfinite(z).all():
                raise WireProtocolError(
                    "logprob_embedding returned non-finite values — refusing to poison tasks."
                )
            out.append(z)
        return out

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        """Named-layer embeddings — every requested layer from ONE forward pass.

        ``layers`` indexes :attr:`layer_names` (see :meth:`_resolve_layers`), so
        ``'last'`` is the last configured name — ``blocks.31`` in the shipped 7B spec —
        and ``'all'`` taps every configured name in a single pass. Pooling is
        ``mean``/``max`` over the valid (non-pad) token positions.
        """
        if pool not in ("mean", "max"):
            raise ValueError(
                f"Evo2ArcAdapter.embed supports pool 'mean' or 'max' (pooled over valid "
                f"tokens); got {pool!r}. (bos/eos/last/none are not implemented — the "
                f"per-token path would need token_spans, which no task asks of Evo2.)"
            )
        resolved = self._resolve_layers(layers)
        params = self._common_params()
        params["layers"] = resolved
        params["layer_names"] = [self.layer_names[i] for i in resolved]
        params["pool"] = pool
        result = self._backend.execute("embed", params, sequences, output_name="output.npz")
        payload = result.payload
        layer_ids = [int(x) for x in payload["layer_ids"]]
        if layer_ids != resolved:
            raise WireProtocolError(
                f"embed returned layer ids {layer_ids} for request {resolved} — the runner "
                f"must echo the requested layers in order (a silent reorder would mislabel "
                f"every layer_N column on the leaderboard)."
            )
        emb_dim = int(payload["embedding_dim"])
        stacked = payload["arrays"]  # [n, len(layers), embedding_dim]
        if stacked.shape[0] != len(sequences):
            raise WireProtocolError(
                f"embed returned {stacked.shape[0]} rows for {len(sequences)} inputs."
            )
        if np.isnan(stacked).any():
            raise WireProtocolError("embed returned NaN — refusing to poison downstream tasks.")
        arrays = [stacked[i] for i in range(stacked.shape[0])]
        return EmbeddingResult(
            arrays=arrays,
            layers=layer_ids,
            pool=pool,
            token_spans=None,
            embedding_dim=emb_dim,
        )


def _device_from_gpus(gpus: str | None) -> str:
    """Map ``runner.gpus`` to a device string the runner resolves (``auto`` → cuda)."""
    if gpus is None or str(gpus).strip() in ("", "-1", "none", "cpu"):
        return "cpu"
    return "auto"


__all__ = ["Evo2ArcAdapter", "EVO2_ARC_RUNNER_MODULE", "MODEL2LAYER", "MODEL2ARCH"]
