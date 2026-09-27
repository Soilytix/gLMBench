"""EchoAdapter — a model-free test double for the adapter spine.

It speaks the full wire protocol to :mod:`glmbench.runners.echo_runner` (which
computes deterministic fake outputs with no model), so the spine — capability
gating, hashing, wire round trip, scratch handling, error surfacing — is testable
in CI with no torch and no GPU.

The declared capability set is **instance-configurable** so one class can stand in
for a fully-capable model, a score-only model, or an embedding-only model in
negotiation tests. Each capability method guards on the declaration: calling one
that was not declared raises :class:`CapabilityNotImplemented` (fail-closed),
matching the contract a real adapter must honor.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from glmbench import registry

if TYPE_CHECKING:
    from glmbench.config.model_spec import ModelSpec

from .base import Capability, CapabilityNotImplemented, EmbeddingResult, ModelAdapter, Readout
from .hashing import compute_weights_digest, model_hash
from .runner_backend import LocalSubprocessRunner
from .wire import WireProtocolError

ECHO_RUNNER_MODULE = "glmbench.runners.echo_runner"

# Default: a fully capable echo model. Tests pass a subset to model other shapes.
_ALL_CAPS = frozenset(Capability)


@registry.register("adapter", "echo")
class EchoAdapter(ModelAdapter):
    name = "echo"
    adapter_version = "0.1.0"
    CAPABILITIES = _ALL_CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "by_construction"
    readout_note = (
        "Model-free test double: it synthesizes the layer axis itself, so there is no norm "
        "anywhere for the distinction to be about. Declared so that every registered "
        "adapter answers the question, with no exception to carve out."
    )

    def __init__(
        self,
        *,
        capabilities: set[Capability] | frozenset[Capability] | None = None,
        python_exe: str | None = None,
        scratch_dir: str | None = None,
        keep_scratch: bool = False,
        dry_run: bool = False,
        weights_value: str = "echo-weights-v0",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._capabilities = (
            frozenset(capabilities) if capabilities is not None else type(self).CAPABILITIES
        )
        self._config = config or {"family": "echo", "embedding_dim": 8, "n_layers": 2}
        self._weights_value = weights_value
        self._backend = LocalSubprocessRunner(
            python_exe=python_exe or sys.executable,
            runner_module=ECHO_RUNNER_MODULE,
            scratch_dir=scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> EchoAdapter:
        """Build an EchoAdapter from a ``ModelSpec``.

        Echo-specific knobs live under ``spec.model``: an optional ``capabilities``
        list (to model score-only / embedding-only shapes), a ``weights_value`` (so two
        echo "models" hash distinctly), and a ``config`` echoed into the model hash.
        """
        model = spec.model or {}
        caps = model.get("capabilities")
        capabilities = (
            frozenset(Capability(c) for c in caps) if caps is not None else None
        )
        return cls(
            capabilities=capabilities,
            python_exe=spec.runner.python_exe,
            scratch_dir=spec.runner.scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
            weights_value=model.get("weights_value", "echo-weights-v0"),
            config=model.get("config"),
        )

    # --- metadata ----------------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    def model_hash(self) -> str:
        digest = compute_weights_digest("declared", value=self._weights_value)
        return model_hash(
            weights_digest=digest,
            config=self._config,
            adapter_name=self.name,
            adapter_version=self.adapter_version,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_version": self.adapter_version,
            "capabilities": sorted(c.value for c in self._capabilities),
            **self.readout_declaration(),
            "config": self._config,
            "real_tokens": "all (echo has no special/pad tokens)",
        }

    # --- capability-gated ops (guarded on declaration) ---------------------

    def _require(self, cap: Capability) -> None:
        if cap not in self._capabilities:
            raise CapabilityNotImplemented(cap, self.name)

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        self._require(Capability.TOKENIZE)
        result = self._backend.execute("tokenize", {}, sequences, output_name="output.json")
        tokens = result.payload["json"]
        if len(tokens) != len(sequences):
            raise WireProtocolError(
                f"tokenize returned {len(tokens)} results for {len(sequences)} inputs."
            )
        return [list(map(int, t)) for t in tokens]

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        self._require(Capability.SEQUENCE_LOGLIKELIHOOD)
        result = self._backend.execute(
            "score_sequences", {"reduction": reduction}, sequences, output_name="output.npz"
        )
        scores = result.payload["scores"]
        if scores.shape[0] != len(sequences):
            raise WireProtocolError(
                f"score_sequences returned {scores.shape[0]} scores for "
                f"{len(sequences)} inputs."
            )
        if np.isnan(scores).any():
            raise WireProtocolError("score_sequences returned NaN — refusing to poison metrics.")
        return [float(x) for x in scores]

    def per_token_logprobs(self, sequences: list[str]) -> list[np.ndarray]:
        self._require(Capability.PER_TOKEN_LOGPROBS)
        result = self._backend.execute(
            "per_token_logprobs", {}, sequences, output_name="output.npz"
        )
        n = int(result.payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"per_token_logprobs returned {n} arrays for {len(sequences)} inputs."
            )
        return [result.payload[f"arr_{i}"] for i in range(n)]

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        self._require(Capability.LOGPROB_EMBEDDING)
        result = self._backend.execute(
            "logprob_embedding", {}, sequences, output_name="output.npz"
        )
        n = int(result.payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"logprob_embedding returned {n} arrays for {len(sequences)} inputs."
            )
        out: list[np.ndarray] = []
        for i, seq in enumerate(sequences):
            z = np.asarray(result.payload[f"arr_{i}"], dtype=np.float64)
            if z.shape != (len(seq),):
                raise WireProtocolError(
                    f"logprob_embedding[{i}] has length {z.shape} for a {len(seq)}-nt "
                    f"sequence; expected ({len(seq)},) (one log-prob per nucleotide)."
                )
            if not np.isfinite(z).all():
                raise WireProtocolError(
                    "logprob_embedding returned non-finite values — refusing to poison tasks."
                )
            out.append(z)
        return out

    def score_variant_llr(self, items: list[dict[str, Any]]) -> list[float]:
        self._require(Capability.MASKED_MARGINAL_LLR)
        result = self._backend.execute_variant(
            "score_variant_llr", {}, items, output_name="output.npz"
        )
        scores = result.payload["scores"]
        if scores.shape[0] != len(items):
            raise WireProtocolError(
                f"score_variant_llr returned {scores.shape[0]} scores for "
                f"{len(items)} inputs."
            )
        if not np.isfinite(scores).all():
            raise WireProtocolError(
                "score_variant_llr returned non-finite values — refusing to poison metrics."
            )
        return [float(x) for x in scores]

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        self._require(Capability.EMBEDDING)
        result = self._backend.execute(
            "embed",
            {"layers": layers, "pool": pool},
            sequences,
            output_name="output.npz",
        )
        payload = result.payload
        n_layers = int(payload["n_layers"])
        layer_ids = list(range(n_layers))
        if pool == "none":
            n = int(payload["n"])
            arrays = [payload[f"arr_{i}"] for i in range(n)]
            token_spans = [
                [(int(a), int(b)) for a, b in payload[f"span_{i}"]] for i in range(n)
            ]
            return EmbeddingResult(
                arrays=arrays,
                layers=layer_ids,
                pool=pool,
                token_spans=token_spans,
                embedding_dim=arrays[0].shape[-1] if arrays else self._config["embedding_dim"],
            )
        stacked = payload["arrays"]
        arrays = [stacked[i] for i in range(stacked.shape[0])]
        return EmbeddingResult(
            arrays=arrays,
            layers=layer_ids,
            pool=pool,
            token_spans=None,
            embedding_dim=arrays[0].shape[-1] if arrays else self._config["embedding_dim"],
        )
