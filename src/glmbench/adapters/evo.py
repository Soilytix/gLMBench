"""EvoAdapter — the plain torch+transformers path for Evo 1.5 (StripedHyena, byte-level).

Wraps Arc/TogetherAI's **Evo 1.5** (`evo-design/evo-1.5-8k-base`, 7B) behind the
model-agnostic adapter contract. The heavy lifting (load the HF model, run forwards) happens
in :mod:`glmbench.runners.evo_runner`, which runs in an env with ``torch`` + ``transformers``
+ **``flash_attn``** (the ``glmbench`` env, with flash-attn installed — see below). This module is
**core (torch-free)** — it only assembles wire requests, reads the HF ``config.json`` for the
model-hash arch (plain JSON, no torch, when the checkpoint is a local dir), builds the runner
backend, and validates responses.

This is a **stock-HF-style** adapter (shaped after :class:`~glmbench.adapters.genomeocean.
GenomeOceanAdapter` for the HF-repo-id + ``revision`` + ``trust_remote_code`` + ``hf_revision``
digest plumbing, with the single-nt causal-LM scoring / ``logprob_embedding`` shape), adapted
for Evo 1.5's defining differences:

- **StripedHyena — a custom architecture** (hybrid gated-Hyena conv + 3 attention layers, no
  in-tree ``transformers`` class). So ``trust_remote_code=True`` is **FORCED** (the inverse of
  the GenomeOcean stock-arch trap), and the modeling code lives in a **separate** HF repo
  (``togethercomputer/evo-1-131k-base``, cross-repo ``auto_map``) — pin both ``revision``
  (weights) and ``code_revision`` (that code repo; the ``1.1_fix`` branch's commit).
  Verified to load on ``transformers`` 5.12.1.
- **flash-attn is effectively REQUIRED.** The remote ``AttentionBlock`` instantiates
  ``flash_attn``'s ``MHA`` with **no SDPA/eager fallback**, so the 3 attention layers crash
  without it. Where no prebuilt wheel fits the system, build ``flash-attn`` from source for
  the GPU's architecture only (docs/INSTALL.md). The FFT-conv
  custom kernels are OFF in this config (``use_flashfft=false``, ``use_flash_depthwise=false``)
  → not needed.
- **Byte / single-nucleotide tokenizer** (``ByteTokenizer``: ``id = clamp(ord(char), 32,
  511)``, vocab 512, **no special tokens, no wrapping**). The runner reimplements this one-line
  mapping directly (the ``auto_map`` for the tokenizer is inconsistent across the two repos, so
  we avoid ``AutoTokenizer``). 1 nt/token ⇒ ``token_spans`` are unit-width and
  ``logprob_embedding`` (one log-prob per nucleotide) is servable.
- **Embeddings need forward HOOKS, not ``output_hidden_states``.** The HF wrapper's ``forward``
  hardcodes ``hidden_states=None``, so the runner registers forward hooks on
  ``model.backbone.embedding_layer`` (layer 0) and ``model.backbone.blocks[i]`` (layer i+1) to
  capture per-layer activations in one pass.

Scoring follows Evo's own convention (``evo/scoring.py``): prepend the ``eod`` token (id 0) as
causal context (``prepend_bos=True``), then the standard causal shift scores **all** ``L``
nucleotides, reduced ``mean`` (default) or ``sum``. That prefix is what makes
``logprob_embedding`` a clean length-``L`` vector with ``mean(z) == score(mean)``.

Capabilities: ``{TOKENIZE, SEQUENCE_LOGLIKELIHOOD, PER_TOKEN_LOGPROBS, EMBEDDING,
LOGPROB_EMBEDDING}`` — the full single-nt causal-LM set. ``MASKED_MARGINAL_LLR`` is N/A (a
causal LM has a real likelihood; the masked-marginal fallback is for MLM encoders).

Weights digest: ``hf_revision`` (pin the HF commit) by default — no 13 GB read, fully
reproducible; ``file_sha256`` works when ``checkpoint`` is a local download dir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from glmbench import registry

if TYPE_CHECKING:
    from glmbench.config.model_spec import ModelSpec

from .base import Capability, EmbeddingResult, ModelAdapter, Readout
from .hashing import compute_weights_digest, model_hash
from .runner_backend import DockerRunner, LocalSubprocessRunner, RunnerBackend
from .wire import WireProtocolError

logger = logging.getLogger(__name__)

EVO_RUNNER_MODULE = "glmbench.runners.evo_runner"
# The remote modeling code lives in this repo (cross-repo auto_map from the weights repo).
# A commit, not the `1.1_fix` branch it is the head of: a branch can move under a fixed hash.
DEFAULT_CODE_REVISION = "c206aab77ae5967a069c4200ecb1858588528c9d"
# Evo's ByteTokenizer end-of-document / eos id — what evo/scoring.py prepends (prepend_bos).
DEFAULT_EOD_TOKEN_ID = 0
# Trained context window (max_seqlen). 1 nt/token ⇒ this is both the token cap and the nt cap.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_CONTEXT_NT = 8192

# Arch fields lifted from the StripedHyena config.json into the model-hash config (so the hash
# is sensitive to the architecture, not just the weight digest). All read torch-free.
_ARCH_KEYS = (
    "model_type",
    "num_layers",
    "hidden_size",
    "inner_mlp_size",
    "num_attention_heads",
    "num_filters",
    "attn_layer_idxs",
    "hyena_layer_idxs",
    "short_filter_length",
    "state_size",
    "vocab_size",
    "max_seqlen",
    "rotary_emb_base",
    "eps",
    "tie_embeddings",
    "torch_dtype",
    "dtype",
)

_CAPS = frozenset(
    {
        Capability.TOKENIZE,
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.EMBEDDING,
        Capability.LOGPROB_EMBEDDING,
    }
)


def _read_hf_config(checkpoint: str) -> dict[str, Any]:
    """Read arch-defining fields from a local HF ``config.json`` (torch-free).

    When ``checkpoint`` is a HuggingFace repo id (not a local dir) there is no local config;
    the arch degrades to ``{family}`` and the ``hf_revision`` digest still keys the hash to the
    exact weights. Architecture fields add architecture sensitivity when present.
    """
    arch: dict[str, Any] = {"family": "evo"}
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.exists():
        return {"model": arch}
    data = json.loads(cfg_path.read_text()) or {}
    for k in _ARCH_KEYS:
        if k in data:
            arch[k] = data[k]
    return {"model": arch}


@registry.register("adapter", "evo")
class EvoAdapter(ModelAdapter):
    """Adapter for Evo 1.5 (StripedHyena, byte-level, ``trust_remote_code``)."""

    name = "evo"
    adapter_version = "0.1.0"
    CAPABILITIES = _CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "by_construction"
    readout_note = (
        "Evo's HF wrapper returns hidden_states=None, so evo_runner._embed hooks the "
        "residual stream directly: a forward-PRE-hook on backbone.blocks[0] for layer 0 (the "
        "post-embedding tensor) and a forward hook on each backbone.blocks[j] for layer j+1. "
        "No model-level norm is ever applied — nothing to strip."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        revision: str | None = None,
        code_revision: str | None = DEFAULT_CODE_REVISION,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        score_prefix: str = "eos",
        eod_token_id: int = DEFAULT_EOD_TOKEN_ID,
        max_context: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        batch_size: int = 4,
        token_budget: int | None = None,
        device: str = "auto",
        backend: str = "local",
        python_exe: str = "python",
        image: str | None = None,
        gpus: str = "0",
        docker_exe: str = "docker",
        extra_docker_args: list[str] | None = None,
        weights_digest_strategy: str = "hf_revision",
        weights_digest_value: str | None = None,
        weights_digest_cache: str | None = None,
        scratch_dir: str | None = None,
        keep_scratch: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.revision = revision
        self.code_revision = code_revision
        self.dtype = dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.score_prefix = score_prefix
        self.eod_token_id = int(eod_token_id)
        self.max_tokens = int(max_tokens)
        self.batch_size = batch_size
        # Cap on batch x padded_len: bounds the real tensor (compute AND memory) and
        # adapts on its own. batch_size alone is a sequence count, and _batching sorts
        # ascending by length, so a fixed count OOMs on the long tail even when it is
        # comfortable at the start of the pass. None -> batch_size alone applies.
        self.token_budget = token_budget
        self.device = device
        self.backend = backend
        self.python_exe = python_exe
        self.image = image
        self.gpus = gpus
        self._weights_digest_strategy = weights_digest_strategy
        self._weights_digest_value = weights_digest_value
        self._weights_digest_cache = weights_digest_cache

        self._arch = _read_hf_config(checkpoint)
        model = self._arch["model"]
        self.embedding_dim = int(model.get("hidden_size", 0))
        self.n_layers = int(model.get("num_layers", 0))
        # max_context is in NUCLEOTIDES (tasks chunk by it); 1 nt/token ⇒ equals the token
        # window. The runner enforces the true token ceiling (max_tokens) as the backstop.
        self.max_context = (
            int(max_context) if max_context is not None else DEFAULT_MAX_CONTEXT_NT
        )

        self._backend: RunnerBackend
        if backend == "docker":
            if not image:
                raise ValueError(
                    "EvoAdapter backend 'docker' needs an explicit runner.image "
                    "(a container with torch + transformers + flash-attn); there is no default "
                    "image. The default backend is 'local'."
                )
            self._backend = DockerRunner(
                image=image,
                runner_module=EVO_RUNNER_MODULE,
                python_exe="python",
                gpus=gpus,
                docker_exe=docker_exe,
                extra_args=self._docker_mounts(extra_docker_args or []),
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        elif backend == "local":
            self._backend = LocalSubprocessRunner(
                python_exe=python_exe,
                runner_module=EVO_RUNNER_MODULE,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        else:
            raise ValueError(
                f"EvoAdapter backend must be 'local' or 'docker', got {backend!r}."
            )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> EvoAdapter:
        """Build an EvoAdapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``checkpoint`` (a HF repo id like
        ``evo-design/evo-1.5-8k-base``, or a local download dir), ``revision`` (pin the weights
        commit), ``code_revision`` (the cross-repo remote-code commit; default: the head of
        ``1.1_fix``),
        ``dtype``, ``trust_remote_code`` (default ``True`` — forced by the custom arch),
        ``score_prefix``, ``eod_token_id``, ``max_context`` (nt), ``max_tokens`` (token
        ceiling), ``batch_size``. ``spec.runner`` selects the backend (``local`` default — the
        runner env's ``python_exe``) and ``gpus`` (→ device).
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "EvoAdapter spec.model must set 'checkpoint' (a HF repo id like "
                "'evo-design/evo-1.5-8k-base', or a local download dir)."
            )
        extra = spec.runner.extra
        max_context = model.get("max_context")
        return cls(
            checkpoint=str(checkpoint),
            revision=model.get("revision"),
            code_revision=model.get("code_revision", DEFAULT_CODE_REVISION),
            dtype=str(model.get("dtype", "bfloat16")),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            score_prefix=str(model.get("score_prefix", "eos")),
            eod_token_id=int(model.get("eod_token_id", DEFAULT_EOD_TOKEN_ID)),
            max_context=int(max_context) if max_context is not None else None,
            max_tokens=int(model.get("max_tokens", DEFAULT_MAX_TOKENS)),
            batch_size=int(extra.get("batch_size", model.get("batch_size", 4))),
            token_budget=(
                int(tb)
                if (tb := extra.get("token_budget", model.get("token_budget")))
                else None
            ),
            device=_device_from_gpus(spec.runner.gpus),
            backend=spec.runner.backend,
            python_exe=spec.runner.python_exe or "python",
            image=spec.runner.image,
            gpus=spec.runner.gpus or "0",
            docker_exe=str(extra.get("docker_exe", "docker")),
            extra_docker_args=list(extra.get("docker_args", [])),
            weights_digest_strategy=spec.weights_digest.strategy,
            weights_digest_value=spec.weights_digest.value,
            weights_digest_cache=model.get("weights_digest_cache"),
            scratch_dir=spec.runner.scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    # --- metadata ----------------------------------------------------------

    def model_hash(self) -> str:
        digest = compute_weights_digest(
            self._weights_digest_strategy,
            path=self.checkpoint,
            value=self._weights_digest_value or self.revision,
            cache_path=self._weights_digest_cache,
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
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "code_revision": self.code_revision,
            "backend": self.backend,
            "image": self.image if self.backend == "docker" else None,
            "python_exe": self.python_exe if self.backend == "local" else None,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "score_prefix": self.score_prefix,
            "eod_token_id": self.eod_token_id,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "max_context": self.max_context,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "gpus": self.gpus,
            "arch": self._arch["model"],
            "tokenizer_kind": (
                "byte-level / single-nucleotide (ByteTokenizer: id = clamp(ord(char), 32, 511), "
                "vocab 512, no special tokens, no wrapping) ⇒ 1 nt/token"
            ),
            "real_tokens": (
                "scoring: Evo's convention (evo/scoring.py) — prepend eod (id "
                f"{self.eod_token_id}) as causal context, then the standard causal shift scores "
                "ALL nucleotides; mean|sum per-token log-prob (prefix + pads masked from the "
                "reduction)."
            ),
            "embedding_real_tokens": (
                "embedding: per-layer activations via forward HOOKS (backbone.embedding_layer = "
                "layer 0; backbone.blocks[i] = layer i+1) since the HF wrapper returns "
                "hidden_states=None; any/all layers in ONE pass; no prefix prepended. "
                "pool='none' returns per-nt vectors + unit-width token_spans."
            ),
            "logprob_embedding": (
                "per-NUCLEOTIDE conditional log-prob (1 nt/token ⇒ length == len(seq)); same "
                "eod prefix + gather as scoring, so mean(z) == score(mean)."
            ),
        }

    # --- request plumbing --------------------------------------------------

    def _docker_mounts(self, user_extra: list[str]) -> list[str]:
        """Mount the gLMBench src (+ a local checkpoint dir, if any) into the container."""
        import glmbench

        src_dir = str(Path(glmbench.__file__).resolve().parents[1])
        args: list[str] = [
            "--ipc=host",
            "-v",
            f"{src_dir}:{src_dir}",
            "-e",
            f"PYTHONPATH={src_dir}",
        ]
        # Only bind-mount the checkpoint if it is a local path (HF repo ids are fetched
        # inside the container from the hub).
        ckpt_path = Path(self.checkpoint)
        if ckpt_path.exists():
            ckpt = str(ckpt_path.resolve())
            args += ["-v", f"{ckpt}:{ckpt}"]
        args += user_extra
        return args

    def _resolve_ref(self, value: str) -> str:
        """A local checkpoint dir is passed as an absolute path; a HF repo id
        (``org/name``, no local dir) is passed through verbatim for the hub."""
        p = Path(value)
        return str(p.resolve()) if p.exists() else value

    def _common_params(self) -> dict[str, Any]:
        return {
            "checkpoint": self._resolve_ref(self.checkpoint),
            "revision": self.revision,
            "code_revision": self.code_revision,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "score_prefix": self.score_prefix,
            "eod_token_id": self.eod_token_id,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "max_tokens": self.max_tokens,
        }

    # --- capability-gated ops ----------------------------------------------

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        result = self._backend.execute(
            "tokenize", self._common_params(), sequences, output_name="output.json"
        )
        tokens = result.payload["json"]
        if len(tokens) != len(sequences):
            raise WireProtocolError(
                f"tokenize returned {len(tokens)} results for {len(sequences)} inputs."
            )
        return [list(map(int, t)) for t in tokens]

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        """Per-sequence causal log-likelihood over all nucleotides (mean|sum)."""
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"EvoAdapter.score_sequences supports reduction 'mean'|'sum', got {reduction!r}."
            )
        params = self._common_params()
        params["reduction"] = reduction
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

    def per_token_logprobs(self, sequences: list[str]) -> list[np.ndarray]:
        """One conditional log-prob per **nucleotide** per sequence (1 nt/token)."""
        result = self._backend.execute(
            "per_token_logprobs", self._common_params(), sequences, output_name="output.npz"
        )
        n = int(result.payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"per_token_logprobs returned {n} arrays for {len(sequences)} inputs."
            )
        out: list[np.ndarray] = []
        for i in range(n):
            z = np.asarray(result.payload[f"arr_{i}"], dtype=np.float64)
            if not np.isfinite(z).all():
                raise WireProtocolError(
                    "per_token_logprobs returned non-finite values — refusing to poison tasks."
                )
            out.append(z)
        return out

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        """Per-nucleotide conditional log-prob vector ``z`` per sequence.

        The reduction-free body of :meth:`score_sequences`: same ``eod`` prefix, same
        per-token gather, but the un-collapsed per-token vector. 1 char/token ⇒ 1 token ⇄ 1 nt,
        so ``z`` has length ``len(seq)`` and ``mean(z) == score_sequences(mean)`` by
        construction. Raises on length drift or non-finite values.
        """
        result = self._backend.execute(
            "logprob_embedding", self._common_params(), sequences, output_name="output.npz"
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
            if z.shape != (len(seq),):
                raise WireProtocolError(
                    f"logprob_embedding[{i}] length {z.shape} != ({len(seq)},) nt for the "
                    f"input sequence (one log-prob per nucleotide; 1 char/token expected)."
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
        """Hidden-state embeddings — any/all layers from one HOOKED forward pass.

        Layer convention (resolved in the runner): layer ``0`` = embedding-layer output
        (``backbone.embedding_layer``); layer ``i`` = block ``i`` output
        (``backbone.blocks[i-1]``); ``'all'`` → ``0..n_layers`` (``n_layers+1`` arrays);
        ``'last'`` → the final block; a list selects indices (negatives ok). Because Evo's HF
        wrapper returns ``hidden_states=None``, the runner captures these via forward hooks.

        Pooling over the real nucleotide positions (no prefix prepended): ``mean``/``max``/
        ``last`` reduce the token axis → ``[len(layers), dim]`` per sequence; ``none`` returns
        the full per-nt tensor ``[n_tokens, len(layers), dim]`` plus unit-width ``token_spans``.
        """
        params = self._common_params()
        params["layers"] = layers
        params["pool"] = pool
        result = self._backend.execute("embed", params, sequences, output_name="output.npz")
        payload = result.payload
        layer_ids = [int(x) for x in payload["layer_ids"]]
        emb_dim = int(payload["embedding_dim"])
        if pool == "none":
            n = int(payload["n"])
            if n != len(sequences):
                raise WireProtocolError(
                    f"embed(pool='none') returned {n} arrays for {len(sequences)} inputs."
                )
            arrays = [payload[f"arr_{i}"] for i in range(n)]
            token_spans = [
                [(int(a), int(b)) for a, b in payload[f"span_{i}"]] for i in range(n)
            ]
            for arr in arrays:
                if np.isnan(arr).any():
                    raise WireProtocolError("embed returned NaN — refusing to poison tasks.")
            return EmbeddingResult(
                arrays=arrays,
                layers=layer_ids,
                pool=pool,
                token_spans=token_spans,
                embedding_dim=emb_dim,
            )
        stacked = payload["arrays"]
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
