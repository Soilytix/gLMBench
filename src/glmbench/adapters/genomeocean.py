"""GenomeOceanAdapter — the plain torch+transformers path for GenomeOcean (BPE Mistral).

Wraps JGI's GenomeOcean models (``pGenomeOcean/GenomeOcean-{100M,500M}``) behind the
model-agnostic adapter contract. The heavy lifting (load the HF model + tokenizer, run
forwards) happens in :mod:`glmbench.runners.genomeocean_runner`, which runs in any env with
``torch`` + ``transformers`` (the ``glmbench`` env after ``pip install -e ".[hf]"``). This
module is **core (torch-free)** — it only assembles wire requests, reads the HF
``config.json`` for the model-hash arch (plain JSON, no torch, when the checkpoint is a
local dir), builds the runner backend, and validates responses.

This is the **stock-HF template** adapter, with GenomeOcean's defining differences from a
single-nucleotide causal LM:

- **BPE tokenizer** (DNABERT-2-style, vocab 4096, **variable nt/token**) — not
  1-char/token. So ``token_spans`` come from the fast tokenizer's offset mapping, and
  ``per_token_logprobs`` is **per BPE token**, not per nucleotide.
- **``[CLS] … [SEP]`` wrapping** with BERT-style specials used as BOS/EOS.
- **Stock load, no ``trust_remote_code``** — the architecture is plain
  ``MistralForCausalLM``; the repos' bundled ``modeling_mistral.py`` crashes on
  transformers ≥5 (removed ``DynamicCache.from_legacy_cache``), so we use the in-tree
  Mistral. ``trust_remote_code`` is a knob (default ``False``).

Capabilities: ``{TOKENIZE, SEQUENCE_LOGLIKELIHOOD, PER_TOKEN_LOGPROBS, EMBEDDING}``.
``LOGPROB_EMBEDDING`` is **not** declared: that contract is one log-prob *per nucleotide*,
which a variable-width BPE tokenizer cannot produce (1 token ⇄ many nt) — it surfaces as a
clean first-class ``N/A`` (as for any multi-nucleotide tokenizer).

Context: the 100M/500M models are trained at **1024 tokens** (``max_position_embeddings``
of 32768 is the RoPE ceiling, not the usable window). Since tasks chunk by ``max_context``
in **nucleotides**, the spec sets ``max_context`` to a conservative nt bound (~5000) and the
runner hard-caps tokenization at ``max_tokens`` (1024) as the loud backstop.

Weights digest: ``hf_revision`` (pin the HF commit) by default — no multi-GB read, fully
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

GENOMEOCEAN_RUNNER_MODULE = "glmbench.runners.genomeocean_runner"
# The model's trained token window (max_position_embeddings=32768 is the RoPE ceiling).
DEFAULT_MAX_TOKENS = 1024
# Conservative nt bound for task-level chunking (~1024 BPE tokens at ~5 nt/token).
DEFAULT_MAX_CONTEXT_NT = 5000

# Arch fields lifted from the HF config.json into the model-hash config (so the hash is
# sensitive to the architecture, not just the weight digest). All read torch-free.
_ARCH_KEYS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "rope_parameters",
    "rope_scaling",
    "rms_norm_eps",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
)

_CAPS = frozenset(
    {
        Capability.TOKENIZE,
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.EMBEDDING,
    }
)


def _read_hf_config(checkpoint: str) -> dict[str, Any]:
    """Read arch-defining fields from a local HF ``config.json`` (torch-free).

    When ``checkpoint`` is a HuggingFace repo id (not a local dir) there is no local
    config; the arch degrades to ``{family}`` and the ``hf_revision`` digest still keys the
    hash to the exact weights. Architecture fields add architecture sensitivity when present.
    """
    arch: dict[str, Any] = {"family": "genomeocean"}
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.exists():
        return {"model": arch}
    data = json.loads(cfg_path.read_text()) or {}
    for k in _ARCH_KEYS:
        if k in data:
            arch[k] = data[k]
    return {"model": arch}


@registry.register("adapter", "genomeocean")
class GenomeOceanAdapter(ModelAdapter):
    """Adapter for GenomeOcean checkpoints (stock HuggingFace ``MistralForCausalLM``, BPE)."""

    name = "genomeocean"
    # 0.2.0: `embed` returns the RAW last-block output, not norm(last_block_output).
    # A semantics change bumps the version, so the hash moves and old rows are not reused.
    adapter_version = "0.2.0"
    CAPABILITIES = _CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "probed"
    readout_note = (
        "Mistral-style pre-LN: hidden_states[-1] measured rel_delta 5.4 / cos 0.983 away "
        "from the last block's raw output, so the runner hooks the block."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        tokenizer: str | None = None,
        revision: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        max_context: int | None = None,
        n_layers: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        pool_include_special: bool = True,
        batch_size: int = 8,
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
        self.tokenizer = tokenizer if tokenizer is not None else checkpoint
        self.revision = revision
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.max_tokens = int(max_tokens)
        self.pool_include_special = bool(pool_include_special)
        self.batch_size = batch_size
        # Cap on batch x padded_len: bounds the real tensor (compute AND memory) and
        # adapts on its own. batch_size alone is a sequence count, so its cost swings
        # ~9x between a 900 nt and an 8192 nt window. None -> batch_size alone applies.
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
        # `n_layers` is the model depth `describe()` reports: how many taps a full sweep
        # yields. It comes from the HF config when the checkpoint is a local dir; for a bare
        # hub repo id there is no local config.json to read and it lands at 0, so
        # `spec.model.n_layers` lets the spec supply it.
        #
        # DELIBERATELY NOT part of `_arch`, and therefore not part of the model hash: it is a
        # description of the weights, not a property of them, and folding it in would change
        # the hash of every existing spec for a field their runs already satisfy.
        self.n_layers = int(n_layers) if n_layers else int(model.get("num_hidden_layers", 0))
        # max_context is in NUCLEOTIDES (tasks chunk by it); default to a conservative bound
        # for the 1024-token window. The runner enforces the true token ceiling (max_tokens).
        self.max_context = int(max_context) if max_context is not None else DEFAULT_MAX_CONTEXT_NT

        self._backend: RunnerBackend
        if backend == "docker":
            if not image:
                raise ValueError(
                    "GenomeOceanAdapter backend 'docker' needs an explicit runner.image "
                    "(a container with torch + transformers); there is no default "
                    "image. The default backend is 'local'."
                )
            self._backend = DockerRunner(
                image=image,
                runner_module=GENOMEOCEAN_RUNNER_MODULE,
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
                runner_module=GENOMEOCEAN_RUNNER_MODULE,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        else:
            raise ValueError(
                f"GenomeOceanAdapter backend must be 'local' or 'docker', got {backend!r}."
            )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> GenomeOceanAdapter:
        """Build a GenomeOceanAdapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``checkpoint`` (a HF repo id or a local download dir),
        optional ``tokenizer`` (default = checkpoint), ``revision`` (pin the HF commit),
        ``dtype``, ``attn_implementation``, ``trust_remote_code``, ``max_context`` (nt),
        ``max_tokens`` (token ceiling), ``pool_include_special``, ``batch_size``.
        ``spec.runner`` selects the backend (``local`` default — the runner env's
        ``python_exe``) and ``gpus`` (→ device).
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "GenomeOceanAdapter spec.model must set 'checkpoint' (a HF repo id like "
                "'pGenomeOcean/GenomeOcean-100M', or a local download dir)."
            )
        extra = spec.runner.extra
        max_context = model.get("max_context")
        return cls(
            checkpoint=str(checkpoint),
            tokenizer=model.get("tokenizer"),
            revision=model.get("revision"),
            dtype=str(model.get("dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation"),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            max_context=int(max_context) if max_context is not None else None,
            n_layers=model.get("n_layers"),
            max_tokens=int(model.get("max_tokens", DEFAULT_MAX_TOKENS)),
            pool_include_special=bool(model.get("pool_include_special", True)),
            batch_size=int(extra.get("batch_size", model.get("batch_size", 8))),
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
            "tokenizer": self.tokenizer,
            "revision": self.revision,
            "backend": self.backend,
            "image": self.image if self.backend == "docker" else None,
            "python_exe": self.python_exe if self.backend == "local" else None,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "max_context": self.max_context,
            "max_tokens": self.max_tokens,
            "pool_include_special": self.pool_include_special,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "gpus": self.gpus,
            "arch": self._arch["model"],
            "tokenizer_kind": (
                "BPE (DNABERT-2-style, vocab 4096, variable nt/token); sequences wrapped "
                "[CLS]…[SEP]; specials [CLS]=1 [SEP]=2 [PAD]=3 [UNK]=0"
            ),
            "real_tokens": (
                "scoring: every real BPE token (special_tokens_mask==0), with [CLS] as causal "
                "context; [SEP]/[PAD] masked from the loss; mean|sum per-token causal log-prob"
            ),
            "embedding_real_tokens": (
                "embedding: pool_include_special=True (vendor default) pools over all non-pad "
                "tokens incl [CLS]/[SEP]; False → real BPE tokens only. Layers via "
                "output_hidden_states (0=embedding output, i=block i); any/all in ONE pass. "
                "pool='none' returns per-real-token vectors + BPE nt token_spans."
            ),
            "logprob_embedding": (
                "N/A — per-NUCLEOTIDE log-probs require 1 nt/token; BPE is variable-width "
                "(clean first-class N/A, as for any multi-nucleotide tokenizer)."
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
            tok_path = Path(self.tokenizer)
            if tok_path.exists() and str(tok_path.resolve()) != ckpt:
                args += ["-v", f"{tok_path.resolve()}:{tok_path.resolve()}"]
        args += user_extra
        return args

    def _resolve_ref(self, value: str) -> str:
        """A local checkpoint/tokenizer dir is passed as an absolute path; a HF repo id
        (``org/name``, no local dir) is passed through verbatim for the hub."""
        p = Path(value)
        return str(p.resolve()) if p.exists() else value

    def _common_params(self) -> dict[str, Any]:
        return {
            "checkpoint": self._resolve_ref(self.checkpoint),
            "tokenizer": self._resolve_ref(self.tokenizer),
            "revision": self.revision,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
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
        """Per-sequence causal log-likelihood over the real BPE tokens (mean|sum)."""
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"GenomeOceanAdapter.score_sequences supports reduction 'mean'|'sum', got "
                f"{reduction!r}."
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
        """One conditional log-prob per **real BPE token** per sequence (BPE granularity)."""
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

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        """Hidden-state embeddings — any/all layers from one HF forward pass.

        Layer convention (resolved in the runner): layer ``0`` = embedding output; layer
        ``i`` = block ``i``; ``'all'`` → ``0..n_layers``; ``'last'`` → final block; a list
        selects indices (negatives ok).

        Pooling (``pool_include_special`` set on the adapter — default True matches the
        vendor's mask-only-padding mean): ``mean``/``max``/``last`` reduce the token axis →
        ``[len(layers), dim]`` per sequence; ``bos``/``cls`` read the ``[CLS]`` position;
        ``none`` returns the full per-real-token tensor ``[n_tokens, len(layers), dim]`` plus
        ``token_spans`` (half-open BPE nt spans).
        """
        params = self._common_params()
        params["layers"] = layers
        params["pool"] = pool
        params["pool_include_special"] = self.pool_include_special
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
