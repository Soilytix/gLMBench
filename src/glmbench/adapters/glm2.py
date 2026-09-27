"""gLM2Adapter — TattaBio's mixed-modality masked-LM encoder (gLM2_150M / gLM2_650M).

Wraps TattaBio's gLM2 (`tattabio/gLM2_{150M,650M}`) behind the model-agnostic adapter
contract. The heavy lifting (load the HF model + tokenizer, run forwards) happens in
:mod:`glmbench.runners.glm2_runner`, which runs in any env with ``torch`` +
``transformers`` + ``einops`` (the ``glmbench`` env after ``pip install -e ".[hf]"``). This
module is **core (torch-free)** — it only assembles wire requests, reads the HF
``config.json`` for the model-hash arch (plain JSON, no torch, when the checkpoint is a
local dir), builds the runner backend, and validates responses.

Copied from :class:`~glmbench.adapters.genomeocean.GenomeOceanAdapter` (the stock-HF, local,
HF-repo-id launch shape) and **stripped to embeddings-only**, because gLM2 is the suite's
first **encoder**:

- **Masked-LM, bidirectional encoder** (``gLM2ForMaskedLM``). Causal scoring is undefined,
  so ``SEQUENCE_LOGLIKELIHOOD`` / ``PER_TOKEN_LOGPROBS`` / ``LOGPROB_EMBEDDING`` are **N/A**
  (clean first-class — a task needing them names the missing capability, never crashes).
  But the bidirectional MLM head *does* serve **``MASKED_MARGINAL_LLR``** — the
  variant-effect scoring path (mask each mutated position on the WT background, read the
  mut−wt logit difference; see :meth:`~glmbench.adapters.base.ModelAdapter.score_variant_llr`).
  ``embed``/``tokenize`` load via ``AutoModel`` (base encoder → ``last_hidden_state`` +
  ``hidden_states``, no ``logits``); ``score_variant_llr`` loads ``gLM2ForMaskedLM`` via
  ``AutoModelForMaskedLM`` (for the ``logits`` head). Capabilities:
  ``{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}``.
- **``trust_remote_code=True`` is FORCED** — gLM2 is a custom arch with no in-tree impl
  (unlike GenomeOcean's stock Mistral). The bundled code loads on transformers 5.x
  (probe-confirmed); ``revision`` pins the commit for reproducibility + to satisfy the
  remote-code-review warning. One extra runner dep: ``einops``.
- **Mixed-modality tokenizer (case-sensitive!).** gLM2 encodes amino acids UPPERCASE and
  nucleotides **lowercase**; uppercase ``ACGT`` is silently read as amino acids. The runner
  **lowercases** every input. Every genomic element is prefixed by a **strand marker**
  (``<+>``=33 / ``<->``=34, ordinary vocab tokens, not "special"); the runner prepends one
  (``strand_prefix``, default ``"+"``) so a bare contig is in-distribution. 1 nt/token (the
  nucleotide vocab has no multi-nt merges), so ``token_spans`` are unit-width — but built in
  **original-string** coordinates (marker excluded), not from the offset mapping (which
  indexes the lowercased+marker-prefixed string).

Context: gLM2 is trained at **4096 tokens** (``config.json`` has no ``max_position_embeddings``
— it is RoPE). Tasks chunk by ``max_context`` in **nucleotides**, so the spec sets
``max_context`` ≈ 4095 nt (4096 tok − 1 marker) and the runner hard-caps tokenization at
``max_tokens`` (4096) as the loud backstop.

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

GLM2_RUNNER_MODULE = "glmbench.runners.glm2_runner"
# gLM2's trained token window (config.json has no max_position_embeddings — RoPE).
DEFAULT_MAX_TOKENS = 4096
# Conservative nt bound for task-level chunking (1 nt/token; 4096 tok − 1 strand marker).
DEFAULT_MAX_CONTEXT_NT = 4095

# Arch fields lifted from the HF config.json into the model-hash config (so the hash is
# sensitive to the architecture, not just the weight digest). All read torch-free. gLM2 uses
# its own key names (``dim``/``depth``/``heads``), not the Llama/Mistral ones.
_ARCH_KEYS = (
    "model_type",
    "depth",  # n transformer layers
    "dim",  # hidden size
    "heads",
    "vocab_size",
    "norm_eps",
    "ffn_dim_multiplier",
    "swiglu_multiple_of",
    "max_position_embeddings",
    "torch_dtype",  # transformers <5
    "dtype",  # transformers >=5
)

_CAPS = frozenset(
    {Capability.TOKENIZE, Capability.EMBEDDING, Capability.MASKED_MARGINAL_LLR}
)


def _read_hf_config(checkpoint: str) -> dict[str, Any]:
    """Read arch-defining fields from a local HF ``config.json`` (torch-free).

    When ``checkpoint`` is a HuggingFace repo id (not a local dir) there is no local config;
    the arch degrades to ``{family}`` and the ``hf_revision`` digest still keys the hash to
    the exact weights. Architecture fields add architecture sensitivity when present.
    """
    arch: dict[str, Any] = {"family": "glm2"}
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.exists():
        return {"model": arch}
    data = json.loads(cfg_path.read_text()) or {}
    for k in _ARCH_KEYS:
        if k in data:
            arch[k] = data[k]
    return {"model": arch}


@registry.register("adapter", "glm2")
class GLM2Adapter(ModelAdapter):
    """Adapter for TattaBio gLM2 (custom ``gLM2ForMaskedLM`` encoder; embeddings-only)."""

    name = "glm2"
    # Measured against a forward hook on the last block: gLM2 was already on the target
    # convention (raw block output), so `embed` never needed a change or a version bump.
    adapter_version = "0.1.0"
    CAPABILITIES = _CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "probed"

    @classmethod
    def expected_sweep_taps(cls, describe: dict[str, Any]) -> int | None:
        """gLM2 returns ``depth`` hidden states, NOT ``depth + 1``.

        Measured, not inferred from the tuple length: hidden_states[0] is **block 0's
        output** (delta 0.0 against a hook on it), while the input to block 0 differs from it
        by 4.33 — so layer 0 of the layer convention, the embedding-layer output, is not
        exposed at all.
        A complete gLM2 sweep therefore has ``n_layers`` taps and can never have
        ``n_layers + 1``; the default denominator would mark every full gLM2 sweep as one
        tap short forever.
        """
        n = describe.get("n_layers")
        return int(n) if isinstance(n, int) and n > 0 else None
    readout_note = (
        "No change needed — but NOT because gLM2 is post-LN. Its closing norm lives in "
        "`lm_head`, which the runner's AutoModel load drops entirely, so the encoder's last "
        "hidden state is already the raw block output (measured delta 0.0). Separately: "
        "gLM2's hidden_states tuple has `depth` entries and hidden_states[0] is **block 0's "
        "output**, not the embedding-layer output — layer 0 of the layer convention is not "
        "exposed at all, so a full gLM2 sweep has n_layers taps and can never have n_layers+1."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        tokenizer: str | None = None,
        revision: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        strand_prefix: str = "+",
        pool_include_special: bool = False,
        max_context: int | None = None,
        n_layers: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
        self.trust_remote_code = bool(trust_remote_code)
        if strand_prefix not in ("+", "-", "none"):
            raise ValueError(
                f"GLM2Adapter strand_prefix must be '+'|'-'|'none', got {strand_prefix!r}."
            )
        self.strand_prefix = strand_prefix
        self.pool_include_special = bool(pool_include_special)
        self.max_tokens = int(max_tokens)
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
        self.embedding_dim = int(model.get("dim", 0))
        # `n_layers` is the model depth `describe()` reports: how many taps a full sweep
        # yields. It comes from the HF config when the checkpoint is a local dir; for a bare
        # hub repo id there is no local config.json to read and it lands at 0, so
        # `spec.model.n_layers` lets the spec supply it.
        #
        # DELIBERATELY NOT part of `_arch`, and therefore not part of the model hash: it is a
        # description of the weights, not a property of them, and folding it in would change
        # the hash of every existing spec for a field their runs already satisfy.
        self.n_layers = int(n_layers) if n_layers else int(model.get("depth", 0))
        # max_context is in NUCLEOTIDES (tasks chunk by it); default to a conservative bound
        # for the 4096-token window. The runner enforces the true token ceiling (max_tokens).
        self.max_context = int(max_context) if max_context is not None else DEFAULT_MAX_CONTEXT_NT

        self._backend: RunnerBackend
        if backend == "docker":
            if not image:
                raise ValueError(
                    "GLM2Adapter backend 'docker' needs an explicit runner.image "
                    "(a container with torch + transformers + einops); there is no default "
                    "image. The default backend is 'local'."
                )
            self._backend = DockerRunner(
                image=image,
                runner_module=GLM2_RUNNER_MODULE,
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
                runner_module=GLM2_RUNNER_MODULE,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        else:
            raise ValueError(
                f"GLM2Adapter backend must be 'local' or 'docker', got {backend!r}."
            )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> GLM2Adapter:
        """Build a GLM2Adapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``checkpoint`` (a HF repo id like ``tattabio/gLM2_650M`` or a
        local download dir), optional ``tokenizer`` (default = checkpoint), ``revision``
        (pin the HF commit), ``dtype``, ``attn_implementation``, ``trust_remote_code``
        (default ``True`` — forced for the custom arch), ``strand_prefix`` (``+``/``-``/
        ``none``), ``pool_include_special``, ``max_context`` (nt), ``max_tokens`` (token
        ceiling), ``batch_size``. ``spec.runner`` selects the backend (``local`` default —
        the runner env's ``python_exe``) and ``gpus`` (→ device).
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "GLM2Adapter spec.model must set 'checkpoint' (a HF repo id like "
                "'tattabio/gLM2_650M', or a local download dir)."
            )
        extra = spec.runner.extra
        max_context = model.get("max_context")
        return cls(
            checkpoint=str(checkpoint),
            tokenizer=model.get("tokenizer"),
            revision=model.get("revision"),
            dtype=str(model.get("dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation"),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            strand_prefix=str(model.get("strand_prefix", "+")),
            pool_include_special=bool(model.get("pool_include_special", False)),
            max_context=int(max_context) if max_context is not None else None,
            n_layers=model.get("n_layers"),
            max_tokens=int(model.get("max_tokens", DEFAULT_MAX_TOKENS)),
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
            "strand_prefix": self.strand_prefix,
            "pool_include_special": self.pool_include_special,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "max_context": self.max_context,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "gpus": self.gpus,
            "arch": self._arch["model"],
            "objective": (
                "masked-LM bidirectional encoder (gLM2ForMaskedLM). Embeddings + "
                "masked-marginal LLR variant scoring (MASKED_MARGINAL_LLR). "
                "SEQUENCE_LOGLIKELIHOOD / PER_TOKEN_LOGPROBS / LOGPROB_EMBEDDING are N/A "
                "(causal likelihood undefined for an MLM) — clean first-class N/A; "
                "variant-effect tasks fall back to the masked-marginal LLR path."
            ),
            "tokenizer_kind": (
                "mixed-modality, CASE-SENSITIVE: amino acids UPPERCASE, nucleotides "
                "lowercase (uppercase ACGT → amino acids!). The runner lowercases input and "
                "prepends a strand marker (<+>=33/<->=34). 1 nt/token (no multi-nt merges); "
                "no cls/eos wrapping; no BOS. <unk>=3 for non-acgt bases."
            ),
            "embedding_real_tokens": (
                "embedding: pool_include_special=False (default) pools over nucleotide "
                "(content) tokens only, EXCLUDING the strand marker; True includes the "
                "marker. Layers via output_hidden_states (gLM2 returns `depth` states; the "
                "runner resolves layers against the actual tuple length). pool='none' "
                "returns per-content-token vectors + unit-width token_spans (marker excluded)."
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
            "strand_prefix": self.strand_prefix,
            "device": self.device,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "max_tokens": self.max_tokens,
        }

    # --- capability-gated ops ----------------------------------------------

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        """Token ids of the **model input** (lowercased nucleotides + the strand marker).

        These are the ids gLM2 actually consumes — the strand marker (when ``strand_prefix``
        != ``none``) is the leading id, followed by one id per nucleotide.
        """
        result = self._backend.execute(
            "tokenize", self._common_params(), sequences, output_name="output.json"
        )
        tokens = result.payload["json"]
        if len(tokens) != len(sequences):
            raise WireProtocolError(
                f"tokenize returned {len(tokens)} results for {len(sequences)} inputs."
            )
        return [list(map(int, t)) for t in tokens]

    def score_variant_llr(self, items: list[dict[str, Any]]) -> list[float]:
        """Masked-marginal log-likelihood ratio per variant (the MLM variant-effect path).

        Each item carries ``{"reference": <WT DNA str>, "mutations": [[pos0, wt, mut], ...]}``
        (0-based positions). The runner loads ``gLM2ForMaskedLM`` (for the ``logits`` head),
        masks each mutated position on the WT background, and returns
        ``Σ_i (logit_mut_i − logit_wt_i)`` per variant — 1:1 with *items*, in input order.
        One masked forward per **(reference, position)** pair is cached and reused across
        every variant touching that position (forward passes scale with unique masked
        positions, not variant count).
        """
        result = self._backend.execute_variant(
            "score_variant_llr", self._common_params(), items, output_name="output.npz"
        )
        scores = result.payload["scores"]
        if scores.shape[0] != len(items):
            raise WireProtocolError(
                f"score_variant_llr returned {scores.shape[0]} scores for "
                f"{len(items)} variants — must be 1:1 with its inputs."
            )
        if not np.isfinite(scores).all():
            raise WireProtocolError(
                "score_variant_llr returned non-finite values — refusing to poison metrics."
            )
        return [float(x) for x in scores]

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        """Hidden-state embeddings — any/all layers from one ``AutoModel`` forward.

        Layer convention (resolved in the runner against the **actual** ``hidden_states``
        tuple length — gLM2 returns ``depth`` states, not ``depth+1``): ``'last'`` → final
        state; ``'all'`` → every state; an int / list selects indices (negatives ok).

        Pooling (``pool_include_special`` set on the adapter — default ``False``):
        ``mean``/``max``/``last`` reduce the token axis over the **nucleotide (content)**
        tokens, excluding the strand marker → ``[len(layers), dim]`` per sequence; ``True``
        includes the marker. ``marker`` reads the strand-marker position. ``none`` returns
        the full per-content-token tensor ``[n_tokens, len(layers), dim]`` plus
        unit-width ``token_spans`` in original-string coordinates (marker excluded).
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
