"""NTv3Adapter — InstaDeep's Nucleotide Transformer v3 (U-Net masked-LM encoder).

Wraps ``InstaDeepAI/NTv3_100M_pre`` behind the model-agnostic adapter contract. The heavy
lifting (load the HF model + tokenizer, run forwards) happens in
:mod:`glmbench.runners.ntv3_runner`, which runs in any env with ``torch`` + ``transformers``
(the ``glmbench`` env, as-is — no extra install). This module is **core (torch-free)** — it
only assembles wire requests, reads the HF ``config.json`` for the model-hash arch (plain
JSON, no torch, when the checkpoint is a local dir), builds the runner backend, and validates
responses.

Copied from :class:`~glmbench.adapters.glm2.GLM2Adapter` (the closest pair: a masked-LM
**encoder** with a single-nt tokenizer, ``local`` backend, HF repo id, ``trust_remote_code``)
and adapted for NTv3's specifics:

- **Masked-LM U-Net** (``NTv3PreTrained``): ``conv tower (7× downsample) → 6-layer transformer
  torso → deconv tower (7× upsample) → LM head``. Causal scoring is undefined ⇒
  ``SEQUENCE_LOGLIKELIHOOD`` / ``PER_TOKEN_LOGPROBS`` / ``LOGPROB_EMBEDDING`` are **N/A**
  (clean first-class). The MLM ``logits`` head serves **``MASKED_MARGINAL_LLR``** (variant
  effects); ``output_hidden_states`` serves ``EMBEDDING``. Capabilities:
  ``{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}`` — same shape as gLM2/ProkBERT.
- **``trust_remote_code=True`` is FORCED** (custom ``ntv3`` arch, no in-tree impl), and the
  ``auto_map`` is the **cross-repo** ``InstaDeepAI/ntv3_base_model--…`` form (like ProkBERT):
  pin **both** ``revision`` (weights) and ``code_revision`` (the modeling-code repo). Loads on
  ``transformers 5.x`` (probe-confirmed on 5.12.1). No extra deps.
- **Single-nucleotide tokenizer** (``NTv3Tokenizer``, slow — no offset mapping),
  ``padding_side=right``, vocab 11 (``<unk>=0 <pad>=1 <mask>=2 <cls>=3 <eos>=4 <bos>=5 A=6 T=7
  C=8 G=9 N=10``). **No sequence wrapping** (``build_inputs_with_special_tokens`` is a no-op
  for a single sequence). 1 nt/token ⇒ ``token_spans`` unit-width; non-ACGTN → ``<unk>=0``.
- **The load-bearing quirk: the U-Net needs a length that is a multiple of ``2^num_downsamples``
  = 128, and its transformer torso runs with NO attention mask** (padding is not masked and
  contaminates real positions through global torso attention). So the runner pads **each
  sequence to its own next multiple of 128** and buckets equal-padded-length sequences into a
  batch — never to a batch-wide max — which makes scores/embeddings batch-invariant
  (probe-verified Δ = 0.0). ``pad_multiple`` (default 128) is a knob = ``2^num_downsamples``.
- **Mixed-resolution hidden states.** ``output_hidden_states`` returns ``2·num_downsamples
  + num_layers`` states at lengths ``L, L/2 … L/128 …(torso)… L/2, L`` (probe: 20 states for the
  100M). Default ``layers='last'`` = the final deconv state (nucleotide resolution, dim 768).
  Pooling reduces each selected layer over its own non-pad span (``ceil(L / f)`` positions,
  ``f = padded_len // Lh``); ``pool='none'`` returns per-position vectors + ``f``-wide
  ``token_spans`` (default final layer ⇒ unit-width, 1 nt/token).

Context: NTv3's usable window is very long (model-card claim: up to 1 Mb), but memory-bound;
``max_context`` (nucleotides, tasks chunk by it) defaults to a conservative 12288 (= 96×128)
and the runner hard-caps tokenization at ``max_tokens``.

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

NTV3_RUNNER_MODULE = "glmbench.runners.ntv3_runner"
# NTv3's U-Net downsamples 2^num_downsamples = 128×; input length MUST be a multiple of this.
DEFAULT_PAD_MULTIPLE = 128
# Conservative nucleotide window for task-level chunking (multiple of 128); the runner enforces
# the true token ceiling (max_tokens). 96 × 128 = 12288.
DEFAULT_MAX_CONTEXT_NT = 12288
DEFAULT_MAX_TOKENS = 12288  # 1 nt/token ⇒ token ceiling == max_context

# Arch fields lifted from the HF config.json into the model-hash config (so the hash is
# sensitive to the architecture, not just the weight digest). All read torch-free. NTv3 uses
# its own key names (``embed_dim``/``num_layers``/``attention_heads``), not the Llama ones.
_ARCH_KEYS = (
    "model_type",
    "num_layers",  # n transformer torso layers
    "embed_dim",  # hidden size
    "attention_heads",
    "key_size",
    "alphabet_size",  # vocab
    "ffn_embed_dim",
    "num_downsamples",
    "token_embed_dim",
    "conv_init_embed_dim",
    "use_skip_connection",
    "deconv_upsample_type",
    "layer_norm_eps",
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
    arch: dict[str, Any] = {"family": "ntv3"}
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.exists():
        return {"model": arch}
    data = json.loads(cfg_path.read_text()) or {}
    for k in _ARCH_KEYS:
        if k in data:
            arch[k] = data[k]
    return {"model": arch}


@registry.register("adapter", "ntv3")
class NTv3Adapter(ModelAdapter):
    """Adapter for InstaDeep NTv3 (custom ``NTv3PreTrained`` U-Net encoder; MLM)."""

    name = "ntv3"
    # The tuple was already on the target convention (see readout_note): no version bump.
    adapter_version = "0.1.0"
    CAPABILITIES = _CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "probed"

    @classmethod
    def expected_sweep_taps(cls, describe: dict[str, Any]) -> int | None:
        """NTv3's depth axis is a U-Net: ``2 * num_downsamples + num_layers`` states.

        embed → stem → ``num_downsamples`` conv (down) → ``num_layers`` transformer (at the
        bottleneck) → ``num_downsamples`` deconv (up), one hidden state appended per block.
        For the 100M model that is 2*7 + 6 = 20 taps of MIXED resolution — not a residual
        stream, and nothing like ``n_layers + 1``.
        """
        n = describe.get("n_layers")
        d = describe.get("num_downsamples")
        if isinstance(n, int) and n > 0 and isinstance(d, int) and d > 0:
            return 2 * d + n
        return None
    readout_note = (
        "No change needed, and this is the one architecture where trusting the hook over "
        "the tuple would have made things WORSE. NTv3Core's deconv loop is `y = block(x); "
        "y = y + r; x = y; hidden_states.append(x)` — the tuple holds the state AFTER the "
        "U-Net skip add, while a forward hook on the deconv block sees only `block(x)`, the "
        "branch BEFORE it. No model-level norm follows the deconv tower. Note the depth axis "
        "is 2*num_downsamples + num_layers MIXED-resolution states, not a residual stream."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        tokenizer: str | None = None,
        revision: str | None = None,
        code_revision: str | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        pool_include_special: bool = False,
        pad_multiple: int = DEFAULT_PAD_MULTIPLE,
        max_context: int | None = None,
        n_layers: int | None = None,
        num_downsamples: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        batch_size: int = 4,
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
        self.code_revision = code_revision
        self.dtype = dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.pool_include_special = bool(pool_include_special)
        if int(pad_multiple) <= 0:
            raise ValueError(f"NTv3Adapter pad_multiple must be positive, got {pad_multiple!r}.")
        self.pad_multiple = int(pad_multiple)
        self.max_tokens = int(max_tokens)
        self.batch_size = batch_size
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
        self.embedding_dim = int(model.get("embed_dim", 0))
        # `n_layers` is the model depth `describe()` reports: how many taps a full sweep
        # yields. It comes from the HF config when the checkpoint is a local dir; for a bare
        # hub repo id there is no local config.json to read and it lands at 0, so
        # `spec.model.n_layers` lets the spec supply it.
        #
        # DELIBERATELY NOT part of `_arch`, and therefore not part of the model hash: it is a
        # description of the weights, not a property of them, and folding it in would change
        # the hash of every existing spec for a field their runs already satisfy.
        self.n_layers = int(n_layers) if n_layers else int(model.get("num_layers", 0))
        self.num_downsamples = int(
            num_downsamples if num_downsamples else model.get("num_downsamples", 0)
        )
        # max_context is in NUCLEOTIDES (tasks chunk by it); default to a conservative bound.
        # The runner enforces the true token ceiling (max_tokens) + the 128-multiple padding.
        self.max_context = (
            int(max_context) if max_context is not None else DEFAULT_MAX_CONTEXT_NT
        )

        self._backend: RunnerBackend
        if backend == "docker":
            if not image:
                raise ValueError(
                    "NTv3Adapter backend 'docker' needs an explicit runner.image "
                    "(a container with torch + transformers); there is no default "
                    "image. The default backend is 'local'."
                )
            self._backend = DockerRunner(
                image=image,
                runner_module=NTV3_RUNNER_MODULE,
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
                runner_module=NTV3_RUNNER_MODULE,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        else:
            raise ValueError(
                f"NTv3Adapter backend must be 'local' or 'docker', got {backend!r}."
            )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> NTv3Adapter:
        """Build an NTv3Adapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``checkpoint`` (a HF repo id like ``InstaDeepAI/NTv3_100M_pre``
        or a local download dir), optional ``tokenizer`` (default = checkpoint), ``revision``
        (weights commit), ``code_revision`` (the cross-repo ``ntv3_base_model`` code commit),
        ``dtype``, ``trust_remote_code`` (default ``True`` — forced for the custom arch),
        ``pool_include_special``, ``pad_multiple`` (= ``2^num_downsamples``, default 128),
        ``max_context`` (nt), ``max_tokens`` (token ceiling), ``batch_size``. ``spec.runner``
        selects the backend (``local`` default — the runner env's ``python_exe``) and ``gpus``
        (→ device).
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "NTv3Adapter spec.model must set 'checkpoint' (a HF repo id like "
                "'InstaDeepAI/NTv3_100M_pre', or a local download dir)."
            )
        extra = spec.runner.extra
        max_context = model.get("max_context")
        return cls(
            checkpoint=str(checkpoint),
            tokenizer=model.get("tokenizer"),
            revision=model.get("revision"),
            code_revision=model.get("code_revision"),
            dtype=str(model.get("dtype", "bfloat16")),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            pool_include_special=bool(model.get("pool_include_special", False)),
            pad_multiple=int(model.get("pad_multiple", DEFAULT_PAD_MULTIPLE)),
            max_context=int(max_context) if max_context is not None else None,
            n_layers=model.get("n_layers"),
            num_downsamples=model.get("num_downsamples"),
            max_tokens=int(model.get("max_tokens", DEFAULT_MAX_TOKENS)),
            batch_size=int(extra.get("batch_size", model.get("batch_size", 4))),
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
            "code_revision": self.code_revision,
            "backend": self.backend,
            "image": self.image if self.backend == "docker" else None,
            "python_exe": self.python_exe if self.backend == "local" else None,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "pool_include_special": self.pool_include_special,
            "pad_multiple": self.pad_multiple,
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            # The other half of NTv3's depth denominator: its sweep yields
            # 2*num_downsamples + num_layers MIXED-resolution states, not n_layers + 1.
            "num_downsamples": self.num_downsamples,
            "max_context": self.max_context,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "batch_size": self.batch_size,
            "gpus": self.gpus,
            "arch": self._arch["model"],
            "objective": (
                "masked-LM U-Net encoder (NTv3PreTrained: conv-down → transformer torso → "
                "deconv-up → LM head). Embeddings + masked-marginal LLR variant scoring "
                "(MASKED_MARGINAL_LLR). SEQUENCE_LOGLIKELIHOOD / PER_TOKEN_LOGPROBS / "
                "LOGPROB_EMBEDDING are N/A (causal likelihood undefined for an MLM) — clean "
                "first-class N/A; variant-effect tasks fall back to the masked-marginal path."
            ),
            "tokenizer_kind": (
                "single-nucleotide, char-level (NTv3Tokenizer, slow — no offset mapping). "
                "1 nt/token; NO sequence wrapping (no <cls>/<eos>/<bos> added). vocab 11: "
                "<unk>=0 <pad>=1 <mask>=2 <cls>=3 <eos>=4 <bos>=5 A=6 T=7 C=8 G=9 N=10; "
                "non-ACGTN → <unk>=0."
            ),
            "padding_note": (
                "The U-Net requires length % pad_multiple == 0 (=2^num_downsamples=128) "
                "and the transformer torso runs with NO attention mask, so <pad> contaminates "
                "real positions. The runner pads EACH sequence to its own next multiple of "
                "pad_multiple and buckets equal-padded-length sequences — never batch-max — so "
                "scores/embeddings are batch-invariant."
            ),
            "embedding_note": (
                "hidden_states are at MIXED resolutions (conv-down L→L/128, torso L/128, "
                "deconv-up L/128→L). Default 'last' = final deconv (nucleotide resolution). "
                "Each selected layer pools over its own non-pad span (ceil(L/f) positions, "
                "f=padded_len//Lh). pool='none' returns per-position vectors + f-wide "
                "token_spans (final layer ⇒ unit-width). pool_include_special=False excludes "
                "the pad region (the norm)."
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
            "code_revision": self.code_revision,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "pad_multiple": self.pad_multiple,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
        }

    # --- capability-gated ops ----------------------------------------------

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        """Token ids of the model input (one id per nucleotide; no wrapping, capped)."""
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
        (0-based positions). The runner loads ``NTv3PreTrained`` (``AutoModelForMaskedLM``, for
        the ``logits`` head), masks each mutated position on the WT background (the reference
        padded to its own next multiple of ``pad_multiple``), and returns
        ``Σ_i (logit_mut_i − logit_wt_i)`` per variant — 1:1 with *items*, in input order.
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
        """Hidden-state embeddings — any/all layers from one forward pass.

        Layer convention (resolved in the runner against the **actual** ``hidden_states``
        tuple length — NTv3 returns ``2·num_downsamples + num_layers`` MIXED-resolution
        states): ``'last'`` → final deconv state (nucleotide resolution); ``'all'`` → every
        state; an int / list selects indices (negatives ok).

        Pooling (``pool_include_special`` set on the adapter — default ``False``, exclude the
        pad region): ``mean``/``max``/``last`` reduce each selected layer over its own non-pad
        span (``ceil(len(seq)/f)`` positions, ``f = padded_len // layer_len``) →
        ``[len(layers), dim]`` per sequence. ``none`` returns the full per-position tensor
        ``[n_tokens, len(layers), dim]`` plus ``f``-wide ``token_spans`` covering ``len(seq)``
        (final layer ⇒ unit-width, 1 nt/token) — requires all selected layers to share one
        resolution (the runner errors loudly otherwise).
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
