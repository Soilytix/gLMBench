"""ProkBertAdapter — ProkBERT's BERT-style masked-LM encoder (mini / mini-long / mini-c).

Wraps the ProkBERT family (`neuralbioinfo/prokbert-{mini,mini-long,mini-c}`) behind the
model-agnostic adapter contract. The heavy lifting (load the HF model + tokenizer, run
forwards) happens in :mod:`glmbench.runners.prokbert_runner`, which runs in any env with
``torch`` + ``transformers`` (the ``glmbench`` env — ProkBERT's remote code runs on
``transformers==5.12.1`` via ``trust_remote_code`` **without** the ``prokbert`` pip
package). This module is **core (torch-free)** — it only
assembles wire requests, reads the HF ``config.json`` for the model-hash arch (plain JSON,
no torch, when the checkpoint is a local dir), builds the runner backend, and validates
responses.

Copied from :class:`~glmbench.adapters.glm2.GLM2Adapter` (the suite's first masked-LM
encoder — same ``{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}`` capability shape, same
in-process ``local`` trust_remote_code serving path) and re-worked for ProkBERT's
**overlapping-k-mer LCA tokenizer**, the one thing that makes it a distinct adapter:

- **Masked-LM, bidirectional encoder** (``ProkBertForMaskedLM``). Causal scoring is
  undefined, so ``SEQUENCE_LOGLIKELIHOOD`` / ``PER_TOKEN_LOGPROBS`` / ``LOGPROB_EMBEDDING``
  are clean first-class **N/A**. The MLM head *does* serve **``MASKED_MARGINAL_LLR``** — the
  variant-effect path — but generalized to overlapping k-mers (see below).
  ``embed``/``tokenize`` load via ``AutoModel`` (→ ``ProkBertModel``: ``last_hidden_state`` +
  ``hidden_states``, no ``logits``); ``score_variant_llr`` loads ``ProkBertForMaskedLM`` via
  ``AutoModelForMaskedLM`` (for the ``logits`` head).
- **``trust_remote_code=True`` is FORCED** — ProkBERT is a custom ``prokbert`` arch with no
  in-tree impl. The remote code (``models.py`` / ``tokenizer.py``) is pulled from a
  **separate** HF repo ``neuralbioinfo/nbrg-transformers`` (the ``auto_map`` cross-repo
  ``nbrg-transformers--…`` form), so ``from_pretrained`` silently fetches *latest* code
  unless pinned: pin **both** ``revision`` (weights) **and** ``code_revision`` (the code
  repo). Do **not** ``pip install prokbert`` — it pins ``transformers<=5.3.0`` + drags
  ``torchvision`` and breaks the shared env; the remote code alone runs on 5.12.1.
- **Overlapping-k-mer LCA tokenizer** (``LCATokenizer``, ``is_fast=False``, no
  ``offset_mapping``). mini = k=6/shift=1 ⇒ a length-``L`` sequence → ``L−5`` content tokens
  (each is the 6-mer ``seq[j:j+6]``, windows sliding by 1 → heavily overlapping), wrapped
  ``[CLS] … [SEP]`` (``[CLS]``=2, ``[SEP]``=3, ``[MASK]``=4, ``[PAD]``=0, ``[UNK]``=1). So
  **nt/token = k (overlapping), NOT 1** → ``LOGPROB_EMBEDDING`` is N/A, ``token_spans`` are
  k-mer windows (overlapping — their *union* covers the sequence; a partition is impossible),
  ``per_token`` semantics are per-k-mer-token. The masked-marginal LLR generalizes ESM's
  single-token convention: a single-nt substitution alters up to ``k`` overlapping k-mer
  tokens, so the runner masks *all* affected tokens on the WT background and sums their
  mut−wt logit differences (see :meth:`score_variant_llr`).

Context: ProkBERT is trained at **1024 tokens** (``max_position_embeddings=1024``, a hard
ceiling for its ``relative_key_query`` position embeddings). Tasks chunk by ``max_context``
in **nucleotides**, so the spec sets ``max_context`` ≈ 1020 nt (mini; 2030 for mini-long's
shift=2) and the runner hard-caps tokenization at ``max_tokens`` (1024) as the loud backstop.

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

PROKBERT_RUNNER_MODULE = "glmbench.runners.prokbert_runner"
# ProkBERT's trained token window == max_position_embeddings (a hard ceiling).
DEFAULT_MAX_TOKENS = 1024
# Conservative nt bound for task-level chunking. mini (k6s1): L nt → L−3 tokens ≤ 1024 ⇒
# L ≤ 1027; 1020 keeps a margin. mini-long (k6s2) overrides to ~2030 in its spec.
DEFAULT_MAX_CONTEXT_NT = 1020

# Arch fields lifted from the HF config.json into the model-hash config (so the hash is
# sensitive to the architecture, not just the weight digest). All read torch-free. ProkBERT
# uses the standard BERT-style key names + the LCA kmer/shift.
_ARCH_KEYS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "kmer",  # LCA k-mer size
    "shift",  # LCA stride
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
    arch: dict[str, Any] = {"family": "prokbert"}
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.exists():
        return {"model": arch}
    data = json.loads(cfg_path.read_text()) or {}
    for k in _ARCH_KEYS:
        if k in data:
            arch[k] = data[k]
    return {"model": arch}


@registry.register("adapter", "prokbert")
class ProkBertAdapter(ModelAdapter):
    """Adapter for ProkBERT (custom ``ProkBertForMaskedLM`` encoder; embeddings + LLR)."""

    name = "prokbert"
    # 0.2.0: `embed` returns the RAW last-block output. ProkBERT was PREDICTED not to need
    # this (a post-LN encoder keeps its norm *inside* the block); a hook on the last block
    # measured rel_delta 3.14 / cos 0.948 and the prediction was wrong. Its encoder is
    # pre-LN, with a final LayerNorm on the encoder, outside the blocks.
    adapter_version = "0.2.0"
    CAPABILITIES = _CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "probed"
    readout_note = (
        "Pre-LN BERT encoder with a final encoder LayerNorm outside the blocks: "
        "hidden_states[-1] measured rel_delta 3.14 / cos 0.948 away from the last block's "
        "raw output, so the runner hooks the block. The 'post-LN encoders need no change' "
        "rule of thumb does NOT hold here."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        tokenizer: str | None = None,
        revision: str | None = None,
        code_revision: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        pool_include_special: bool = False,
        max_context: int | None = None,
        n_layers: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        batch_size: int = 16,
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
        self.code_revision = code_revision
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = bool(trust_remote_code)
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
                    "ProkBertAdapter backend 'docker' needs an explicit runner.image "
                    "(a container with torch + transformers); there is no default "
                    "image. The default backend is 'local'."
                )
            self._backend = DockerRunner(
                image=image,
                runner_module=PROKBERT_RUNNER_MODULE,
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
                runner_module=PROKBERT_RUNNER_MODULE,
                scratch_dir=scratch_dir,
                keep_scratch=keep_scratch,
                dry_run=dry_run,
            )
        else:
            raise ValueError(
                f"ProkBertAdapter backend must be 'local' or 'docker', got {backend!r}."
            )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> ProkBertAdapter:
        """Build a ProkBertAdapter from a :class:`ModelSpec`.

        ``spec.model`` carries ``checkpoint`` (a HF repo id like
        ``neuralbioinfo/prokbert-mini`` or a local download dir), optional ``tokenizer``
        (default = checkpoint), ``revision`` (pin the weights commit), ``code_revision``
        (pin the ``nbrg-transformers`` code repo — the second of the two pins), ``dtype``,
        ``attn_implementation``, ``trust_remote_code`` (default ``True`` — forced for the
        custom arch), ``pool_include_special``, ``max_context`` (nt), ``max_tokens`` (token
        ceiling), ``batch_size``. ``spec.runner`` selects the backend (``local`` default —
        the runner env's ``python_exe``) and ``gpus`` (→ device).
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "ProkBertAdapter spec.model must set 'checkpoint' (a HF repo id like "
                "'neuralbioinfo/prokbert-mini', or a local download dir)."
            )
        extra = spec.runner.extra
        max_context = model.get("max_context")
        return cls(
            checkpoint=str(checkpoint),
            tokenizer=model.get("tokenizer"),
            revision=model.get("revision"),
            code_revision=model.get("code_revision"),
            dtype=str(model.get("dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation"),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            pool_include_special=bool(model.get("pool_include_special", False)),
            max_context=int(max_context) if max_context is not None else None,
            n_layers=model.get("n_layers"),
            max_tokens=int(model.get("max_tokens", DEFAULT_MAX_TOKENS)),
            batch_size=int(extra.get("batch_size", model.get("batch_size", 16))),
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
            "code_revision": self.code_revision,
            "backend": self.backend,
            "image": self.image if self.backend == "docker" else None,
            "python_exe": self.python_exe if self.backend == "local" else None,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
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
                "masked-LM bidirectional encoder (ProkBertForMaskedLM). Embeddings + "
                "masked-marginal LLR variant scoring (MASKED_MARGINAL_LLR). "
                "SEQUENCE_LOGLIKELIHOOD / PER_TOKEN_LOGPROBS / LOGPROB_EMBEDDING are N/A "
                "(causal likelihood undefined for an MLM) — clean first-class N/A; "
                "variant-effect tasks fall back to the masked-marginal LLR path."
            ),
            "tokenizer_kind": (
                "LCATokenizer — OVERLAPPING k-mer (Local Context-Aware). mini k=6/shift=1 ⇒ "
                "L nt → L−5 content 6-mer tokens (windows slide by 1, heavily overlapping), "
                "wrapped [CLS]…[SEP] ([CLS]=2,[SEP]=3,[MASK]=4,[PAD]=0,[UNK]=1). is_fast=False "
                "(no offset_mapping); non-ACGT windows → [UNK]. nt/token = k (NOT 1) ⇒ "
                "logprob_embedding N/A; per-token is per-k-mer-token."
            ),
            "embedding_real_tokens": (
                "embedding: pool_include_special=False (default) pools over content (k-mer) "
                "tokens only, EXCLUDING [CLS]/[SEP]; True includes them. pool='cls' reads the "
                "[CLS] position. Layers via output_hidden_states (n_layers+1 = 7 states; the "
                "runner resolves layers against the actual tuple length). pool='none' returns "
                "per-content-token vectors + token_spans that are k-mer windows (overlapping — "
                "their union covers len(seq); a partition is impossible for overlapping k-mers)."
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
            "code_revision": self.code_revision,
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
        """Token ids of the **model input** — the LCA k-mer ids incl. ``[CLS]``/``[SEP]``.

        These are the ids ProkBERT actually consumes: ``[CLS]`` then one id per overlapping
        k-mer window then ``[SEP]`` (capped at ``max_tokens``).
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
        """Masked-marginal LLR per variant — the MLM variant-effect path, k-mer generalized.

        Each item carries ``{"reference": <WT DNA str>, "mutations": [[pos0, wt, mut], ...]}``
        (0-based positions). Because ProkBERT's LCA tokenizer is **overlapping** (a single-nt
        substitution at position ``p`` alters up to ``k`` overlapping k-mer tokens, and the
        vocab is k-mers, not nucleotides), the runner generalizes ESM's single-token
        convention: for each mutated ``p`` it masks *all* content tokens whose k-mer window
        covers ``p`` on the WT background, runs one forward, and sums per affected token ``j``
        the difference ``logit[mut_kmer_j] − logit[wt_kmer_j]`` (the mutant/WT k-mer ids come
        straight from re-tokenizing the mutated reference — no vocab-string reconstruction).
        Single-site LLR = Σ over affected tokens; multi-site = additive across sites (each
        masked independently on the WT background). Any k-mer mapping to ``[UNK]`` is
        skipped+counted. Returns ``Σ`` per variant, 1:1 with *items*, in input order. One
        masked forward per unique **(reference, masked position)**, cached across variants.
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
        tuple length — ProkBERT returns ``n_layers+1`` = 7 states, the standard BERT layout):
        ``'last'`` → final state; ``'all'`` → every state; an int / list selects indices
        (negatives ok).

        Pooling (``pool_include_special`` set on the adapter — default ``False``):
        ``mean``/``max``/``last`` reduce the token axis over the **content (k-mer)** tokens,
        excluding ``[CLS]``/``[SEP]`` → ``[len(layers), dim]`` per sequence; ``True`` includes
        the specials. ``cls`` reads the ``[CLS]`` position. ``none`` returns the full
        per-content-token tensor ``[n_tokens, len(layers), dim]`` plus ``token_spans`` that
        are k-mer windows in original-string coordinates (overlapping — their union covers
        ``len(seq)``; ``[CLS]``/``[SEP]`` excluded).
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
