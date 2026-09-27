"""LOAMHFAdapter — LOAM models in Hugging Face format (``Soilytix/LOAM-*``).

Serves a LOAM checkpoint exported to a Hugging Face model directory (``model_type: "loam"``,
custom modeling code loaded with ``trust_remote_code``) through the same adapter contract as
every other model. ``model.checkpoint`` is either

* a local export directory, e.g. a download of ``Soilytix/LOAM-25M``, or
* a Hub repo id, e.g. ``Soilytix/LOAM-25M`` (optionally with ``model.revision``). It is
  downloaded once with ``huggingface_hub.snapshot_download``, hashed from that download, and
  loaded by repo id at the downloaded commit. The model hash covers the weight bytes only, so it
  is identical whichever way the weights arrived.

**The LOAM serving conventions are fixed here, not exposed as knobs**, because each one has a
wrong value that yields a plausible, wrong number:

* A BOS token is prepended to every input. It is context only: never scored, never pooled.
* The usable context is ``max_position_embeddings - 1`` nucleotides, because BOS takes a slot.
  Tasks window their inputs to this bound.
* Embedding layer ``i`` is the **raw output of block i** (layer 0 = the token embeddings),
  captured with forward hooks. The model's final RMSNorm is **not** applied to embeddings: the
  LOAM residual stream carries a few very large dimensions, and a per-token RMSNorm divides
  every informative dimension by them. Log-probabilities *do* go through the final norm,
  because that is the model's own forward.

Numerics follow the native LOAM implementation the manuscript's benchmark rows were scored
with, so the two agree bit for bit on the same inputs: bf16 compute on GPU (fp32 on CPU),
length-bucketed right-padded batches under a causal mask only (a real token never attends to
a pad), mean pooling in the compute dtype, log-softmax in fp32.

This module is core (torch-free). The model runs in :mod:`glmbench.runners.loam_hf_runner`,
in an environment with ``torch`` + ``transformers``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from glmbench import registry

if TYPE_CHECKING:
    from glmbench.config.model_spec import ModelSpec

from .base import Capability, CapabilityNotImplemented, EmbeddingResult, ModelAdapter, Readout
from .hashing import compute_weights_digest, model_hash
from .runner_backend import LocalSubprocessRunner
from .wire import WireProtocolError

LOAM_HF_RUNNER_MODULE = "glmbench.runners.loam_hf_runner"

#: ``config.json`` fields that define the trained function. They enter the model hash, so two
#: exports of the same weights under a different architecture can never share a row.
_ARCH_KEYS: tuple[str, ...] = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "norm_placement",
    "qk_norm",
    "gpas_enabled",
    "rope_theta",
    "rms_norm_eps",
    "tie_word_embeddings",
    "use_bias",
    "loam_tokenizer_hash",
)

#: The weight files of a LOAM export. The weights digest covers these files and nothing else,
#: so editing the model card, licence or README of a release never moves the hash.
_WEIGHT_GLOB = "*.safetensors"

_POOL_DTYPES = ("compute", "float32")

_BASE_CAPS = frozenset(
    {
        Capability.TOKENIZE,
        Capability.SEQUENCE_LOGLIKELIHOOD,
        Capability.PER_TOKEN_LOGPROBS,
        Capability.EMBEDDING,
    }
)


def _looks_like_hub_id(value: str) -> bool:
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not value.startswith((".", "/", "~"))


def _resolve_checkpoint(
    checkpoint: str, *, revision: str | None, spec_dir: Path | None
) -> tuple[Path, tuple[str, str] | None]:
    """Local export dir for *checkpoint* (as given, relative to the spec, or from the Hub), and
    for a Hub id the ``(repo id, commit)`` it was downloaded at, else ``None``."""
    p = Path(checkpoint).expanduser()
    if p.is_dir():
        return p.resolve(), None
    if not p.is_absolute() and spec_dir is not None and (spec_dir / p).is_dir():
        return (spec_dir / p).resolve(), None
    if _looks_like_hub_id(checkpoint):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise ImportError(
                f"model.checkpoint {checkpoint!r} is not a local directory, so it is treated "
                "as a Hugging Face Hub repo id, which needs `huggingface_hub` "
                "(pip install huggingface_hub). Or download the repo and point "
                "model.checkpoint at the directory."
            ) from exc
        snapshot = Path(snapshot_download(repo_id=checkpoint, revision=revision))
        # A Hub-cache snapshot is named by its commit and holds symlinks into blobs/.
        return snapshot.resolve(), (checkpoint, snapshot.name)
    raise FileNotFoundError(
        f"LOAM-HF checkpoint {checkpoint!r} is neither a local directory nor a Hub repo id "
        "of the form 'org/name'."
    )


def _read_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{model_dir} has no config.json — not a Hugging Face export.")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_type") != "loam":
        raise ValueError(
            f"{model_dir}/config.json has model_type={cfg.get('model_type')!r}; the loam-hf "
            "adapter serves LOAM exports only (model_type 'loam')."
        )
    return cfg


def _weight_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob(_WEIGHT_GLOB))
    if not files:
        raise FileNotFoundError(f"No {_WEIGHT_GLOB} weight files in {model_dir}.")
    return files


@registry.register("adapter", "loam-hf")
class LOAMHFAdapter(ModelAdapter):
    """Adapter for LOAM checkpoints in Hugging Face format."""

    name = "loam-hf"
    adapter_version = "1.0.0"
    CAPABILITIES = _BASE_CAPS
    readout = Readout.BLOCK_OUTPUT
    readout_evidence = "by_construction"
    readout_note = (
        "loam_hf_runner registers forward hooks on the token embedding and on every block of "
        "model.model.layers and returns those outputs; the final RMSNorm is not applied to "
        "embeddings (the log-prob head still goes through it — that is the model's forward)."
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        revision: str | None = None,
        python_exe: str | None = None,
        batch_size: int = 8,
        token_budget: int | None = 16384,
        dtype: str = "auto",
        pool_dtype: str = "compute",
        device: str = "auto",
        weights_digest_strategy: str = "file_sha256",
        weights_digest_value: str | None = None,
        weights_digest_cache: str | None = None,
        spec_dir: str | Path | None = None,
        scratch_dir: str | None = None,
        keep_scratch: bool = False,
        dry_run: bool = False,
    ) -> None:
        if pool_dtype not in _POOL_DTYPES:
            raise ValueError(f"pool_dtype must be one of {_POOL_DTYPES}, got {pool_dtype!r}")
        self.checkpoint = checkpoint
        self.revision = revision
        self.model_dir, self._hub = _resolve_checkpoint(
            checkpoint, revision=revision, spec_dir=Path(spec_dir) if spec_dir else None
        )
        self.batch_size = int(batch_size)
        # Cap on batch × padded length (tokens). It bounds the real tensor, so it bounds both
        # compute and memory, and it adapts on its own: short inputs get large batches, long
        # inputs small ones. 16384 is what the manuscript's LOAM rows were scored with.
        self.token_budget = int(token_budget) if token_budget else None
        self.dtype = dtype
        self.pool_dtype = pool_dtype
        self.device = device
        self._weights_digest_strategy = weights_digest_strategy
        self._weights_digest_value = weights_digest_value
        self._weights_digest_cache = weights_digest_cache
        self._model_hash: str | None = None

        self._config = _read_config(self.model_dir)
        self.embedding_dim = int(self._config.get("hidden_size", 0))
        self.n_layers = int(self._config.get("num_hidden_layers", 0))
        max_pos = int(self._config.get("max_position_embeddings", 0))
        # One position is the BOS. Without this, an input of exactly max_position_embeddings
        # nucleotides becomes max_position_embeddings + 1 positions and the model refuses it.
        self.max_context = max(0, max_pos - 1)
        # 1 token = 1 nucleotide only for the nucleotide tokenizer, and only then is a
        # per-token log-prob a per-nucleotide one. The LOAM release is nucleotide-level.
        self.tokenizer_name = self._config.get("loam_tokenizer_name") or _tokenizer_name(
            self.model_dir
        )
        self._is_single_nt = self.tokenizer_name == "nucleotide"

        self._backend = LocalSubprocessRunner(
            python_exe=python_exe or sys.executable,
            runner_module=LOAM_HF_RUNNER_MODULE,
            scratch_dir=scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, keep_scratch: bool = False, dry_run: bool = False
    ) -> LOAMHFAdapter:
        """Build from a :class:`ModelSpec`.

        ``spec.model``: ``checkpoint`` (required; local export dir or Hub repo id), and
        optionally ``revision``, ``batch_size``, ``token_budget``, ``dtype``, ``pool_dtype``,
        ``weights_digest_cache``. ``spec.runner.python_exe`` defaults to the current
        interpreter, i.e. one environment holding both ``glmbench`` and ``torch`` +
        ``transformers``.
        """
        model = spec.model or {}
        checkpoint = model.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "loam-hf spec.model.checkpoint must name a LOAM Hugging Face export: a local "
                "directory (e.g. a download of Soilytix/LOAM-25M) or a Hub repo id (e.g. "
                "Soilytix/LOAM-25M)."
            )
        extra = spec.runner.extra or {}
        token_budget = extra.get("token_budget", model.get("token_budget", 16384))
        return cls(
            checkpoint=str(checkpoint),
            revision=model.get("revision"),
            python_exe=spec.runner.python_exe,
            batch_size=int(extra.get("batch_size", model.get("batch_size", 8))),
            token_budget=int(token_budget) if token_budget else None,
            dtype=str(extra.get("dtype", model.get("dtype", "auto"))),
            pool_dtype=str(model.get("pool_dtype", "compute")),
            device=_device_from_gpus(spec.runner.gpus),
            weights_digest_strategy=spec.weights_digest.strategy,
            weights_digest_value=spec.weights_digest.value,
            weights_digest_cache=model.get("weights_digest_cache"),
            spec_dir=Path(spec.source_path).parent if spec.source_path else None,
            scratch_dir=spec.runner.scratch_dir,
            keep_scratch=keep_scratch,
            dry_run=dry_run,
        )

    # --- metadata ----------------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        caps = set(type(self).CAPABILITIES)
        if self._is_single_nt:
            caps.add(Capability.LOGPROB_EMBEDDING)
        return frozenset(caps)

    def _hashed_config(self) -> dict[str, Any]:
        arch = {"family": "loam"}
        arch.update({k: self._config[k] for k in _ARCH_KEYS if k in self._config})
        # Serving choices that change the numbers (not batching, which only moves float noise
        # at the last bits). A run in fp32, or with fp32 pooling, is a different row.
        return {"model": arch, "serving": {"dtype": self.dtype, "pool_dtype": self.pool_dtype}}

    def _weights_digest(self) -> str:
        if self._weights_digest_strategy != "file_sha256":
            return compute_weights_digest(
                self._weights_digest_strategy,
                value=self._weights_digest_value,
                cache_path=self._weights_digest_cache,
            )
        files = _weight_files(self.model_dir)
        # The optional sidecar cache applies to the usual single-file export only.
        cache = self._weights_digest_cache if len(files) == 1 else None
        digests = {
            f.name: compute_weights_digest("file_sha256", path=f, cache_path=cache).split(":", 1)[1]
            for f in files
        }
        _check_against_export_record(self.model_dir, digests)
        if len(digests) == 1:
            return f"file_sha256:{next(iter(digests.values()))}"
        # Sharded weights: one digest over the sorted `<file>:<sha256>` lines.
        joined = "".join(f"{name}:{d}\n" for name, d in sorted(digests.items()))
        return f"file_sha256:{_sha256_text(joined)}"

    def model_hash(self) -> str:
        """Content hash: weight bytes + architecture + serving dtype + adapter@version.

        Memoized (hashing a 2.5 GB weight file on every call adds up). Only the weight files
        are hashed, so the same weights give the same hash from a local export, a Hub download
        or a re-download.
        """
        if self._model_hash is None:
            self._model_hash = model_hash(
                weights_digest=self._weights_digest(),
                config=self._hashed_config(),
                adapter_name=self.name,
                adapter_version=self.adapter_version,
            )
        return self._model_hash

    def describe(self) -> dict[str, Any]:
        export = _read_export_record(self.model_dir)
        return {
            "name": self.name,
            "adapter_version": self.adapter_version,
            "capabilities": sorted(c.value for c in self.capabilities()),
            **self.readout_declaration(),
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "hub_commit": self._hub[1] if self._hub else None,
            "source_checkpoint_id": export.get("checkpoint_id"),
            "parameter_count": export.get("parameter_count"),
            "embedding_dim": self.embedding_dim,
            "n_layers": self.n_layers,
            "max_context": self.max_context,
            "tokenizer": self.tokenizer_name,
            "device": self.device,
            "dtype": self.dtype,
            "pool_dtype": self.pool_dtype,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "real_tokens": "all non-BOS, non-pad positions (BOS prepended as context)",
        }

    # --- request plumbing --------------------------------------------------

    def _common_params(self) -> dict[str, Any]:
        # A Hub download is loaded by repo id at the exact commit that was hashed, not by its
        # cache path: transformers resolves remote-code files through the cache's symlinks and
        # then looks for their relative imports among the blobs, where they do not exist.
        repo, commit = self._hub or (str(self.model_dir), None)
        return {
            "checkpoint": repo,
            "revision": commit,
            "batch_size": self.batch_size,
            "token_budget": self.token_budget,
            "dtype": self.dtype,
            "pool_dtype": self.pool_dtype,
            "device": self.device,
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
        result = self._backend.execute(
            "per_token_logprobs", self._common_params(), sequences, output_name="output.npz"
        )
        n = int(result.payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"per_token_logprobs returned {n} arrays for {len(sequences)} inputs."
            )
        return [result.payload[f"arr_{i}"] for i in range(n)]

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        """Per-nucleotide conditional log-prob ``log P(n_i | BOS, n_1..n_{i-1})`` per input."""
        if not self._is_single_nt:
            raise CapabilityNotImplemented(Capability.LOGPROB_EMBEDDING, self.name)
        result = self._backend.execute(
            "logprob_embedding", self._common_params(), sequences, output_name="output.npz"
        )
        n = int(result.payload["n"])
        if n != len(sequences):
            raise WireProtocolError(
                f"logprob_embedding returned {n} arrays for {len(sequences)} inputs."
            )
        return [_checked_z(result.payload[f"arr_{i}"], seq, i) for i, seq in enumerate(sequences)]

    def embed(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> EmbeddingResult:
        """Hidden states per requested layer, from one forward pass per batch.

        ``layers``: ``"last"`` (the raw output of the final block), ``"all"`` (``0..L``, where
        0 is the token-embedding output), or explicit indices into that axis (negatives
        allowed). ``pool``: ``mean``/``max`` over the real tokens, ``bos``, ``last``/``eos``
        (final real token), or ``none`` (per-token arrays plus nucleotide spans).
        """
        params = self._common_params()
        params["layers"] = layers
        params["pool"] = pool
        result = self._backend.execute("embed", params, sequences, output_name="output.npz")
        emb, _ = _parse_embed_payload(result.payload, pool)
        return emb

    def embed_with_logprob(
        self, sequences: list[str], *, layers: list[int] | str = "last", pool: str = "mean"
    ) -> tuple[EmbeddingResult, list[np.ndarray]]:
        """Embeddings and per-nucleotide log-probs from **one** forward pass.

        Used automatically when one run needs both from the same inputs; each output equals
        what :meth:`embed` and :meth:`logprob_embedding` return on their own.
        """
        if not self._is_single_nt:
            raise CapabilityNotImplemented(Capability.LOGPROB_EMBEDDING, self.name)
        params = self._common_params()
        params["layers"] = layers
        params["pool"] = pool
        result = self._backend.execute(
            "embed_logprob", params, sequences, output_name="output.npz"
        )
        emb, payload = _parse_embed_payload(result.payload, pool)
        n_z = int(payload["n_z"])
        if n_z != len(sequences):
            raise WireProtocolError(
                f"embed_logprob returned {n_z} log-prob vectors for {len(sequences)} inputs."
            )
        zs = [_checked_z(payload[f"z_{i}"], seq, i) for i, seq in enumerate(sequences)]
        return emb, zs


# --- helpers ------------------------------------------------------------------


def _parse_embed_payload(
    payload: dict[str, np.ndarray], pool: str
) -> tuple[EmbeddingResult, dict[str, np.ndarray]]:
    layer_ids = [int(x) for x in payload["layer_ids"]]
    emb_dim = int(payload["embedding_dim"])
    if pool == "none":
        n = int(payload["n"])
        arrays = [payload[f"arr_{i}"] for i in range(n)]
        spans: list[list[tuple[int, int]]] | None = [
            [(int(a), int(b)) for a, b in payload[f"span_{i}"]] for i in range(n)
        ]
    else:
        stacked = payload["arrays"]
        arrays = [stacked[i] for i in range(stacked.shape[0])]
        spans = None
    emb = EmbeddingResult(
        arrays=arrays, layers=layer_ids, pool=pool, token_spans=spans, embedding_dim=emb_dim
    )
    return emb, payload


def _checked_z(arr: np.ndarray, seq: str, i: int) -> np.ndarray:
    z = np.asarray(arr, dtype=np.float64)
    if z.shape != (len(seq),):
        raise WireProtocolError(
            f"log-prob vector {i} has shape {z.shape}, expected ({len(seq)},): one log-prob "
            "per nucleotide."
        )
    if not np.isfinite(z).all():
        raise WireProtocolError("non-finite per-nucleotide log-probs — refusing to poison tasks.")
    return z


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _read_export_record(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "loam_export.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _check_against_export_record(model_dir: Path, digests: dict[str, str]) -> None:
    """Refuse weights whose sha256 disagrees with the export's own record of them.

    ``loam_export.json`` is written by the exporter next to the weights. A mismatch means the
    directory holds different weights from the ones that were exported (a partial download, a
    stale copy); scoring them under this export's name would be wrong in silence.
    """
    recorded = _read_export_record(model_dir).get("weight_sha256") or {}
    for name, digest in digests.items():
        want = recorded.get(name)
        if want and want != digest:
            raise ValueError(
                f"{model_dir}/{name} has sha256 {digest}, but loam_export.json records {want}. "
                "The weights on disk are not the exported ones; re-download the model."
            )


def _tokenizer_name(model_dir: Path) -> str | None:
    p = model_dir / "tokenizer_config.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("loam_tokenizer_name")


def _device_from_gpus(gpus: str | None) -> str:
    """``runner.gpus`` → device: empty / ``cpu`` / ``none`` / ``-1`` → CPU, else CUDA if present.

    The physical GPU is chosen with ``CUDA_VISIBLE_DEVICES``.
    """
    if gpus is None or str(gpus).strip() in ("", "-1", "none", "cpu"):
        return "cpu"
    return "auto"
