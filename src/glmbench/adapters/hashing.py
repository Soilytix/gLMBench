"""Model hash (signature) — keys every leaderboard row.

::

    model_hash = "glmb:" + sha256(
        weights_digest + "\\x00" + canonical_config_json + "\\x00"
        + adapter_name + "@" + adapter_version
    )[:24]

Same weights + same config + same adapter ⇒ same hash ⇒ idempotent leaderboard
rows (re-running replaces, never duplicates). Computing the hash must **not**
require running the model, so the leaderboard can be keyed before any GPU time.
This module is core (torch-free): stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WEIGHTS_DIGEST_STRATEGIES = frozenset({"file_sha256", "hf_revision", "declared"})

_CHUNK = 1 << 20  # 1 MiB streaming read so giant checkpoints don't load into RAM


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(path: Path) -> str:
    """Stable digest of a multi-file checkpoint dir.

    sha256 over the sorted ``<relpath>:<per-file-sha256>`` lines, so the result is
    deterministic regardless of filesystem walk order and sensitive to any file's
    contents, additions, or removals.
    """
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files under checkpoint dir {path} to hash.")
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(path).as_posix()
        h.update(f"{rel}:{_sha256_file(f)}\n".encode())
    return h.hexdigest()


def file_sha256_digest(path: str | Path, *, cache_path: str | Path | None = None) -> str:
    """sha256 over a checkpoint file, or over a checkpoint dir's per-file hashes.

    A ``cache_path`` sidecar lets a giant checkpoint be hashed once: if it exists
    its contents are returned verbatim; otherwise the digest is computed and
    written there. The cache is keyed by the caller choosing a path tied to the
    checkpoint — it is not invalidated automatically, so point it at the checkpoint
    (e.g. ``<ckpt>.sha256``) and delete it if you mutate weights in place.
    """
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        cached = cache.read_text().strip()
        if cached:
            logger.debug("weights_digest cache hit at %s", cache)
            return cached
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    digest = _sha256_dir(p) if p.is_dir() else _sha256_file(p)
    if cache is not None:
        cache.write_text(digest + "\n")
    return digest


def compute_weights_digest(
    strategy: str,
    *,
    path: str | Path | None = None,
    value: str | None = None,
    cache_path: str | Path | None = None,
) -> str:
    """Resolve a weights digest by *strategy*, without running the model.

    - ``file_sha256``: hash ``path`` (file or dir). Default for local checkpoints.
    - ``hf_revision``: trust the HuggingFace repo commit hash in ``value`` (default
      for Evo2 — avoids reading 16 GB). Returned as ``hf:<value>``.
    - ``declared``: a user-supplied digest in ``value``; logged as **untrusted**.
    """
    if strategy not in WEIGHTS_DIGEST_STRATEGIES:
        raise ValueError(
            f"Unknown weights_digest strategy {strategy!r}; "
            f"expected one of {sorted(WEIGHTS_DIGEST_STRATEGIES)}."
        )
    if strategy == "file_sha256":
        if path is None:
            raise ValueError("strategy 'file_sha256' requires a checkpoint path.")
        return f"file_sha256:{file_sha256_digest(path, cache_path=cache_path)}"
    if strategy == "hf_revision":
        if not value:
            raise ValueError("strategy 'hf_revision' requires the commit hash in 'value'.")
        return f"hf:{value}"
    # declared
    if not value:
        raise ValueError("strategy 'declared' requires the digest string in 'value'.")
    logger.warning("Using DECLARED (untrusted) weights digest %r — not weight-verified.", value)
    return f"declared:{value}"


def canonical_config_json(config: dict[str, Any]) -> str:
    """Canonical JSON for the model config dict: sorted keys, no whitespace."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def model_hash(
    *,
    weights_digest: str,
    config: dict[str, Any],
    adapter_name: str,
    adapter_version: str,
) -> str:
    """Compute the ``glmb:`` model signature."""
    payload = (
        weights_digest
        + "\x00"
        + canonical_config_json(config)
        + "\x00"
        + f"{adapter_name}@{adapter_version}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return "glmb:" + digest
