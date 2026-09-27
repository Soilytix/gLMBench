"""The file-mediated wire protocol between core and an adapter's runner.

Each capability call is a batched, file-mediated round trip:

1. The adapter serializes a :class:`Request` (op + params + a path to a JSONL
   input file of ``{"id", "seq"}`` records) and invokes its runner backend.
2. The runner (running in the adapter's *own* environment) reads the request,
   runs the model, and writes a payload — **heavy tensors always via ``.npz``,
   never inline in JSON** — plus a :class:`Response` describing where it landed.
3. The adapter reads + validates the response and returns native Python objects.

Both messages are versioned (:data:`WIRE_VERSION`); a mismatch is a loud error.
This module is core (torch-free): it only touches stdlib + numpy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

WIRE_VERSION = "1.0"

# The ops a runner must understand. Mirrors the capability methods.
WIRE_OPS = frozenset(
    {
        "tokenize",
        "score_sequences",
        "per_token_logprobs",
        "embed",
        "logprob_embedding",
        "score_variant_llr",
        # One forward pass, both outputs: hidden states AND per-token log-probs. `embed` and
        # `logprob_embedding` read two different heads of the SAME forward (hidden states vs
        # logits), so running them as separate ops pushes the identical corpus through the
        # model twice. Used by ReusingAdapter when a fusion group needs both. The payload is
        # the `embed` payload plus `z_{i}` / `n_z` keys.
        "embed_logprob",
    }
)


class WireVersionError(RuntimeError):
    """Raised when a Request/Response declares a wire version we don't speak."""


class WireProtocolError(RuntimeError):
    """Raised on any other protocol violation (bad op, missing file, shape drift)."""


@dataclass
class Request:
    """A single batched op sent to a runner.

    Attributes:
        op: One of :data:`WIRE_OPS`.
        params: Op-specific keyword params (e.g. ``{"reduction": "mean"}``).
        input_path: Path to the JSONL sequence file (``{"id", "seq"}`` per line).
        output_path: Where the runner must write its payload (``.npz`` or ``.json``).
        wire_version: Protocol version; checked on read.
    """

    op: str
    params: dict[str, Any]
    input_path: str
    output_path: str
    wire_version: str = WIRE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Request:
        data = json.loads(text)
        _check_wire_version(data.get("wire_version"))
        if data.get("op") not in WIRE_OPS:
            raise WireProtocolError(
                f"Unknown wire op {data.get('op')!r}; expected one of {sorted(WIRE_OPS)}."
            )
        return cls(**data)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def read(cls, path: str | Path) -> Request:
        return cls.from_json(Path(path).read_text())


@dataclass
class Response:
    """A runner's reply describing where its payload landed.

    Status is ``"ok"`` on success, ``"error"`` on a handled failure (with
    ``error`` set), or ``"dry_run"`` for the no-execute path.
    """

    op: str
    status: str
    output_path: str | None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    wire_version: str = WIRE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Response:
        data = json.loads(text)
        _check_wire_version(data.get("wire_version"))
        return cls(**data)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def read(cls, path: str | Path) -> Response:
        return cls.from_json(Path(path).read_text())


def _check_wire_version(version: Any) -> None:
    if version != WIRE_VERSION:
        raise WireVersionError(
            f"Wire version mismatch: message declares {version!r}, this gLMBench "
            f"speaks {WIRE_VERSION!r}. Re-run both sides on the same version."
        )


# --- JSONL sequence I/O ----------------------------------------------------


def write_sequences_jsonl(
    path: str | Path, sequences: list[str], ids: list[str] | None = None
) -> None:
    """Write sequences as JSONL (``{"id", "seq"}`` per line).

    IDs default to the positional index as a string, preserving input order so
    runners can echo results back in order.
    """
    if ids is not None and len(ids) != len(sequences):
        raise WireProtocolError(
            f"ids length ({len(ids)}) must match sequences length ({len(sequences)})."
        )
    use_ids = ids if ids is not None else [str(i) for i in range(len(sequences))]
    with Path(path).open("w") as fh:
        for sid, seq in zip(use_ids, sequences, strict=True):
            fh.write(json.dumps({"id": sid, "seq": seq}) + "\n")


def read_sequences_jsonl(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a JSONL sequence file → ``(ids, sequences)`` in file order."""
    ids: list[str] = []
    seqs: list[str] = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or "seq" not in rec:
                raise WireProtocolError(
                    f"{path}:{lineno}: each record needs 'id' and 'seq' keys, got {rec!r}."
                )
            ids.append(str(rec["id"]))
            seqs.append(str(rec["seq"]))
    return ids, seqs


# --- variant JSONL I/O (masked-marginal LLR) -------------------------------
#
# Variant-effect records carry a reference (WT) sequence + a mutation list, which the
# ``{"id","seq"}`` sequence JSONL can't express. One JSON object per line:
# ``{"id", "reference", "mutations"}`` where ``mutations`` is a list of
# ``[pos0, "WT_base", "MUT_base"]`` (0-based positions). Heavy output stays a single 1-D
# ``scores`` array via :func:`write_arrays_npz` (the ``score_sequences`` return shape).


def write_variant_jsonl(
    path: str | Path,
    items: list[dict[str, Any]],
    ids: list[str] | None = None,
) -> None:
    """Write variant records as JSONL (``{"id", "reference", "mutations"}`` per line).

    Each item must carry ``reference`` (WT DNA str) and ``mutations`` (a list of
    ``[pos0, wt_base, mut_base]`` triples). IDs default to the positional index as a
    string, preserving input order so runners can echo results back in order.
    """
    if ids is not None and len(ids) != len(items):
        raise WireProtocolError(
            f"ids length ({len(ids)}) must match items length ({len(items)})."
        )
    use_ids = ids if ids is not None else [str(i) for i in range(len(items))]
    with Path(path).open("w") as fh:
        for vid, item in zip(use_ids, items, strict=True):
            if "reference" not in item or "mutations" not in item:
                raise WireProtocolError(
                    f"variant record needs 'reference' and 'mutations' keys, got {item!r}."
                )
            rec = {
                "id": vid,
                "reference": str(item["reference"]),
                "mutations": [list(m) for m in item["mutations"]],
            }
            fh.write(json.dumps(rec) + "\n")


def read_variant_jsonl(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a variant JSONL file → ``(ids, items)`` in file order.

    Each item is ``{"reference", "mutations"}`` with ``mutations`` a list of
    ``[pos0, wt_base, mut_base]`` triples (positions coerced to ``int``).
    """
    ids: list[str] = []
    items: list[dict[str, Any]] = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or "reference" not in rec or "mutations" not in rec:
                raise WireProtocolError(
                    f"{path}:{lineno}: each record needs 'id', 'reference', 'mutations' "
                    f"keys, got {rec!r}."
                )
            mutations = [
                [int(m[0]), str(m[1]), str(m[2])] for m in rec["mutations"]
            ]
            ids.append(str(rec["id"]))
            items.append({"reference": str(rec["reference"]), "mutations": mutations})
    return ids, items


# --- npz array I/O ---------------------------------------------------------
#
# Heavy tensors always travel via .npz. Scores are a single 1-D ``scores`` array;
# ragged per-sequence arrays (per_token_logprobs, pool='none' embeddings) are
# stored one key per sequence (``arr_0``, ``arr_1``, …) plus an ``n`` count, since
# np.savez cannot stack variable-length arrays.


def write_arrays_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Save a flat mapping of name → ndarray to an ``.npz`` file."""
    np.savez(str(path), **arrays)  # type: ignore[arg-type]  # numpy stub: **kwds vs allow_pickle


def read_arrays_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array from an ``.npz`` file into a dict (copies into memory)."""
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def write_ragged_npz(path: str | Path, arrays: list[np.ndarray]) -> None:
    """Save a list of variable-length arrays as ``arr_0..arr_{n-1}`` + ``n``."""
    payload: dict[str, np.ndarray] = {"n": np.array(len(arrays), dtype=np.int64)}
    for i, arr in enumerate(arrays):
        payload[f"arr_{i}"] = np.asarray(arr)
    write_arrays_npz(path, payload)


def read_ragged_npz(path: str | Path) -> list[np.ndarray]:
    """Inverse of :func:`write_ragged_npz` — restore the ordered list of arrays."""
    data = read_arrays_npz(path)
    n = int(data["n"])
    out: list[np.ndarray] = []
    for i in range(n):
        key = f"arr_{i}"
        if key not in data:
            raise WireProtocolError(f"ragged npz missing expected array '{key}' in {path}.")
        out.append(data[key])
    return out


# --- JSON payload I/O (tokenize) -------------------------------------------


def write_json_payload(path: str | Path, obj: Any) -> None:
    """Write a small JSON payload (e.g. token-id lists). Not for heavy tensors."""
    Path(path).write_text(json.dumps(obj))


def read_json_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
