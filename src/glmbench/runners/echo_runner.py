"""Echo runner — a model-free test double that exercises the wire boundary.

Invoked as ``python -m glmbench.runners.echo_runner <request.json>``. It reads the
wire request, computes **deterministic** fake outputs from the sequence bytes (no
model, no torch), and writes the payload + a Response. This lets the entire
adapter spine — capability gating, wire round trip, scratch handling, runner-error
surfacing — be tested in CI with no real model.

Determinism: every output is a pure function of the input string, so a run is
reproducible and a task built on the echo adapter has a stable answer.
"""

from __future__ import annotations

import sys

import numpy as np

from glmbench.adapters.wire import (
    Request,
    Response,
    read_sequences_jsonl,
    read_variant_jsonl,
    write_arrays_npz,
    write_json_payload,
    write_ragged_npz,
)

# A small fixed "model" geometry the echo embeddings advertise.
ECHO_EMBED_DIM = 8
ECHO_N_LAYERS = 2


def _seed(seq: str) -> int:
    """A stable integer fingerprint of a sequence (deterministic across runs)."""
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    for ch in seq.encode("utf-8"):
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _score(seq: str, reduction: str) -> float:
    """A deterministic pseudo-log-likelihood: mean/sum of per-byte fractions."""
    if not seq:
        return 0.0
    per_tok = [-(((b * 2654435761) & 0xFFFF) / 0xFFFF) for b in seq.encode("utf-8")]
    total = float(sum(per_tok))
    return total / len(per_tok) if reduction == "mean" else total


def _per_token(seq: str) -> np.ndarray:
    return np.array(
        [-(((b * 2654435761) & 0xFFFF) / 0xFFFF) for b in seq.encode("utf-8")],
        dtype=np.float64,
    )


def _fake_logit(reference: str, pos0: int, base: str) -> float:
    """A deterministic pseudo-logit for ``base`` at masked position ``pos0`` of ``reference``.

    Depends only on the masked context (reference + position) and the candidate base, so —
    like a real masked-marginal head — the WT and MUT terms at one position share the same
    context and the softmax normalizer cancels in the difference.
    """
    h = _seed(f"{reference}|{pos0}|{base.lower()}")
    return -((h & 0xFFFF) / 0xFFFF)


def _variant_llr(reference: str, mutations: list[list]) -> float:
    """Deterministic masked-marginal LLR: Σ_i (logit_mut_i − logit_wt_i)."""
    total = 0.0
    for pos0, wt, mut in mutations:
        total += _fake_logit(reference, int(pos0), str(mut)) - _fake_logit(
            reference, int(pos0), str(wt)
        )
    return total


def _embed_pooled(seq: str, n_layers: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(seq))
    return rng.standard_normal((n_layers, ECHO_EMBED_DIM)).astype(np.float64)


def _embed_none(seq: str, n_layers: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n_tok = len(seq)
    rng = np.random.default_rng(_seed(seq))
    arr = rng.standard_normal((n_tok, n_layers, ECHO_EMBED_DIM)).astype(np.float64)
    spans = [(i, i + 1) for i in range(n_tok)]  # echo is 1 bp/token
    return arr, spans


def run(request_path: str) -> None:
    req = Request.read(request_path)
    op = req.op

    # The variant-effect op carries reference + mutation records, not {"id","seq"}.
    if op == "score_variant_llr":
        _ids, items = read_variant_jsonl(req.input_path)
        scores = np.array(
            [_variant_llr(it["reference"], it["mutations"]) for it in items],
            dtype=np.float64,
        )
        write_arrays_npz(req.output_path, {"scores": scores})
        Response(
            op=op,
            status="ok",
            output_path=req.output_path,
            meta={"n": len(items), "runner": "echo"},
        ).write(_response_path(request_path))
        return

    _ids, seqs = read_sequences_jsonl(req.input_path)
    meta: dict[str, object] = {"n": len(seqs), "runner": "echo"}

    if op == "score_sequences":
        reduction = str(req.params.get("reduction", "mean"))
        scores = np.array([_score(s, reduction) for s in seqs], dtype=np.float64)
        write_arrays_npz(req.output_path, {"scores": scores})

    elif op == "per_token_logprobs":
        write_ragged_npz(req.output_path, [_per_token(s) for s in seqs])

    elif op == "logprob_embedding":
        # Reduction-free form of _score: 1 byte/token ⇒ 1 nt/token for the echo double,
        # so per-token == per-nt and mean(z) == _score(s, "mean") exactly.
        write_ragged_npz(req.output_path, [_per_token(s) for s in seqs])

    elif op == "tokenize":
        tokens = [[int(b) for b in s.encode("utf-8")] for s in seqs]
        write_json_payload(req.output_path, tokens)

    elif op == "embed":
        layers = req.params.get("layers", "last")
        pool = str(req.params.get("pool", "mean"))
        n_layers = ECHO_N_LAYERS if layers in ("all",) else 1
        if pool == "none":
            # Ragged per-sequence arrays + per-sequence (n_tok, 2) span tables, all
            # in the one npz so they survive scratch cleanup (no sidecar files).
            payload: dict[str, np.ndarray] = {"n": np.array(len(seqs), dtype=np.int64)}
            for i, s in enumerate(seqs):
                arr, spans = _embed_none(s, n_layers)
                payload[f"arr_{i}"] = arr
                payload[f"span_{i}"] = np.array(spans, dtype=np.int64).reshape(-1, 2)
            payload["n_layers"] = np.array(n_layers, dtype=np.int64)
            write_arrays_npz(req.output_path, payload)
        else:
            arrs = np.stack([_embed_pooled(s, n_layers) for s in seqs], axis=0)
            write_arrays_npz(
                req.output_path, {"arrays": arrs, "n_layers": np.array(n_layers)}
            )
        meta["pool"] = pool
        meta["embedding_dim"] = ECHO_EMBED_DIM

    else:  # pragma: no cover - Request.from_json already rejects unknown ops
        raise ValueError(f"echo runner cannot handle op {op!r}")

    Response(op=op, status="ok", output_path=req.output_path, meta=meta).write(
        _response_path(request_path)
    )


def _response_path(request_path: str) -> str:
    # The backend reads response.json from the scratch dir (next to the request).
    import os

    return os.path.join(os.path.dirname(request_path), "response.json")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.echo_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
