"""Evo2-Arc runner — the ArcInstitute ``evo2`` package path.

Invoked as ``<evo2-env-python> -m glmbench.runners.evo2_arc_runner <request.json>`` in the
conda env that has ``evo2`` + ``vtx`` + ``flash-attn``. It loads Evo2 once via
``Evo2(model_name)`` (at the pinned Hub revision) and implements three ops, all from the
**same** forward:

- ``embed`` — named-layer activations captured with ``return_embeddings=True,
  layer_names=[...]``. Under the hood that registers a forward **hook** on
  ``model.get_submodule(name)``, so the full stack still runs and the LM head is untouched;
  nothing is truncated. Any number of names come back in one pass.
- ``score_sequences`` — per-sequence causal log-likelihood from the full-depth logits.
- ``logprob_embedding`` — the same per-token log-probs, un-collapsed (1 value per nt).

**Why the reduction is hand-rolled rather than delegated to ``Evo2.score_sequences``.**
The package's ``_score_sequences`` reduces with ``logprobs[idx][:seq_lengths[idx]]`` over a
right-padded batch. With ``prepend_bos=False`` the shifted log-prob row is one shorter than
the sequence, so for every non-longest member of a mixed-length batch that slice runs past
the real tokens and **averages in predictions of PAD**. Our corpora are mixed-length by
nature (genes run 188 → 8192 nt), so we mask explicitly instead. The arithmetic is otherwise
identical to ``evo2.scoring.logits_to_logprobs``: ``log_softmax`` over the vocab, drop the
last position, gather the realized next token.

**``prepend_bos`` is a misnomer.** The package prepends ``tokenizer.eod_id``, not a distinct
BOS::

    ([tokenizer.eod_id] * int(prepend_bos)) + tokenizer.tokenize(seq) + padding

And for Evo2's ``CharLevelTokenizer`` ``eod_id == eos_id == 0``, so ``score_prefix='bos'``
and ``'eos'`` select the *same token id* and cannot disagree. Only ``none`` vs.
prepend-something is a real choice.

**No attention mask, and none needed.** ``StripedHyena.forward`` takes ``input_ids`` alone —
there is no mask argument. That is safe with right-padding because every mixer in the stack
(Hyena convolutions and attention alike) is causal, so a real token at position *i* never
reads a pad at *j > i*. Pads are excluded from the reduction, not from the forward.

This is a ``runners/*`` module — the one place allowed to import heavy deps;
:class:`~glmbench.adapters.evo2_arc.Evo2ArcAdapter` stays torch-free.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from glmbench.adapters.wire import (
    Request,
    Response,
    read_sequences_jsonl,
    write_arrays_npz,
    write_ragged_npz,
)
from glmbench.runners._batching import bucketed_batches
from glmbench.runners._progress import BatchProgress


def _resolve_device(requested: str) -> Any:
    import torch

    if requested in ("auto", None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _pinned_weights(model_name: str, revision: str) -> str:
    """Local path of *model_name*'s weight file at exactly Hub commit *revision*.

    ``Evo2(model_name)`` alone downloads whatever the Hub repo's ``main`` is today (and
    prefers a previously merged ``<model>.pt`` in the cache over the Hub), so it cannot
    promise the weights the model hash names. We resolve the pinned commit here and hand the
    file to ``Evo2(model_name, local_path=...)``, which builds the same architecture from the
    package's bundled config and loads the same checkpoint format. Sharded repos are merged
    once, into a file keyed by the revision, the way the package merges them.
    """
    import glob

    from evo2.utils import HF_MODEL_NAME_MAP
    from huggingface_hub import constants, snapshot_download

    repo_dir = snapshot_download(repo_id=HF_MODEL_NAME_MAP[model_name], revision=revision)
    filename = f"{model_name}.pt"
    path = os.path.join(repo_dir, filename)
    if os.path.exists(path):
        return path
    parts = sorted(
        glob.glob(os.path.join(repo_dir, f"{filename}.part*")),
        key=lambda p: int(p.rsplit(".part", 1)[1]),
    )
    if not parts:
        raise FileNotFoundError(
            f"{HF_MODEL_NAME_MAP[model_name]}@{revision} has neither {filename} nor its shards."
        )
    merged = os.path.join(
        os.path.dirname(constants.HF_HUB_CACHE), f"{model_name}@{revision}.pt"
    )
    if not os.path.exists(merged):
        tmp = merged + ".partial"
        with open(tmp, "wb") as out:
            for part in parts:
                with open(part, "rb") as src:
                    while chunk := src.read(8192 * 1024):
                        out.write(chunk)
        os.replace(tmp, merged)
    return merged


def _load(req: Request) -> tuple[Any, Any, Any]:
    """Load Evo2 via the Arc package. Returns ``(evo2, tokenizer, device)``.

    With a ``revision`` (the spec's ``hf_revision`` weights digest) the weights are exactly
    that Hub commit (:func:`_pinned_weights`); without one, the package's own download.

    Vortex places the model on CUDA itself (and will shard across every visible GPU), so we
    deliberately do **not** call ``.to(device)`` — the package's docstring warns against it
    for multi-GPU. Pin the run to one card with ``CUDA_VISIBLE_DEVICES`` in the caller's
    environment instead.
    """
    import torch  # noqa: F401  (import torch before the CUDA extensions vtx pulls in)

    params = req.params
    hf_home = params.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)

    from evo2 import Evo2

    model_name = str(params["model_name"])
    device = _resolve_device(str(params.get("device", "auto")))
    if device.type != "cuda":
        raise RuntimeError(
            f"evo2-arc requires a CUDA device (Evo2/vortex has no CPU path), got {device}. "
            f"Check CUDA_VISIBLE_DEVICES and runner.gpus in the spec."
        )
    revision = params.get("revision")
    if revision:
        evo2 = Evo2(model_name, local_path=_pinned_weights(model_name, str(revision)))
    else:
        evo2 = Evo2(model_name)
    return evo2, evo2.tokenizer, device


def _tokenize(tokenizer: Any, seq: str, max_context: int) -> list[int]:
    """Byte-level token ids for one sequence, truncated to ``max_context``.

    Evo2's ``CharLevelTokenizer`` is ``np.frombuffer(text.encode('utf-8'), np.uint8)`` — one
    token per byte, hence one token per nucleotide for ASCII DNA. That 1:1 relation is what
    makes ``logprob_embedding`` per-*nucleotide* rather than per-token.
    """
    ids = [int(x) for x in tokenizer.tokenize(seq)]
    return ids[:max_context] if max_context and len(ids) > max_context else ids


def _prefix_id(tokenizer: Any, score_prefix: str) -> int | None:
    """The token id prepended as causal context, masked out of the loss.

    ``bos`` mirrors the package's ``prepend_bos=True`` (which prepends ``eod_id``); ``eos``
    names the same token explicitly. For ``CharLevelTokenizer`` these are both id 0 — see
    the module docstring.
    """
    if score_prefix == "none":
        return None
    if score_prefix == "bos":
        return int(getattr(tokenizer, "eod_id", 0))
    if score_prefix == "eos":
        return int(getattr(tokenizer, "eos_id", getattr(tokenizer, "eod_id", 0)))
    raise ValueError(f"score_prefix must be 'bos'|'eos'|'none', got {score_prefix!r}.")


def _build_batch(
    chunk: list[list[int]], *, prefix_id: int | None, pad_id: int, device: Any
) -> tuple[Any, Any]:
    """Right-padded ``[B, L]`` ids + a bool ``[B, L]`` mask of real (non-pad) positions."""
    import torch

    prefix = 1 if prefix_id is not None else 0
    real_lens = [len(s) for s in chunk]
    width = max((n + prefix) for n in real_lens) if chunk else prefix + 1
    width = max(width, prefix + 1)
    input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
    real = torch.zeros((len(chunk), width), dtype=torch.bool, device=device)
    for i, s in enumerate(chunk):
        col = 0
        if prefix_id is not None:
            input_ids[i, 0] = prefix_id
            real[i, 0] = True  # the prefix is real context, but never a scored target
            col = 1
        if s:
            input_ids[i, col : col + len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            real[i, col : col + len(s)] = True
    return input_ids, real


def _token_logprobs(evo2: Any, input_ids: Any) -> Any:
    """``[B, L-1]`` log-prob of each realized next token (``evo2.scoring`` arithmetic)."""
    import torch

    with torch.inference_mode():
        logits, _ = evo2.model.forward(input_ids)
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    return log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)


def _score(
    evo2: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    reduction: str,
    score_prefix: str,
    max_context: int,
    batch_size: int,
    token_budget: int | None = None,
) -> np.ndarray:
    """Per-sequence causal log-likelihood (mean|sum over real next-token log-probs)."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"evo2-arc score reduction must be 'mean'|'sum', got {reduction!r}.")
    prefix_id = _prefix_id(tokenizer, score_prefix)
    pad_id = int(getattr(tokenizer, "pad_id", 1))
    token_seqs = [_tokenize(tokenizer, s, max_context) for s in seqs]

    out = np.full(len(seqs), np.nan, dtype=np.float64)
    lengths = [len(s) for s in token_seqs]
    prog = BatchProgress(len(token_seqs), "evo2_arc/score", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
        chunk = [token_seqs[i] for i in idx_batch]
        input_ids, real = _build_batch(
            chunk, prefix_id=prefix_id, pad_id=pad_id, device=device
        )
        tok_lp = _token_logprobs(evo2, input_ids)  # [B, L-1]
        valid = real[:, 1:]  # the prefix is never a target; pads are excluded
        counts = valid.sum(dim=1)
        sums = tok_lp.masked_fill(~valid, 0.0).sum(dim=1)
        vals = sums / counts.clamp(min=1) if reduction == "mean" else sums
        vals_cpu = vals.double().detach().cpu().numpy()
        empty_cpu = (counts == 0).detach().cpu().numpy()
        for r, i in enumerate(idx_batch):
            out[i] = 0.0 if empty_cpu[r] else float(vals_cpu[r])
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    if np.isnan(out).any():
        raise ValueError("evo2-arc score produced NaN — refusing to emit.")
    return out


def _logprob_embedding(
    evo2: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    score_prefix: str,
    max_context: int,
    batch_size: int,
    token_budget: int | None = None,
) -> list[np.ndarray]:
    """Per-nucleotide conditional log-prob vector ``z`` per sequence.

    The reduction-free body of :func:`_score` — same prefix, same gather, no collapse. With
    a byte-level tokenizer 1 token ⇄ 1 nt, so ``len(z) == len(seq)`` (or ``max_context``),
    and ``mean(z) == _score(seq, 'mean')`` by construction.

    Note the first nucleotide only gets a log-prob when a prefix is prepended (nothing
    predicts token 0 otherwise) — with ``score_prefix='none'`` ``z`` is one shorter, which
    the adapter's length check would reject. That is intentional: the shipped spec prepends.
    """
    prefix_id = _prefix_id(tokenizer, score_prefix)
    pad_id = int(getattr(tokenizer, "pad_id", 1))
    token_seqs = [_tokenize(tokenizer, s, max_context) for s in seqs]

    out: list[np.ndarray | None] = [None] * len(seqs)
    lengths = [len(s) for s in token_seqs]
    prog = BatchProgress(len(token_seqs), "evo2_arc/logprob", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
        chunk = [token_seqs[i] for i in idx_batch]
        input_ids, real = _build_batch(
            chunk, prefix_id=prefix_id, pad_id=pad_id, device=device
        )
        tok_lp = _token_logprobs(evo2, input_ids)
        valid = real[:, 1:]
        tok_cpu = tok_lp.double().detach().cpu().numpy()  # one D2H per batch
        valid_cpu = valid.detach().cpu().numpy()
        for r, i in enumerate(idx_batch):
            z = tok_cpu[r][valid_cpu[r]]
            if not np.isfinite(z).all():
                raise ValueError("evo2-arc logprob_embedding produced non-finite values.")
            out[i] = z
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert all(z is not None for z in out), "bucketing dropped a sequence"
    return out  # type: ignore[return-value]


def _embed(
    evo2: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    layer_ids: list[int],
    layer_names: list[str],
    pool: str,
    max_context: int,
    batch_size: int,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Named-layer embeddings, every requested layer from ONE forward pass.

    No prefix token is prepended: the embedding must represent the real sequence.
    Pooling is over the valid (non-pad) positions only.
    """
    import torch

    if pool not in ("mean", "max"):
        raise ValueError(f"evo2-arc embed pool must be 'mean'|'max'; got {pool!r}.")
    if len(layer_ids) != len(layer_names):
        raise ValueError(
            f"evo2-arc embed: layer_ids ({layer_ids}) and layer_names ({layer_names}) must "
            f"be the same length — they are the same list in two representations."
        )
    # Fail loudly and early on a bad layer name, before burning a forward pass on the whole
    # corpus. get_submodule raises AttributeError for an unknown path; a wrong-but-valid
    # name would otherwise silently return a different tensor.
    for name in layer_names:
        try:
            evo2.model.get_submodule(name)
        except AttributeError as e:
            raise ValueError(
                f"evo2-arc: layer name {name!r} does not exist in this model. Layer names "
                f"are arch-specific (evo2_7b* → 'blocks.28.mlp.l3', evo2_1b_base → "
                f"'blocks.24.mlp.l3'). Inspect evo2.model.state_dict().keys() for valid "
                f"paths."
            ) from e

    pad_id = int(getattr(tokenizer, "pad_id", 1))
    token_seqs = [_tokenize(tokenizer, s, max_context) for s in seqs]
    lengths = [len(s) for s in token_seqs]

    pooled: list[np.ndarray | None] = [None] * len(token_seqs)
    emb_dim = 0
    prog = BatchProgress(len(token_seqs), "evo2_arc/embed", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
        chunk = [token_seqs[i] for i in idx_batch]
        real_lens = [len(s) for s in chunk]
        input_ids, _real = _build_batch(chunk, prefix_id=None, pad_id=pad_id, device=device)
        with torch.inference_mode():
            _logits, emb = evo2.forward(
                input_ids, return_embeddings=True, layer_names=list(layer_names)
            )
        missing = [n for n in layer_names if n not in emb]
        if missing:
            raise ValueError(
                f"evo2-arc embed: hook produced no tensor for {missing} — refusing to "
                f"guess a substitute layer."
            )
        # [B, L, n_sel, H] in the REQUESTED order, so the layer axis matches layer_ids.
        stacked = torch.stack([emb[n] for n in layer_names], dim=2)
        emb_dim = int(stacked.shape[-1])
        for r, i in enumerate(idx_batch):
            n_real = real_lens[r]
            if n_real == 0:
                pooled[i] = np.zeros((len(layer_ids), emb_dim), dtype=np.float32)
                continue
            valid = stacked[r, :n_real]  # [n_real, n_sel, H] — pads excluded
            reduced = valid.float().mean(dim=0) if pool == "mean" else valid.float().amax(dim=0)
            pooled[i] = reduced.detach().cpu().numpy().astype(np.float32)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert all(p is not None for p in pooled), "bucketing dropped a sequence"

    payload: dict[str, np.ndarray] = {
        "layer_ids": np.asarray(layer_ids, dtype=np.int64),
        "embedding_dim": np.asarray(emb_dim, dtype=np.int64),
        "arrays": (
            np.stack([p for p in pooled if p is not None], axis=0)
            if pooled
            else np.zeros((0, len(layer_ids), emb_dim), dtype=np.float32)
        ),
    }
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"evo2-arc embed produced non-finite values in {key} — refusing.")
    return payload


def run(request_path: str) -> None:
    req = Request.read(request_path)
    if req.op not in ("score_sequences", "embed", "logprob_embedding"):
        raise ValueError(
            f"evo2-arc runner implements 'score_sequences', 'embed', and "
            f"'logprob_embedding', got {req.op!r}."
        )
    _ids, seqs = read_sequences_jsonl(req.input_path)
    evo2, tokenizer, device = _load(req)
    params = req.params
    batch_size = int(params.get("batch_size", 1))
    token_budget = params.get("token_budget")
    token_budget = int(token_budget) if token_budget else None
    max_context = int(params.get("max_context", 8192))
    scratch_dir = os.path.dirname(request_path)
    meta: dict[str, object] = {
        "n": len(seqs),
        "runner": "evo2_arc",
        "device": str(device),
        "model_name": str(params.get("model_name")),
    }

    if req.op == "logprob_embedding":
        z = _logprob_embedding(
            evo2,
            tokenizer,
            seqs,
            device=device,
            score_prefix=str(params.get("score_prefix", "bos")),
            max_context=max_context,
            batch_size=batch_size,
            token_budget=token_budget,
        )
        write_ragged_npz(req.output_path, z)
    elif req.op == "score_sequences":
        scores = _score(
            evo2,
            tokenizer,
            seqs,
            device=device,
            reduction=str(params.get("reduction", "mean")),
            score_prefix=str(params.get("score_prefix", "bos")),
            max_context=max_context,
            batch_size=batch_size,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, {"scores": scores})
        meta["reduction"] = str(params.get("reduction", "mean"))
    else:  # embed
        layer_ids = [int(x) for x in params.get("layers", [0])]
        layer_names = [str(x) for x in params["layer_names"]]
        payload = _embed(
            evo2,
            tokenizer,
            seqs,
            device=device,
            layer_ids=layer_ids,
            layer_names=layer_names,
            pool=str(params.get("pool", "mean")),
            max_context=max_context,
            batch_size=batch_size,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, payload)
        meta["pool"] = str(params.get("pool", "mean"))
        meta["layer_names"] = layer_names

    Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
        os.path.join(scratch_dir, "response.json")
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.evo2_arc_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
