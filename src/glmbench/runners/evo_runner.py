"""Evo 1.5 runner — the plain torch+transformers+flash_attn path (StripedHyena, byte-level).

Invoked as ``<env-python> -m glmbench.runners.evo_runner <request.json>`` in an env with
``torch`` + ``transformers`` + **``flash_attn``** (the ``glmbench`` env; build flash-attn
from source where no prebuilt wheel fits). It loads Evo 1.5
(``evo-design/evo-1.5-8k-base``) via ``AutoModelForCausalLM`` with ``trust_remote_code=True``
(the StripedHyena arch is a *custom* remote arch, cross-repo code in
``togethercomputer/evo-1-131k-base`` — pin ``revision`` + ``code_revision``) and implements:

- ``tokenize`` — the byte mapping (``id = clamp(ord(char), 32, 511)``) reimplemented directly
  (the ByteTokenizer ``auto_map`` is inconsistent across the two repos, so we avoid
  ``AutoTokenizer``). 1 char/token.
- ``score_sequences`` / ``per_token_logprobs`` / ``logprob_embedding`` — Evo's own scoring
  convention (``evo/scoring.py``): prepend ``eod`` (id 0) as causal context, standard causal
  shift, per-token log-prob of each realized nucleotide, ``mean`` (default) or ``sum``. The
  prefix makes every nucleotide a target, so ``logprob_embedding`` is length ``len(seq)`` and
  ``mean(z) == score(mean)``. Right-padding is safe: StripedHyena is fully causal
  (causal Hyena convolutions + causal attention), so trailing pads never affect real positions.
- ``embed`` — per-layer activations via forward **hooks** (the HF wrapper returns
  ``hidden_states=None``): layer 0 = ``backbone.embedding_layer`` output; layer ``i`` =
  ``backbone.blocks[i-1]`` output. Any/all layers in ONE forward pass. Pooling
  ``mean``/``max``/``last``/``none`` over the real nucleotide positions (no prefix prepended);
  ``none`` returns per-nt vectors + unit-width ``token_spans``.

This is a ``runners/*`` module — the one place allowed to import heavy deps.
The core ``EvoAdapter`` stays torch-free and only shells out here over the wire.
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
    write_json_payload,
    write_ragged_npz,
)
from glmbench.runners._batching import bucketed_batches
from glmbench.runners._progress import BatchProgress

# Evo's ByteTokenizer clamps ord() into this range (vocab 512); irrelevant for ASCII DNA but
# kept for byte-for-byte fidelity with the vendor tokenizer.
_MIN_BYTE = 32
_MAX_BYTE = 511


def _resolve_device(requested: str) -> Any:
    import torch

    if requested in ("auto", None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve_dtype(name: str | None, device: Any) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name in (None, "", "auto"):
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name not in mapping:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def _byte_tokenize(seq: str, max_tokens: int | None) -> list[int]:
    """Evo ByteTokenizer: one clamped ord() per character (1 char/token). Optional token cap."""
    ids = [min(max(ord(c), _MIN_BYTE), _MAX_BYTE) for c in seq]
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return ids


def _load(req: Request) -> tuple[Any, Any]:
    """Build the Evo StripedHyena model (trust_remote_code) and move to device/dtype. Eval."""
    import torch  # noqa: F401  (ensure torch present before transformers import)
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    params = req.params
    device = _resolve_device(str(params.get("device", "auto")))
    dtype = _resolve_dtype(params.get("dtype"), device)
    model_dir = str(params["checkpoint"])
    revision = params.get("revision")
    code_revision = params.get("code_revision")
    trust = bool(params.get("trust_remote_code", True))

    cfg_kwargs: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        cfg_kwargs["revision"] = revision
    if code_revision:
        cfg_kwargs["code_revision"] = code_revision
    config = AutoConfig.from_pretrained(model_dir, **cfg_kwargs)
    config.use_cache = False

    # transformers >=5 renamed the from_pretrained dtype kwarg (torch_dtype → dtype).
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    load_kwargs: dict[str, Any] = {"config": config, dtype_key: dtype, "trust_remote_code": trust}
    if revision:
        load_kwargs["revision"] = revision
    if code_revision:
        load_kwargs["code_revision"] = code_revision
    model: Any = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.to(device)
    model.eval()
    return model, device


def _prefix_id(score_prefix: str, eod_token_id: int) -> int | None:
    """The token id Evo prepends as causal context (``evo/scoring.py`` prepends ``eod_id``)."""
    if score_prefix == "none":
        return None
    if score_prefix in ("eos", "eod", "bos"):
        # Evo's ByteTokenizer has no distinct BOS — prepend_bos actually prepends eod (id 0).
        return int(eod_token_id)
    raise ValueError(f"score_prefix must be 'eos'|'eod'|'bos'|'none', got {score_prefix!r}.")


def _forward_logits(model: Any, input_ids: Any) -> Any:
    """Run the StripedHyena forward and return logits ``[B, S, vocab]`` (return_dict path)."""
    import torch

    with torch.inference_mode():
        out = model(input_ids=input_ids)
    return out.logits if hasattr(out, "logits") else out[0]


def _per_token(
    model: Any,
    seqs: list[str],
    *,
    device: Any,
    score_prefix: str,
    eod_token_id: int,
    max_tokens: int | None,
    batch_size: int,
    token_budget: int | None = None,
) -> list[np.ndarray]:
    """Per-nucleotide conditional log-prob vector per sequence (the shared score body).

    Prepend ``eod`` as causal context, run the forward, causal-shift, gather each realized
    nucleotide's log-prob, drop the prefix + pads. 1 char/token ⇒ the returned vector has
    length == len(tokenized seq); with no truncation that is ``len(seq)`` nt.
    """
    import torch
    import torch.nn.functional as F

    prefix_id = _prefix_id(score_prefix, eod_token_id)
    pad_id = int(eod_token_id)  # any valid id works: right-pad is causally invisible to reals
    token_seqs = [_byte_tokenize(s, max_tokens) for s in seqs]

    out: list[np.ndarray | None] = [None] * len(seqs)
    lengths = [len(s) for s in token_seqs]
    prog = BatchProgress(len(token_seqs), "evo/logprob", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [token_seqs[i] for i in idx_batch]
        real_lens = [len(s) for s in chunk]
        prefix = 1 if prefix_id is not None else 0
        max_full = max((n + prefix) for n in real_lens) if chunk else prefix
        max_full = max(max_full, prefix + 1)
        input_ids = torch.full((len(chunk), max_full), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros((len(chunk), max_full), dtype=torch.bool, device=device)
        for i, s in enumerate(chunk):
            col = 0
            if prefix_id is not None:
                input_ids[i, 0] = prefix_id
                col = 1
            if s:
                input_ids[i, col : col + len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                valid[i, col : col + len(s)] = True  # real tokens only (prefix/pads excluded)
        logits = _forward_logits(model, input_ids)
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
        valid_targets = valid[:, 1:]  # a position is a real target iff its token is real
        tok_cpu = tok_lp.double().detach().cpu().numpy()  # one D2H per batch, not one per row
        valid_cpu = valid_targets.detach().cpu().numpy()
        for r, i in enumerate(idx_batch):
            z = tok_cpu[r][valid_cpu[r]]
            if z.shape[0] != real_lens[r]:
                raise ValueError(
                    f"evo per-token produced {z.shape[0]} values for a {real_lens[r]}-token "
                    f"sequence (expected 1:1, 1 char/token)."
                )
            if not np.isfinite(z).all():
                raise ValueError("evo per-token produced non-finite values — refusing to emit.")
            out[i] = z  # keyed by INPUT index (bucketing reorders the batches)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert all(z is not None for z in out), "bucketing dropped a sequence"
    return out  # type: ignore[return-value]


def _score(vectors: list[np.ndarray], reduction: str) -> np.ndarray:
    """Reduce per-token vectors to per-sequence scores (mean|sum over real nucleotides)."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"evo score reduction must be 'mean'|'sum', got {reduction!r}.")
    scores = np.empty(len(vectors), dtype=np.float64)
    for i, z in enumerate(vectors):
        if z.size == 0:
            scores[i] = 0.0
        elif reduction == "mean":
            scores[i] = float(z.mean())
        else:
            scores[i] = float(z.sum())
    if np.isnan(scores).any():
        raise ValueError("evo score produced NaN — refusing to emit.")
    return scores


def _resolve_layers(layers_req: str | list[int] | None, n_blocks: int) -> list[int]:
    """Resolve ``last``/``all``/int/list to concrete layer indices (0..n_blocks).

    Convention: 0 = embedding-layer output; i = block i output (1..n_blocks). Negatives index
    from the end.
    """
    all_layers = list(range(n_blocks + 1))
    if layers_req in ("last", None):
        return [n_blocks]
    if layers_req == "all":
        return all_layers
    if isinstance(layers_req, int):
        return [all_layers[layers_req]]
    if isinstance(layers_req, (list, tuple)) and layers_req:
        return [all_layers[int(i)] for i in layers_req]
    raise ValueError(
        f"layers must be 'last', 'all', an int, or a non-empty list of ints; got {layers_req!r}."
    )


def _embed(
    model: Any,
    seqs: list[str],
    *,
    device: Any,
    layers_req: str | list[int] | None,
    pool: str,
    max_tokens: int | None,
    batch_size: int,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Hidden-state embeddings via forward HOOKS, all layers from one pass.

    Evo's HF wrapper returns ``hidden_states=None``, so we hook ``backbone.embedding_layer``
    (layer 0) and each ``backbone.blocks[j]`` (layer j+1) to capture activations. No prefix is
    prepended (embeddings represent the real nucleotides); right-padding is causally invisible.
    """
    import torch

    if pool not in ("mean", "max", "last", "none"):
        raise ValueError(f"evo embed pool must be one of mean|max|last|none; got {pool!r}.")
    backbone = model.backbone
    blocks = backbone.blocks
    n_blocks = len(blocks)
    sel = _resolve_layers(layers_req, n_blocks)
    emb_dim = int(model.config.hidden_size)

    def _tensor_of(obj: Any) -> Any:
        return obj[0] if isinstance(obj, tuple) else obj

    payload: dict[str, np.ndarray] = {}
    pooled: list[np.ndarray | None] = [None] * len(seqs)
    # Tokenize up front so batches can be bucketed by real token length.
    all_token_seqs = [_byte_tokenize(s, max_tokens) for s in seqs]
    lengths = [len(t) for t in all_token_seqs]

    prog = BatchProgress(len(seqs), "evo/embed", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk_seqs = [seqs[i] for i in idx_batch]
        token_seqs = [all_token_seqs[i] for i in idx_batch]
        real_lens = [len(t) for t in token_seqs]
        max_full = max(real_lens) if token_seqs else 1
        max_full = max(max_full, 1)
        input_ids = torch.zeros((len(chunk_seqs), max_full), dtype=torch.long, device=device)
        for i, t in enumerate(token_seqs):
            if t:
                input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        captured: dict[int, Any] = {}
        handles = []
        need = set(sel)

        def _mk(li: int, store: dict[int, Any] = captured):
            # store bound as a default arg so the closure captures THIS batch's dict (B023).
            return lambda m, inp, out: store.__setitem__(li, _tensor_of(out).detach())

        # Layer 0 = embedding output. StripedHyena applies it via embedding_layer.embed()
        # (not __call__), so a forward hook never fires — instead grab it as the INPUT to
        # blocks[0] via a forward-PRE-hook (that input IS the post-embedding tensor).
        if 0 in need:
            handles.append(
                blocks[0].register_forward_pre_hook(
                    lambda m, args, store=captured: store.__setitem__(0, _tensor_of(args).detach())
                )
            )
        for layer_idx in need:
            if layer_idx == 0:
                continue
            j = layer_idx - 1  # block index
            handles.append(blocks[j].register_forward_hook(_mk(layer_idx)))
        try:
            with torch.inference_mode():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        missing = [li for li in sel if li not in captured]
        if missing:
            raise ValueError(f"evo embed: forward hooks did not fire for layers {missing}.")
        # [B, S, n_sel, H]
        stacked = torch.stack([captured[li] for li in sel], dim=2)
        for r, i in enumerate(idx_batch):
            n_real = real_lens[r]
            real = stacked[r, :n_real]  # [n_real, n_sel, H]
            if pool == "none":
                payload[f"arr_{i}"] = real.float().cpu().numpy().astype(np.float32)
                spans = [(k, k + 1) for k in range(len(chunk_seqs[r]))][:n_real]
                payload[f"span_{i}"] = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
            elif pool == "mean":
                pooled[i] = real.float().mean(dim=0).cpu().numpy().astype(np.float32)
            elif pool == "max":
                pooled[i] = real.float().amax(dim=0).cpu().numpy().astype(np.float32)
            elif pool == "last":
                pooled[i] = real[-1].float().cpu().numpy().astype(np.float32)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()

    payload["layer_ids"] = np.asarray(sel, dtype=np.int64)
    payload["embedding_dim"] = np.asarray(emb_dim, dtype=np.int64)
    if pool == "none":
        payload["n"] = np.asarray(len(seqs), dtype=np.int64)
    else:
        assert all(p is not None for p in pooled), "bucketing dropped a sequence"
        payload["arrays"] = (
            np.stack(pooled, axis=0)  # in INPUT order
            if pooled
            else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
        )
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"evo embed produced non-finite values in {key} — refusing.")
    return payload


def run(request_path: str) -> None:
    req = Request.read(request_path)
    valid_ops = ("tokenize", "score_sequences", "per_token_logprobs", "logprob_embedding", "embed")
    if req.op not in valid_ops:
        raise ValueError(
            f"evo runner implements {valid_ops}, got {req.op!r}. (EvoAdapter declares TOKENIZE + "
            "SEQUENCE_LOGLIKELIHOOD + PER_TOKEN_LOGPROBS + EMBEDDING + LOGPROB_EMBEDDING.)"
        )
    _ids, seqs = read_sequences_jsonl(req.input_path)
    params = req.params
    max_tokens = params.get("max_tokens")
    max_tokens = int(max_tokens) if max_tokens is not None else None
    scratch_dir = os.path.dirname(request_path)
    meta: dict[str, object] = {"n": len(seqs), "runner": "evo"}

    if req.op == "tokenize":
        tokens = [_byte_tokenize(s, max_tokens) for s in seqs]
        write_json_payload(req.output_path, tokens)
        Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
            os.path.join(scratch_dir, "response.json")
        )
        return

    model, device = _load(req)
    meta["device"] = str(device)
    batch_size = int(params.get("batch_size", 4))
    token_budget = params.get("token_budget")
    token_budget = int(token_budget) if token_budget else None

    if req.op == "embed":
        payload = _embed(
            model,
            seqs,
            device=device,
            layers_req=params.get("layers", "last"),
            pool=str(params.get("pool", "mean")),
            max_tokens=max_tokens,
            batch_size=batch_size,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, payload)
        meta["pool"] = str(params.get("pool", "mean"))
        Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
            os.path.join(scratch_dir, "response.json")
        )
        return

    vectors = _per_token(
        model,
        seqs,
        device=device,
        score_prefix=str(params.get("score_prefix", "eos")),
        eod_token_id=int(params.get("eod_token_id", 0)),
        max_tokens=max_tokens,
        batch_size=batch_size,
        token_budget=token_budget,
    )
    if req.op == "score_sequences":
        scores = _score(vectors, str(params.get("reduction", "mean")))
        write_arrays_npz(req.output_path, {"scores": scores})
        meta["reduction"] = str(params.get("reduction", "mean"))
    else:  # per_token_logprobs | logprob_embedding (same per-nt vectors)
        write_ragged_npz(req.output_path, vectors)

    Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
        os.path.join(scratch_dir, "response.json")
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.evo_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
