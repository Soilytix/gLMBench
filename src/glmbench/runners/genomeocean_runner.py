"""GenomeOcean runner — the plain torch+transformers path (BPE Mistral).

Invoked as ``<env-python> -m glmbench.runners.genomeocean_runner <request.json>`` in any
env with ``torch`` + ``transformers`` (the ``glmbench`` env works after
``pip install -e ".[hf]"``). It loads a GenomeOcean checkpoint
(``pGenomeOcean/GenomeOcean-{100M,500M}``) as a **stock** ``MistralForCausalLM`` via
``AutoModelForCausalLM``/``AutoTokenizer`` and implements four ops:

- ``tokenize`` — the BPE token ids of each sequence (no special tokens).
- ``score_sequences`` — per-sequence causal log-likelihood (mean|sum over the real BPE
  tokens; ``[CLS]`` supplies BOS context, ``[SEP]``/``[PAD]`` are masked from the loss).
- ``per_token_logprobs`` — one conditional log-prob per **real BPE token** (not per nt).
- ``embed`` — hidden states from one ``output_hidden_states=True`` pass, pooled.
  Loaded head-less (``AutoModel``): the ops that read logits (``score_sequences`` /
  ``per_token_logprobs``) pay for the ``AutoModelForCausalLM`` head, the ones that don't
  (``embed`` / ``tokenize``) don't.

All batched ops length-bucket their batches (:mod:`glmbench.runners._batching`) and scatter
results back to **input** order by index — padding waste, not sequence order, is what shrinks.
A ``token_budget`` param (optional) bounds ``batch × padded_len`` instead of just row count.

**Why stock load (no ``trust_remote_code``):** the repos ship a bundled
``modeling_mistral.py`` (``auto_map``) written against ``transformers==4.38.2`` that calls
the long-removed ``DynamicCache.from_legacy_cache`` and so **crashes on transformers ≥5**.
The architecture is plain ``MistralForCausalLM`` (``model_type: mistral``), so we load it
with the in-tree Mistral implementation (``trust_remote_code=False``), which is
version-robust and numerically the model. ``trust_remote_code`` stays a knob (default
``False``) for anyone who pins transformers 4.38 and wants the original code path.

**BPE specifics (vs a 1-char/token causal-LM runner):** the tokenizer
is a DNABERT-2-style BPE (`PreTrainedTokenizerFast`, vocab 4096) with **variable
nucleotides per token** and BERT-style specials used as BOS/EOS — every sequence is wrapped
``[CLS] … [SEP]`` (`[CLS]`=1, `[SEP]`=2, `[PAD]`=3, `[UNK]`=0). Real tokens are
``special_tokens_mask == 0``; ``token_spans`` come from the fast tokenizer's offset mapping
(specials map to ``(0, 0)``). Scoring uses **right-padding internally** so per-token gather
is padding-invariant regardless of the tokenizer's ``padding_side``.

This is a ``runners/*`` module — the one place allowed to import heavy deps.
The core ``GenomeOceanAdapter`` stays torch-free and only shells out here over the wire.
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
)
from glmbench.runners._batching import bucketed_batches
from glmbench.runners._progress import BatchProgress
from glmbench.runners._readout import forward_hidden_states


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
        # bf16 on GPU (the model's native dtype), fp32 on CPU (bf16 CPU matmul is slow/uneven).
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name not in mapping:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def _load(req: Request, *, for_logits: bool = False) -> tuple[Any, Any, Any]:
    """Build the HF model + tokenizer and move to device/dtype. Eval mode.

    Loads as a **stock** Mistral (``trust_remote_code=False`` by default; see the module
    docstring for why the bundled code is avoided). ``revision`` pins the HF commit for
    reproducibility.

    Model class depends on the op — pay for the LM head only when its logits are read:
    ``score_sequences``/``per_token_logprobs`` set ``for_logits=True`` and get the full
    ``MistralForCausalLM``; ``embed``/``tokenize`` get the bare ``AutoModel``
    (``MistralModel``, no ``lm_head``). GenomeOcean's BPE vocab is 4096, so the head the
    embed path used to run was a real [B, S, 4096] matmul nothing ever read.
    ``output_hidden_states=True`` returns the identical tuple from either class: the
    CausalLM's hidden states *are* the inner ``MistralModel``'s (final-norm included).
    """
    import torch  # noqa: F401  (ensures torch present before transformers import)
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    params = req.params
    device = _resolve_device(str(params.get("device", "auto")))
    dtype = _resolve_dtype(params.get("dtype"), device)
    model_dir = str(params["checkpoint"])
    tokenizer_dir = str(params.get("tokenizer") or model_dir)
    attn_impl = params.get("attn_implementation")  # None → transformers default (sdpa)
    trust = bool(params.get("trust_remote_code", False))
    revision = params.get("revision")  # None → default branch

    common: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        common["revision"] = str(revision)

    # transformers >=5 renamed the from_pretrained dtype kwarg (torch_dtype → dtype).
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_kwargs: dict[str, Any] = {dtype_key: dtype, **common}
    if attn_impl:
        model_kwargs["attn_implementation"] = str(attn_impl)
    loader = AutoModelForCausalLM if for_logits else AutoModel
    model: Any = loader.from_pretrained(model_dir, **model_kwargs)
    # padding_side='right' makes causal per-token gather padding-invariant (we never
    # generate, so the vendor's left-padding default is irrelevant here).
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_dir, padding_side="right", **common
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _encode(tokenizer: Any, seqs: list[str], *, max_tokens: int) -> dict[str, Any]:
    """Batch-encode with specials, padding (right), truncation, special mask + offsets.

    Returns a dict of torch tensors: ``input_ids``/``attention_mask``/``special_tokens_mask``
    each ``[B, S]`` and ``offset_mapping`` ``[B, S, 2]`` (half-open nt spans; specials →
    ``(0, 0)``). Every sequence is wrapped ``[CLS] … [SEP]``.
    """
    enc = tokenizer(
        list(seqs),
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=int(max_tokens),
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    return enc


def _padded_lengths(tokenizer: Any, seqs: list[str], *, max_tokens: int) -> list[int]:
    """Full tokenized width of each sequence (``[CLS] … [SEP]``, truncated) — the bucket key.

    One extra (cheap, Rust, unpadded) tokenizer pass over the corpus buys the length sort
    that removes the padding waste from every forward. The width returned here is exactly
    the row width :func:`_encode` will build for that sequence when it is alone in a batch,
    which is what ``bucketed_batches`` costs its budget against.
    """
    if not seqs:
        return []
    enc = tokenizer(
        list(seqs),
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=int(max_tokens),
    )
    return [len(ids) for ids in enc["input_ids"]]


def _real_token_ids(tokenizer: Any, seq: str, *, max_tokens: int) -> list[int]:
    """The BPE token ids of one sequence, WITHOUT special tokens (for ``tokenize``)."""
    return list(
        tokenizer(
            seq, add_special_tokens=False, truncation=True, max_length=int(max_tokens)
        )["input_ids"]
    )


def _resolve_layers(layers_req: str | list[int] | None, n_blocks: int) -> list[int]:
    """Resolve ``last``/``all``/int/list to concrete hidden_states indices (0..n_blocks)."""
    all_layers = list(range(n_blocks + 1))  # 0 = embedding output; i = block i
    if layers_req in ("last", None):
        return [n_blocks]
    if layers_req == "all":
        return all_layers
    if isinstance(layers_req, int):
        return [all_layers[layers_req]]
    if isinstance(layers_req, (list, tuple)) and layers_req:
        return [all_layers[int(i)] for i in layers_req]  # negatives index from the end
    raise ValueError(
        f"layers must be 'last', 'all', an int, or a non-empty list of ints; got {layers_req!r}."
    )


def _token_logprobs_batch(
    model: Any, enc: dict[str, Any], device: Any
) -> tuple[Any, Any]:
    """Per-token causal log-prob of each realized next token + the real-target mask.

    Returns ``(tok_lp, real_targets)`` both ``[B, S-1]``: ``tok_lp[b, j]`` =
    ``log P(token_{j+1} | tokens_{<=j})`` and ``real_targets[b, j]`` is True iff position
    ``j+1`` is a **real** BPE token (``special_tokens_mask==0`` and attended). ``[CLS]`` at
    position 0 is never a target; ``[SEP]``/``[PAD]`` targets are masked out.
    """
    import torch.nn.functional as F

    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    special = enc["special_tokens_mask"].to(device)
    import torch

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
    real_targets = (special[:, 1:] == 0) & (attn[:, 1:].bool())
    return tok_lp, real_targets


def _score(
    model: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    reduction: str,
    batch_size: int,
    max_tokens: int,
    token_budget: int | None = None,
) -> np.ndarray:
    """Per-sequence causal log-likelihood (mean|sum over real BPE-token log-probs)."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"genomeocean score reduction must be 'mean'|'sum', got {reduction!r}.")
    out = np.full(len(seqs), np.nan, dtype=np.float64)
    lengths = _padded_lengths(tokenizer, seqs, max_tokens=max_tokens)
    prog = BatchProgress(len(seqs), "genomeocean/score", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [seqs[i] for i in idx_batch]  # the BUCKETED chunk — what _encode pads
        enc = _encode(tokenizer, chunk, max_tokens=max_tokens)
        tok_lp, real = _token_logprobs_batch(model, enc, device)
        # Reduce on-device and copy ONCE per batch. The old per-row `.item()` blocked on the
        # GPU B times per batch, serializing the next batch behind each readback.
        counts = real.sum(dim=1)
        sums = tok_lp.masked_fill(~real, 0.0).sum(dim=1)
        vals = sums / counts.clamp(min=1) if reduction == "mean" else sums
        vals_cpu = vals.double().detach().cpu().numpy()
        empty_cpu = (counts == 0).detach().cpu().numpy()
        for r, i in enumerate(idx_batch):
            out[i] = 0.0 if empty_cpu[r] else float(vals_cpu[r])  # keyed by INPUT index
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    if np.isnan(out).any():
        raise ValueError("genomeocean score produced NaN — refusing to emit.")
    return out


def _per_token_logprobs(
    model: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    batch_size: int,
    max_tokens: int,
    token_budget: int | None = None,
) -> list[np.ndarray]:
    """One conditional log-prob per **real BPE token** per sequence (BPE granularity)."""
    out: list[np.ndarray | None] = [None] * len(seqs)
    lengths = _padded_lengths(tokenizer, seqs, max_tokens=max_tokens)
    prog = BatchProgress(len(seqs), "genomeocean/logprob", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [seqs[i] for i in idx_batch]  # the BUCKETED chunk — what _encode pads
        enc = _encode(tokenizer, chunk, max_tokens=max_tokens)
        tok_lp, real = _token_logprobs_batch(model, enc, device)
        tok_cpu = tok_lp.double().detach().cpu().numpy()  # one D2H per batch, not one per row
        real_cpu = real.detach().cpu().numpy()
        for r, i in enumerate(idx_batch):
            z = tok_cpu[r][real_cpu[r]]
            if not np.isfinite(z).all():
                raise ValueError("genomeocean per_token_logprobs produced non-finite values.")
            out[i] = z  # keyed by INPUT index (bucketing reorders the batches)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert all(z is not None for z in out), "bucketing dropped a sequence"
    return out  # type: ignore[return-value]


def _embed(
    model: Any,
    tokenizer: Any,
    seqs: list[str],
    *,
    device: Any,
    layers_req: str | list[int] | None,
    pool: str,
    batch_size: int,
    max_tokens: int,
    pool_include_special: bool,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Hidden-state embeddings (the layer convention), all layers from one forward pass.

    Pooling positions: ``pool_include_special=True`` (the **vendor** convention — its
    ``embedding.py`` masks only padding) pools over all non-pad tokens, ``[CLS]``/``[SEP]``
    included; ``False`` pools over the real BPE tokens only. ``pool='none'`` always returns
    the real-token vectors + their nt ``token_spans``; ``pool='bos'``/``'cls'`` reads the
    ``[CLS]`` position.
    """
    import torch

    if pool not in ("mean", "max", "last", "bos", "cls", "none"):
        raise ValueError(
            f"genomeocean embed pool must be mean|max|last|bos|cls|none; got {pool!r}."
        )
    n_blocks = int(model.config.num_hidden_layers)
    sel = _resolve_layers(layers_req, n_blocks)
    emb_dim = int(model.config.hidden_size)

    payload: dict[str, np.ndarray] = {}
    pooled: list[np.ndarray | None] = [None] * len(seqs)
    lengths = _padded_lengths(tokenizer, seqs, max_tokens=max_tokens)

    prog = BatchProgress(len(seqs), "genomeocean/embed", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [seqs[i] for i in idx_batch]  # the BUCKETED chunk — what _encode pads
        enc = _encode(tokenizer, chunk, max_tokens=max_tokens)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        special = enc["special_tokens_mask"]
        offsets = enc["offset_mapping"]  # [B, S, 2] on cpu
        # `model` here is the bare AutoModel (no LM head) — see _load(for_logits=False):
        # nothing in this function reads logits, so the [B, S, 4096] head matmul is skipped.
        #
        # `forward_hidden_states` hooks the last block and splices its RAW output over
        # hidden_states[-1]: this Mistral-style stack is pre-LN, and the two tensors
        # measured 5.4 relative / cos 0.983 apart.
        #
        # `use_cache=False`: embedding never reads a KV cache, and building one is the
        # difference between working and not under `trust_remote_code=True` — GenomeOcean's
        # bundled modeling code calls `DynamicCache.from_legacy_cache`, removed in current
        # transformers, and raises AttributeError before producing anything. The stock load
        # (this runner's default) does not hit that, but the knob exists and this costs
        # nothing.
        hs = forward_hidden_states(
            model,
            expected_depth=n_blocks,
            where="genomeocean/embed",
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
        )  # list len n_blocks+1, each [B, S, H]
        stacked = torch.stack([hs[li] for li in sel], dim=2)  # [B, S, n_sel, H]
        stacked = stacked.float().detach().cpu()  # one D2H per batch, not one per row
        attn_cpu = attn.bool().cpu()
        spec_cpu = special.bool()
        for r, i in enumerate(idx_batch):  # r = row in this batch, i = INPUT index
            attn_i = attn_cpu[r]
            real_i = attn_i & ~spec_cpu[r]
            if pool == "none":
                rows = stacked[r][real_i]  # [n_real, n_sel, H]
                payload[f"arr_{i}"] = rows.numpy().astype(np.float32)
                payload[f"span_{i}"] = (
                    offsets[r][real_i].to(dtype=torch.int64).numpy().reshape(-1, 2)
                )
                continue
            if pool in ("bos", "cls"):
                pooled[i] = stacked[r, 0].numpy().astype(np.float32)
                continue
            mask = attn_i if pool_include_special else real_i
            if not bool(mask.any()):  # degenerate (empty/all-truncated) → fall back to all non-pad
                mask = attn_i
            sub = stacked[r][mask]  # [n positions, n_sel_layers, H]
            if pool == "mean":
                pooled[i] = sub.mean(dim=0).numpy().astype(np.float32)
            elif pool == "max":
                pooled[i] = sub.amax(dim=0).numpy().astype(np.float32)
            else:  # last
                pooled[i] = sub[-1].numpy().astype(np.float32)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    pooled_rows = [p for p in pooled if p is not None]
    if pool != "none":
        assert len(pooled_rows) == len(seqs), "bucketing dropped a sequence"

    payload["layer_ids"] = np.asarray(sel, dtype=np.int64)
    payload["embedding_dim"] = np.asarray(emb_dim, dtype=np.int64)
    if pool == "none":
        assert all(f"arr_{i}" in payload for i in range(len(seqs))), "bucketing dropped a sequence"
        payload["n"] = np.asarray(len(seqs), dtype=np.int64)
    else:
        payload["arrays"] = (
            np.stack(pooled_rows, axis=0)
            if pooled_rows
            else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
        )
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"genomeocean embed produced non-finite values in {key} — refusing.")
    return payload


def run(request_path: str) -> None:
    req = Request.read(request_path)
    if req.op not in ("tokenize", "score_sequences", "per_token_logprobs", "embed"):
        raise ValueError(
            f"genomeocean runner implements 'tokenize', 'score_sequences', "
            f"'per_token_logprobs', and 'embed', got {req.op!r}. (GenomeOceanAdapter "
            "declares TOKENIZE + SEQUENCE_LOGLIKELIHOOD + PER_TOKEN_LOGPROBS + EMBEDDING; "
            "LOGPROB_EMBEDDING is N/A for a BPE tokenizer.)"
        )
    _ids, seqs = read_sequences_jsonl(req.input_path)
    # Only the two log-prob ops read logits — everything else loads the head-less AutoModel.
    needs_logits = req.op in ("score_sequences", "per_token_logprobs")
    model, tokenizer, device = _load(req, for_logits=needs_logits)
    batch_size = int(req.params.get("batch_size", 8))
    max_tokens = int(req.params.get("max_tokens", 1024))
    token_budget = req.params.get("token_budget")
    token_budget = int(token_budget) if token_budget else None
    scratch_dir = os.path.dirname(request_path)
    meta: dict[str, object] = {"n": len(seqs), "runner": "genomeocean", "device": str(device)}

    if req.op == "tokenize":
        tokens = [_real_token_ids(tokenizer, s, max_tokens=max_tokens) for s in seqs]
        write_json_payload(req.output_path, tokens)
    elif req.op == "score_sequences":
        scores = _score(
            model,
            tokenizer,
            seqs,
            device=device,
            reduction=str(req.params.get("reduction", "mean")),
            batch_size=batch_size,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, {"scores": scores})
        meta["reduction"] = str(req.params.get("reduction", "mean"))
    elif req.op == "per_token_logprobs":
        arrays = _per_token_logprobs(
            model,
            tokenizer,
            seqs,
            device=device,
            batch_size=batch_size,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        payload: dict[str, np.ndarray] = {"n": np.array(len(arrays), dtype=np.int64)}
        for i, arr in enumerate(arrays):
            payload[f"arr_{i}"] = arr
        write_arrays_npz(req.output_path, payload)
    else:  # embed
        payload = _embed(
            model,
            tokenizer,
            seqs,
            device=device,
            layers_req=req.params.get("layers", "last"),
            pool=str(req.params.get("pool", "mean")),
            batch_size=batch_size,
            max_tokens=max_tokens,
            pool_include_special=bool(req.params.get("pool_include_special", True)),
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, payload)
        meta["pool"] = str(req.params.get("pool", "mean"))

    Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
        os.path.join(scratch_dir, "response.json")
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.genomeocean_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
