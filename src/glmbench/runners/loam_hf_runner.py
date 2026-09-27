"""LOAM (Hugging Face format) runner — runs in an environment with torch + transformers.

Invoked as ``<python> -m glmbench.runners.loam_hf_runner <request.json>`` by
:class:`glmbench.adapters.loam_hf.LOAMHFAdapter`. It loads a LOAM export
(``AutoModelForCausalLM`` with ``trust_remote_code=True``: the directory ships its own
``modeling_loam.py``) and serves ``tokenize`` | ``score_sequences`` | ``per_token_logprobs`` |
``logprob_embedding`` | ``embed`` | ``embed_logprob``.

**Every op reproduces the native LOAM implementation's arithmetic, not just its semantics**,
so that the released weights give the manuscript's benchmark numbers:

* Weights are loaded in fp32 and then cast to the compute dtype (bf16 on GPU), as native does.
* Each input is ``[BOS, t_1..t_n]``, right-padded, and run through ``model.model`` with **no
  attention mask**. The export keeps that path bitwise identical to the native forward
  (``attn_mask=None, is_causal=True``). It is also exact for padded batches: under a causal
  mask a real token only attends to positions at or before it, and every pad sits after every
  real token.
* Embedding and per-token log-prob batches are length-bucketed under a token budget
  (``glmbench.runners._batching``); sequence scores use length-sorted chunks of
  ``batch_size``, as the native scoring kernel does. Batch composition only moves float
  noise, but matching it makes agreement exact.
* Mean pooling runs in the compute dtype (``pool_dtype: compute``, the default), as native
  does. ``pool_dtype: float32`` pools in fp32 instead: more accurate, and not bitwise
  comparable with the manuscript's rows.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}

_POOLS = ("none", "mean", "max", "last", "bos", "eos")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    """``auto`` → bf16 on CUDA (what LOAM was trained and benchmarked in), fp32 on CPU."""
    if requested is None or requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    key = str(requested).lower()
    if key not in _DTYPES:
        raise ValueError(f"unknown dtype {requested!r}; expected one of {sorted(_DTYPES)} or 'auto'")
    return _DTYPES[key]


def _load(params: dict[str, Any]) -> tuple[Any, Any, torch.device]:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(params["checkpoint"])
    revision = params.get("revision")  # a Hub commit, or None for a local directory
    device = _resolve_device(str(params.get("device", "auto")))
    # transformers >= 5 renamed the from_pretrained dtype argument (torch_dtype → dtype).
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, trust_remote_code=True, **{dtype_kw: torch.float32}
    )
    if getattr(model.config, "model_type", None) != "loam":
        raise ValueError(f"{path} is not a LOAM export (model_type={model.config.model_type!r})")
    dtype = _resolve_dtype(params.get("dtype"), device)
    if dtype != torch.float32:
        model.to(dtype)
    model.to(device)
    model.eval()
    # trust_remote_code: the tokenizer itself is a stock PreTrainedTokenizerFast, but resolving
    # it reads the export's custom config. Without the flag, transformers stops to ask on stdin,
    # which in a runner subprocess hangs the benchmark.
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True)
    return model, tokenizer, device


def _encode(tokenizer: Any, seqs: list[str], max_positions: int) -> list[list[int]]:
    """Token ids per sequence, without special tokens (BOS is added by each op).

    The LOAM release tokenizer maps every character to exactly one token, so a length
    mismatch means the tokenizer is not the one this runner was written for; refuse rather
    than misalign per-nucleotide outputs.
    """
    ids = tokenizer(seqs, add_special_tokens=False)["input_ids"]
    for i, (s, t) in enumerate(zip(seqs, ids, strict=True)):
        if len(t) != len(s):
            raise ValueError(
                f"sequence {i}: {len(s)} characters became {len(t)} tokens; the LOAM "
                "nucleotide tokenizer emits one token per character."
            )
        if len(t) + 1 > max_positions:
            raise ValueError(
                f"sequence {i} has {len(t)} nt; with BOS that is {len(t) + 1} positions, over "
                f"the model's {max_positions}. Window inputs to at most {max_positions - 1} nt."
            )
    return [list(t) for t in ids]


def _batch_tensor(
    chunk: list[list[int]], *, bos_id: int, pad_id: int, device: torch.device
) -> torch.Tensor:
    width = max(len(s) for s in chunk) + 1  # + BOS
    inp = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
    inp[:, 0] = bos_id
    for r, s in enumerate(chunk):
        if s:
            inp[r, 1 : 1 + len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return inp


class _BlockTaps:
    """Forward hooks that record the token-embedding output and every block's output.

    ``taps[0]`` is the embedding output and ``taps[i]`` the raw output of block ``i`` (the
    residual stream after the block's adds, before any model-level norm). Registered once per
    op, cleared per batch, removed on exit.
    """

    def __init__(self, model: Any) -> None:
        inner = model.model
        self.n_blocks = len(inner.layers)
        self.taps: list[torch.Tensor] = []
        self._handles = [inner.embed_tokens.register_forward_hook(self._record)]
        self._handles += [layer.register_forward_hook(self._record) for layer in inner.layers]

    def _record(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.taps.append(output if torch.is_tensor(output) else output[0])

    def run(self, model: Any, inp: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """One forward; returns the ``L + 1`` taps and the final-normed last hidden state."""
        self.taps = []
        out = model.model(input_ids=inp, use_cache=False)
        if len(self.taps) != self.n_blocks + 1:
            raise RuntimeError(
                f"captured {len(self.taps)} tensors, expected {self.n_blocks + 1} (embedding + "
                "one per block); the hooks are not on the forward path."
            )
        return self.taps, out.last_hidden_state

    def __enter__(self) -> _BlockTaps:
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()


def _select_layers(layers_req: Any, n_blocks: int) -> list[int]:
    all_layers = list(range(n_blocks + 1))
    if layers_req in ("last", None):
        return [n_blocks]
    if layers_req == "all":
        return all_layers
    if not isinstance(layers_req, list):
        raise ValueError(f"layers must be 'last', 'all' or a list of ints, got {layers_req!r}")
    return [all_layers[int(i)] for i in layers_req]  # negatives allowed


def _pool(
    stacked: torch.Tensor, real_lens: list[int], pool: str, pool_dtype: str
) -> torch.Tensor:
    """Pool ``stacked`` ``[B, S, n_sel, H]`` (position 0 = BOS) over each row's real tokens."""
    device = stacked.device
    if pool_dtype == "float32":
        stacked = stacked.float()
    real = stacked[:, 1:]
    lens_t = torch.tensor(real_lens, device=device)
    mask = torch.arange(real.shape[1], device=device)[None, :] < lens_t[:, None]
    if pool == "mean":
        m = mask[:, :, None, None].to(real.dtype)
        return (real * m).sum(dim=1) / lens_t[:, None, None].to(real.dtype)
    if pool == "max":
        neg = torch.finfo(real.dtype).min
        return real.masked_fill(~mask[:, :, None, None], neg).amax(dim=1)
    if pool == "bos":
        return stacked[:, 0]
    # "last" / "eos": the final real token of each row
    return real[torch.arange(real.shape[0], device=device), lens_t - 1]


def _token_logprobs(logits: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Log-prob of each realized next token, ``[B, S-1]``; log-softmax in fp32."""
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return log_probs.gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def _embed(
    model: Any,
    token_seqs: list[list[int]],
    *,
    bos_id: int,
    pad_id: int,
    device: torch.device,
    layers_req: Any,
    pool: str,
    pool_dtype: str,
    batch_size: int,
    token_budget: int | None,
    with_logprob: bool = False,
) -> tuple[dict[str, np.ndarray], list[np.ndarray] | None]:
    if pool not in _POOLS:
        raise ValueError(f"unsupported pool {pool!r}; expected one of {_POOLS}")
    if any(len(s) == 0 for s in token_seqs):
        raise ValueError("empty input sequence — nothing to pool.")

    lengths = [len(s) for s in token_seqs]
    payload: dict[str, np.ndarray] = {}
    pooled: list[np.ndarray | None] = [None] * len(token_seqs)
    zs: list[np.ndarray | None] = [None] * len(token_seqs)
    label = "loam-hf/embed+logprob" if with_logprob else "loam-hf/embed"

    with _BlockTaps(model) as taps:
        sel = _select_layers(layers_req, taps.n_blocks)
        prog = BatchProgress(len(token_seqs), label, unit="seqs")
        n_done = 0
        for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
            chunk = [token_seqs[i] for i in idx_batch]
            real_lens = [len(s) for s in chunk]
            inp = _batch_tensor(chunk, bos_id=bos_id, pad_id=pad_id, device=device)
            hiddens, normed_last = taps.run(model, inp)

            if with_logprob:
                tok_lp = _token_logprobs(model.lm_head(normed_last), inp)
                tok_cpu = tok_lp.detach().to("cpu", torch.float64).numpy()
                for r, i in enumerate(idx_batch):
                    zs[i] = tok_cpu[r, : real_lens[r]].copy()

            stacked = torch.stack([hiddens[li] for li in sel], dim=2)  # [B, S, n_sel, H]
            if pool == "none":
                real_cpu = stacked[:, 1:].detach().to("cpu", torch.float32).numpy()
                for r, i in enumerate(idx_batch):
                    n = real_lens[r]
                    payload[f"arr_{i}"] = real_cpu[r, :n].copy()
                    payload[f"span_{i}"] = np.stack(
                        [np.arange(n), np.arange(1, n + 1)], axis=1
                    ).astype(np.int64)
            else:
                reduced = _pool(stacked, real_lens, pool, pool_dtype)
                reduced_cpu = reduced.detach().to("cpu", torch.float32).numpy()
                for r, i in enumerate(idx_batch):
                    pooled[i] = reduced_cpu[r].copy()
            n_done += len(idx_batch)
            prog.update(n_done)
        prog.close()

    emb_dim = int(model.config.hidden_size)
    payload["layer_ids"] = np.array(sel, dtype=np.int64)
    payload["embedding_dim"] = np.array(emb_dim, dtype=np.int64)
    if pool == "none":
        payload["n"] = np.array(len(token_seqs), dtype=np.int64)
    else:
        assert all(p is not None for p in pooled), "bucketing dropped a sequence"
        payload["arrays"] = np.stack(pooled, axis=0)  # type: ignore[arg-type]  # input order
    if not with_logprob:
        return payload, None
    assert all(z is not None for z in zs), "bucketing dropped a sequence"
    return payload, zs  # type: ignore[return-value]


@torch.inference_mode()
def _per_token_logprobs(
    model: Any,
    token_seqs: list[list[int]],
    *,
    bos_id: int,
    pad_id: int,
    device: torch.device,
    batch_size: int,
    token_budget: int | None,
) -> list[np.ndarray]:
    lengths = [len(s) for s in token_seqs]
    out: list[np.ndarray | None] = [None] * len(token_seqs)
    prog = BatchProgress(len(token_seqs), "loam-hf/logprob", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
        chunk = [token_seqs[i] for i in idx_batch]
        inp = _batch_tensor(chunk, bos_id=bos_id, pad_id=pad_id, device=device)
        tok_lp = _token_logprobs(model(input_ids=inp, use_cache=False).logits, inp)
        tok_cpu = tok_lp.detach().to("cpu", torch.float64).numpy()
        for r, i in enumerate(idx_batch):
            out[i] = tok_cpu[r, : len(chunk[r])].copy()
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert all(a is not None for a in out), "bucketing dropped a sequence"
    return out  # type: ignore[return-value]


@torch.inference_mode()
def _score_sequences(
    model: Any,
    token_seqs: list[list[int]],
    *,
    bos_id: int,
    pad_id: int,
    device: torch.device,
    reduction: str,
    batch_size: int,
) -> list[float]:
    """Mean (or summed) log-prob of each sequence's tokens given BOS; BOS itself not scored.

    Sequences are sorted by length and scored in consecutive chunks of ``batch_size`` (the
    native kernel's batching), then returned in input order.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    order = sorted(range(len(token_seqs)), key=lambda i: len(token_seqs[i]))
    scores = [0.0] * len(token_seqs)
    prog = BatchProgress(len(token_seqs), "loam-hf/score", unit="seqs")
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        chunk = [token_seqs[i] for i in idx]
        real_lens = [len(s) for s in chunk]
        inp = _batch_tensor(chunk, bos_id=bos_id, pad_id=pad_id, device=device)
        tok_lp = _token_logprobs(model(input_ids=inp, use_cache=False).logits, inp)
        pos = torch.arange(inp.shape[1] - 1, device=device).unsqueeze(0)
        lens = torch.tensor(real_lens, dtype=torch.long, device=device).unsqueeze(1)
        summed = (tok_lp * (pos < lens).float()).sum(dim=-1)
        for r, i in enumerate(idx):
            n = real_lens[r]
            if n == 0:
                scores[i] = 0.0
            elif reduction == "mean":
                scores[i] = (summed[r] / n).item()
            else:
                scores[i] = summed[r].item()
        prog.update(min(start + batch_size, len(order)))
    prog.close()
    return scores


def run(request_path: str) -> None:
    req = Request.read(request_path)
    _ids, seqs = read_sequences_jsonl(req.input_path)
    params = req.params
    op = req.op
    model, tokenizer, device = _load(params)
    batch_size = int(params.get("batch_size", 8))
    token_budget = int(params["token_budget"]) if params.get("token_budget") else None
    pool_dtype = str(params.get("pool_dtype", "compute"))
    bos_id = int(model.config.bos_token_id)
    pad_id = int(model.config.pad_token_id)
    token_seqs = _encode(tokenizer, seqs, int(model.config.max_position_embeddings))
    meta: dict[str, object] = {"n": len(seqs), "runner": "loam-hf", "device": str(device)}
    common = {"bos_id": bos_id, "pad_id": pad_id, "device": device}

    if op == "tokenize":
        write_json_payload(req.output_path, token_seqs)

    elif op == "score_sequences":
        scores = _score_sequences(
            model,
            token_seqs,
            reduction=str(params.get("reduction", "mean")),
            batch_size=batch_size,
            **common,
        )
        write_arrays_npz(req.output_path, {"scores": np.asarray(scores, dtype=np.float64)})

    elif op in ("per_token_logprobs", "logprob_embedding"):
        # One token is one nucleotide (enforced by _encode), so per-token IS per-nucleotide.
        arrays = _per_token_logprobs(
            model, token_seqs, batch_size=batch_size, token_budget=token_budget, **common
        )
        if op == "logprob_embedding":
            write_ragged_npz(req.output_path, arrays)
        else:
            payload = {"n": np.array(len(arrays), dtype=np.int64)}
            payload.update({f"arr_{i}": a for i, a in enumerate(arrays)})
            write_arrays_npz(req.output_path, payload)

    elif op in ("embed", "embed_logprob"):
        payload, zs = _embed(
            model,
            token_seqs,
            layers_req=params.get("layers", "last"),
            pool=str(params.get("pool", "mean")),
            pool_dtype=pool_dtype,
            batch_size=batch_size,
            token_budget=token_budget,
            with_logprob=op == "embed_logprob",
            **common,
        )
        if zs is not None:
            payload["n_z"] = np.array(len(zs), dtype=np.int64)
            payload.update({f"z_{i}": z for i, z in enumerate(zs)})
        write_arrays_npz(req.output_path, payload)
        meta["pool"] = str(params.get("pool", "mean"))

    else:
        raise ValueError(f"loam-hf runner cannot handle op {op!r}")

    Response(op=op, status="ok", output_path=req.output_path, meta=meta).write(
        os.path.join(os.path.dirname(request_path), "response.json")
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.loam_hf_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
