"""NTv3 runner — the plain torch+transformers path (InstaDeep NTv3 U-Net masked-LM encoder).

Invoked as ``<env-python> -m glmbench.runners.ntv3_runner <request.json>`` in any env with
``torch`` + ``transformers`` (the ``glmbench`` env, as-is — no extra install). It loads
``InstaDeepAI/NTv3_100M_pre`` with ``trust_remote_code=True`` (the custom ``ntv3`` arch has no
in-tree impl; the ``auto_map`` is the cross-repo ``InstaDeepAI/ntv3_base_model--…`` form, so
both ``revision`` and ``code_revision`` are pinned) and implements three ops:

- ``tokenize`` — one nucleotide id per base (no wrapping), capped at ``max_tokens``.
- ``embed`` — hidden states from one ``output_hidden_states=True`` pass, pooled.
- ``score_variant_llr`` — masked-marginal LLR (the MLM variant-effect scoring path).

**Masked-LM U-Net encoder.** NTv3 is ``NTv3PreTrained`` — ``conv-down → transformer torso →
deconv-up → LM head`` — bidirectional, not causal — so there is no ``score_sequences`` /
``per_token_logprobs`` / ``logprob_embedding`` (the adapter declares only ``{TOKENIZE,
EMBEDDING, MASKED_MARGINAL_LLR}``). ``AutoModelForMaskedLM`` gives ``logits (B, L, alphabet)``
+ ``hidden_states``.

**NTv3 specifics (vs the gLM2 runner this was copied from):**

- **128-multiple padding + NO attention mask (the load-bearing quirk).** The U-Net downsamples
  ``2^num_downsamples = 128×``, so the input length must be a multiple of ``pad_multiple`` (128),
  and the transformer torso runs with ``attention_mask=None`` (the model ignores any mask), so
  ``<pad>`` tokens contaminate real positions through global torso attention. To make results
  batch-invariant we pad **each sequence to its own next multiple of ``pad_multiple``** and
  **bucket equal-padded-length sequences** into a forward — never to a batch-wide max. Probe:
  same sequence padded to 128 vs 256 differs (|Δlogits|≈1.34); per-seq fixed padding is Δ=0.0.
- **Mixed-resolution hidden states.** ``output_hidden_states`` returns ``2·num_downsamples
  + num_layers`` states at lengths ``L, L/2 … L/128 …(torso, L/128)… L/2, L``. They CANNOT be
  stacked across the token axis (different lengths). Each selected layer is pooled over its own
  non-pad span (``ceil(L / f)`` positions, ``f = padded_len // layer_len``), then stacked over
  the layer axis (all ``[H]``). ``pool='none'`` requires all selected layers to share one
  resolution (else a loud error); spans are ``f``-wide (final layer ⇒ unit-width).
- **Single-nt tokenizer, no wrapping, slow (no offset mapping).** 1 nt/token; ``token_spans``
  built by hand; non-ACGTN → ``<unk>``. No strand marker, no lowercasing (unlike gLM2).

This is a ``runners/*`` module — the one place allowed to import heavy deps.
The core ``NTv3Adapter`` stays torch-free and only shells out here over the wire.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

from glmbench.adapters.wire import (
    Request,
    Response,
    read_sequences_jsonl,
    read_variant_jsonl,
    write_arrays_npz,
    write_json_payload,
)
from glmbench.runners._progress import BatchProgress


def _resolve_device(requested: str) -> Any:
    import torch

    if requested in ("auto", None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load(req: Request) -> tuple[Any, Any, Any]:
    """Build the NTv3 model + tokenizer and move to device. Eval mode.

    Loaded via ``AutoModelForMaskedLM`` (the ``NTv3PreTrained`` masked-LM head — serves both
    the ``logits`` the LLR needs and ``hidden_states`` for embeddings). ``trust_remote_code``
    is required (custom arch); ``revision`` pins the weights commit and ``code_revision`` pins
    the cross-repo modeling-code commit (``ntv3_base_model``).

    **Params are loaded in fp32 and NOT cast.** NTv3 manages its own mixed precision internally
    via the per-block ``*_compute_dtype`` config knobs (all ``float32`` in the shipped
    checkpoint) — its hand-rolled conv/attention paths assume fp32 params + autocast compute, so
    casting params to bf16 breaks them (``Input type FloatTensor and weight type BFloat16Type``).
    The ``dtype`` spec knob therefore does not change the param dtype; per-block bf16 compute is
    a model-internal config choice, not a load-time cast.
    """
    import torch  # noqa: F401  (ensure torch present before transformers import)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    params = req.params
    device = _resolve_device(str(params.get("device", "auto")))
    model_dir = str(params["checkpoint"])
    tokenizer_dir = str(params.get("tokenizer") or model_dir)
    trust = bool(params.get("trust_remote_code", True))
    revision = params.get("revision")  # None → default branch
    code_revision = params.get("code_revision")  # None → latest remote code

    common: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        common["revision"] = str(revision)
    if code_revision:
        common["code_revision"] = str(code_revision)

    model: Any = AutoModelForMaskedLM.from_pretrained(model_dir, **common)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_dir, padding_side="right", **common
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _encode_ids(tokenizer: Any, seqs: list[str], *, max_tokens: int) -> list[list[int]]:
    """Per-sequence model-input ids: one nucleotide id per base (no wrapping), capped.

    1 nt/token (char-level, no merges). NTv3's tokenizer adds no ``<cls>/<eos>/<bos>``
    wrapping (``build_inputs_with_special_tokens`` is a no-op for a single sequence), so the
    ids are exactly the nucleotides truncated to ``max_tokens``.
    """
    out: list[list[int]] = []
    cap = max(int(max_tokens), 1)
    for s in seqs:
        ids = list(
            tokenizer(s, add_special_tokens=False, truncation=True, max_length=cap)["input_ids"]
        )
        out.append(ids)
    return out


def _pad_len(n: int, pad_multiple: int) -> int:
    """The next multiple of ``pad_multiple`` >= ``n`` (and >= ``pad_multiple``, since the
    U-Net cannot process a length below one downsample block)."""
    if n <= 0:
        return pad_multiple
    return int(math.ceil(n / pad_multiple)) * pad_multiple


def _resolve_layers(layers_req: str | list[int] | None, n_states: int) -> list[int]:
    """Resolve ``last``/``all``/int/list to concrete ``hidden_states`` indices.

    NTv3 returns ``2·num_downsamples + num_layers`` MIXED-resolution states; we index against
    the **actual** length. Negatives index from the end. ``'last'`` → the final state (the
    deconv output at nucleotide resolution).
    """
    all_layers = list(range(n_states))
    if layers_req in ("last", None):
        return [all_layers[-1]]
    if layers_req == "all":
        return all_layers
    if isinstance(layers_req, int):
        return [all_layers[layers_req]]
    if isinstance(layers_req, (list, tuple)) and layers_req:
        return [all_layers[int(i)] for i in layers_req]
    raise ValueError(
        f"layers must be 'last', 'all', an int, or a non-empty list of ints; got {layers_req!r}."
    )


def _bucketed_batches(
    id_seqs: list[list[int]], pad_multiple: int, batch_size: int
) -> list[list[int]]:
    """Group input indices into batches of EQUAL padded length (batch-invariance).

    Every sequence is padded to its own ``_pad_len(len, pad_multiple)``; batching only
    sequences that share that padded length guarantees each sequence always sees the same
    total length (hence the same fixed pad contamination) regardless of batch composition —
    the only way to be batch-invariant given the mask-free torso.
    """
    by_len: dict[int, list[int]] = {}
    for i, ids in enumerate(id_seqs):
        by_len.setdefault(_pad_len(len(ids), pad_multiple), []).append(i)
    batches: list[list[int]] = []
    for _plen, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), batch_size):
            batches.append(idxs[s : s + batch_size])
    return batches


def _forward_hidden(model: Any, input_ids: Any) -> tuple[Any, ...]:
    """One ``output_hidden_states=True`` forward → the hidden-states tuple.

    No ``attention_mask`` — NTv3's torso ignores it; batch-invariance comes from the
    per-sequence fixed padding (all rows in a batch share one padded length).
    """
    import torch

    with torch.inference_mode():
        return model(input_ids=input_ids, output_hidden_states=True).hidden_states


def _pool_layer(layer: Any, real_len: int, pool: str) -> Any:
    """Pool one hidden-state tensor ``[Lh, H]`` over its first ``real_len`` positions → ``[H]``."""
    sub = layer[:real_len]
    if sub.shape[0] == 0:  # degenerate (all pad / empty) → fall back to the whole layer
        sub = layer[:1]
    if pool == "mean":
        return sub.mean(dim=0)
    if pool == "max":
        return sub.amax(dim=0)
    # last
    return sub[-1]


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
    pad_multiple: int,
    pool_include_special: bool,
) -> dict[str, np.ndarray]:
    """Hidden-state embeddings (the layer convention), all selected layers from one forward pass.

    Mixed-resolution layers are pooled independently over their own non-pad span
    (``ceil(len(seq)/f)`` positions, ``f = padded_len // layer_len``). ``pool_include_special``
    is ``False`` by default and pools over the non-pad (content) span; ``True`` pools over the
    full padded length (includes the pad region). ``pool='none'`` returns per-position vectors
    + ``f``-wide ``token_spans`` and requires all selected layers to share one resolution.
    """
    import torch

    if pool not in ("mean", "max", "last", "none"):
        raise ValueError(
            f"NTv3 embed pool must be mean|max|last|none; got {pool!r}. "
            "(NTv3 has no BOS/CLS/strand-marker; every input token is a nucleotide.)"
        )

    id_seqs = _encode_ids(tokenizer, seqs, max_tokens=max_tokens)
    real_lens = [len(ids) for ids in id_seqs]
    pad_id = int(getattr(tokenizer, "pad_token_id", 1) or 0)

    payload: dict[str, np.ndarray] = {}
    pooled_rows: list[np.ndarray | None] = [None] * len(seqs)
    none_arrays: list[np.ndarray | None] = [None] * len(seqs)
    none_spans: list[np.ndarray | None] = [None] * len(seqs)
    sel: list[int] | None = None
    emb_dim = 0

    # Bucketed batching ⇒ report items done (not batch count) so the % is meaningful
    # across uneven, length-sorted buckets.
    prog = BatchProgress(len(id_seqs), "ntv3/embed", unit="seqs")
    n_done = 0
    for batch_idx in _bucketed_batches(id_seqs, pad_multiple, batch_size):
        plen = _pad_len(max((real_lens[i] for i in batch_idx), default=0), pad_multiple)
        input_ids = torch.full((len(batch_idx), plen), pad_id, dtype=torch.long, device=device)
        for row, i in enumerate(batch_idx):
            ids = id_seqs[i]
            if ids:
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        hs = _forward_hidden(model, input_ids)  # tuple, each [B, Lh, H] (Lh varies per layer)
        if sel is None:
            sel = _resolve_layers(layers_req, len(hs))
            emb_dim = int(hs[-1].shape[-1])
        sel_lens = [int(hs[li].shape[1]) for li in sel]
        if pool == "none" and len(set(sel_lens)) > 1:
            raise ValueError(
                "NTv3 embed(pool='none') needs all selected layers at ONE resolution, but the "
                f"selected layers have lengths {sel_lens} (NTv3's hidden states are at mixed "
                "conv/torso/deconv resolutions). Select a single layer (e.g. 'last') or "
                "same-resolution layers for per-token output."
            )
        for row, i in enumerate(batch_idx):
            n_real = real_lens[i]
            if pool == "none":
                f = plen // sel_lens[0]
                keep = min(max(math.ceil(n_real / f), 1), sel_lens[0])
                vec = torch.stack([hs[lj][row, :keep] for lj in sel], dim=1)  # [keep, n_sel, H]
                none_arrays[i] = vec.float().detach().cpu().numpy().astype(np.float32)
                spans = np.asarray(
                    [(j * f, min((j + 1) * f, n_real)) for j in range(keep)], dtype=np.int64
                ).reshape(-1, 2)
                none_spans[i] = spans
                continue
            layer_vecs = []
            for lj, Lh in zip(sel, sel_lens, strict=True):  # noqa: N806
                f = plen // Lh
                real_in_layer = Lh if pool_include_special else min(
                    max(math.ceil(n_real / f), 1), Lh
                )
                layer_vecs.append(_pool_layer(hs[lj][row].float(), real_in_layer, pool))
            pooled_rows[i] = (
                torch.stack(layer_vecs, dim=0).detach().cpu().numpy().astype(np.float32)
            )
        n_done += len(batch_idx)
        prog.update(n_done)
    prog.close()

    if sel is None:  # no inputs
        sel = []
    payload["layer_ids"] = np.asarray(sel, dtype=np.int64)
    payload["embedding_dim"] = np.asarray(emb_dim, dtype=np.int64)
    if pool == "none":
        payload["n"] = np.asarray(len(seqs), dtype=np.int64)
        for i in range(len(seqs)):
            payload[f"arr_{i}"] = (
                none_arrays[i]
                if none_arrays[i] is not None
                else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
            )
            payload[f"span_{i}"] = (
                none_spans[i]
                if none_spans[i] is not None
                else np.zeros((0, 2), dtype=np.int64)
            )
    else:
        payload["arrays"] = (
            np.stack([r for r in pooled_rows if r is not None], axis=0)
            if any(r is not None for r in pooled_rows)
            else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
        )
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"NTv3 embed produced non-finite values in {key} — refusing.")
    return payload


def _base_token_id(tokenizer: Any, base: str) -> int:
    """Token id for a single (uppercased) nucleotide base. Fails loudly on ``<unk>``.

    NTv3's tokenizer uppercases internally, so WT/MUT bases are uppercased before lookup. A
    base that maps to ``<unk>`` is not a clean nucleotide token and would silently corrupt the
    logit read (the LLR is only defined for ACGT(N) substitutions).
    """
    tid = int(tokenizer.convert_tokens_to_ids(base.upper()))
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        raise ValueError(
            f"NTv3 tokenizer maps base {base!r} to {tid} (<unk>={unk}); a masked-marginal "
            "LLR needs a clean single-nucleotide token id for both WT and MUT bases."
        )
    return tid


def _score_variant_llr(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    max_tokens: int,
    pad_multiple: int,
) -> dict[str, np.ndarray]:
    """Masked-marginal LLR per variant: Σ_i (logit_mut_i − logit_wt_i) on the WT background.

    For every mutated position ``i`` of a variant, the reference (padded to its own next
    multiple of ``pad_multiple``) is masked at token index ``i`` and a single forward yields
    the logit row; the contribution is ``logit[mut_id] − logit[wt_id]`` (the shared masked
    context cancels the softmax normalizer, so the raw logit difference *is* the log-prob
    ratio). Each mutated position is masked **independently** on the WT background (the additive
    single-site approximation) and summed.

    Efficiency: one masked forward per **unique (reference, position)** pair, cached and reused
    across every variant touching that pair. Jobs are bucketed by padded length so each
    reference is always scored at its own fixed length — batch-invariant.
    """
    import torch

    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "NTv3 tokenizer has no mask_token_id; cannot compute masked-marginal LLR "
            "(the MLM [MASK] token is required to mask each mutated position)."
        )
    mask_id = int(mask_id)
    pad_id = int(getattr(tokenizer, "pad_token_id", 1) or 0)

    # Unique references → their model-input id sequences (one nt id per base, capped).
    ref_index: dict[str, int] = {}
    ref_strings: list[str] = []
    for it in items:
        ref = str(it["reference"])
        if ref not in ref_index:
            ref_index[ref] = len(ref_strings)
            ref_strings.append(ref)
    ref_ids = _encode_ids(tokenizer, ref_strings, max_tokens=max_tokens)
    ref_plen = [_pad_len(len(ids), pad_multiple) for ids in ref_ids]

    # Collect the unique (ref_idx, token_index) masked-forward jobs across all variants.
    job_set: dict[tuple[int, int], None] = {}
    for it in items:
        ridx = ref_index[str(it["reference"])]
        ids = ref_ids[ridx]
        for pos0, _wt, _mut in it["mutations"]:
            t = int(pos0)
            if t < 0 or t >= len(ids):
                raise ValueError(
                    f"masked position {pos0} is out of range for a reference of {len(ids)} "
                    f"tokens (cap max_tokens={max_tokens}). The task skips over-context "
                    "references; this is the loud backstop."
                )
            job_set[(ridx, t)] = None
    jobs = list(job_set.keys())

    # Bucket jobs by the reference's padded length (batch-invariance), then batch.
    by_plen: dict[int, list[tuple[int, int]]] = {}
    for (r, t) in jobs:
        by_plen.setdefault(ref_plen[r], []).append((r, t))

    logit_rows: dict[tuple[int, int], np.ndarray] = {}
    prog = BatchProgress(len(jobs), "ntv3/llr", unit="masked-forwards")
    n_done = 0
    for plen, group in sorted(by_plen.items()):
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            input_ids = torch.full(
                (len(chunk), plen), pad_id, dtype=torch.long, device=device
            )
            for row, (r, t) in enumerate(chunk):
                ids = list(ref_ids[r])
                ids[t] = mask_id  # mask exactly the target position on the WT background
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits  # [B, plen, V]
            for row, (r, t) in enumerate(chunk):
                vals = logits[row, t].float().detach().cpu().numpy()
                if not np.isfinite(vals).all():
                    raise ValueError("NTv3 produced non-finite logits — refusing to score.")
                logit_rows[(r, t)] = vals
            n_done += len(chunk)
            prog.update(n_done)
    prog.close()

    base_id_cache: dict[str, int] = {}

    def base_id(b: str) -> int:
        key = b.upper()
        if key not in base_id_cache:
            base_id_cache[key] = _base_token_id(tokenizer, key)
        return base_id_cache[key]

    scores = np.empty(len(items), dtype=np.float64)
    for idx, it in enumerate(items):
        ridx = ref_index[str(it["reference"])]
        total = 0.0
        for pos0, wt, mut in it["mutations"]:
            row = logit_rows[(ridx, int(pos0))]
            total += float(row[base_id(str(mut))] - row[base_id(str(wt))])
        scores[idx] = total
    if not np.isfinite(scores).all():
        raise ValueError("NTv3 masked-marginal LLR produced non-finite scores — refusing.")
    return {"scores": scores}


def run(request_path: str) -> None:
    req = Request.read(request_path)
    if req.op not in ("tokenize", "embed", "score_variant_llr"):
        raise ValueError(
            f"NTv3 runner implements 'tokenize', 'embed', and 'score_variant_llr', got "
            f"{req.op!r}. (NTv3Adapter declares TOKENIZE + EMBEDDING + MASKED_MARGINAL_LLR; "
            "NTv3 is a masked-LM encoder, so the causal SEQUENCE_LOGLIKELIHOOD / "
            "PER_TOKEN_LOGPROBS / LOGPROB_EMBEDDING are N/A.)"
        )
    batch_size = int(req.params.get("batch_size", 4))
    max_tokens = int(req.params.get("max_tokens", 12288))
    pad_multiple = int(req.params.get("pad_multiple", 128))
    scratch_dir = os.path.dirname(request_path)

    if req.op == "score_variant_llr":
        _ids, items = read_variant_jsonl(req.input_path)
        model, tokenizer, device = _load(req)
        payload = _score_variant_llr(
            model,
            tokenizer,
            items,
            device=device,
            batch_size=batch_size,
            max_tokens=max_tokens,
            pad_multiple=pad_multiple,
        )
        write_arrays_npz(req.output_path, payload)
        Response(
            op=req.op,
            status="ok",
            output_path=req.output_path,
            meta={"n": len(items), "runner": "ntv3", "device": str(device)},
        ).write(os.path.join(scratch_dir, "response.json"))
        return

    _ids, seqs = read_sequences_jsonl(req.input_path)
    model, tokenizer, device = _load(req)
    meta: dict[str, object] = {"n": len(seqs), "runner": "ntv3", "device": str(device)}

    if req.op == "tokenize":
        tokens = _encode_ids(tokenizer, seqs, max_tokens=max_tokens)
        write_json_payload(req.output_path, tokens)
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
            pad_multiple=pad_multiple,
            pool_include_special=bool(req.params.get("pool_include_special", False)),
        )
        write_arrays_npz(req.output_path, payload)
        meta["pool"] = str(req.params.get("pool", "mean"))

    Response(op=req.op, status="ok", output_path=req.output_path, meta=meta).write(
        os.path.join(scratch_dir, "response.json")
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m glmbench.runners.ntv3_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
