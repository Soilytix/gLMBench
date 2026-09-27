"""ProkBERT runner — the plain torch+transformers path (ProkBERT masked-LM encoder).

Invoked as ``<env-python> -m glmbench.runners.prokbert_runner <request.json>`` in any env
with ``torch`` + ``transformers`` (the ``glmbench`` env — ProkBERT's remote code runs on
``transformers==5.12.1`` via ``trust_remote_code`` **without** the ``prokbert`` pip
package). It loads a ProkBERT checkpoint
(``neuralbioinfo/prokbert-{mini,mini-long,mini-c}``) with ``trust_remote_code=True`` (the
custom ``prokbert`` arch has no in-tree impl) and implements three ops:

- ``tokenize`` — the LCA k-mer ids ProkBERT consumes, incl. ``[CLS]``/``[SEP]``.
- ``embed`` — hidden states from one ``output_hidden_states=True`` pass, pooled.
- ``score_variant_llr`` — masked-marginal LLR generalized to overlapping k-mers.

**Masked-LM encoder** (``ProkBertForMaskedLM`` — bidirectional, not causal): no
``score_sequences`` / ``per_token_logprobs`` / ``logprob_embedding`` (the adapter declares
only ``{TOKENIZE, EMBEDDING, MASKED_MARGINAL_LLR}``). ``AutoModel`` → base encoder
(``last_hidden_state`` + ``hidden_states``, no ``logits``); ``AutoModelForMaskedLM`` → the
``logits`` head the LLR needs.

**ProkBERT specifics (vs the 1-nt-per-token gLM2 runner this was copied from):**

- **Overlapping-k-mer LCA tokenizer** (``LCATokenizer``, ``is_fast=False``, no
  ``offset_mapping``, ``padding_side=right``). mini = k=6/shift=1 ⇒ ``L`` nt → ``L−5``
  content 6-mer tokens, wrapped ``[CLS] … [SEP]``. ``k``/``shift`` read from the tokenizer
  (``tokenizer.kmer`` / ``tokenizer.shift``).
- **``[CLS]…[SEP]`` wrapping** (gLM2 had a strand marker). Content tokens are the slice
  between them (full index ``1 … n_content``); ``[CLS]`` at index 0, ``[SEP]`` last. Content
  pooling excludes both by default (``pool_include_special`` flips it); ``pool='cls'`` reads
  ``[CLS]``.
- **token_spans are k-mer windows** ``(j·shift, j·shift+k)`` (overlapping for shift<k — their
  *union* covers the sequence; a partition is impossible), built from ``k``/``shift`` (no
  ``offset_mapping``).
- **Masked-marginal LLR over overlapping k-mers.** A single-nt substitution at ``p`` alters
  up to ``k`` overlapping k-mer tokens and the vocab is k-mers, not nucleotides, so ESM's
  single-token convention is generalized: mask *all* content tokens whose window covers ``p``
  on the WT background (one forward), and sum per affected token ``j`` the difference
  ``logit[mut_kmer_j] − logit[wt_kmer_j]``. The mutant/WT k-mer ids are read straight from
  re-tokenizing the (mutated) reference — no vocab-string reconstruction. Any k-mer that maps
  to ``[UNK]`` is skipped+counted.

This is a ``runners/*`` module — the one place allowed to import heavy deps.
The core ``ProkBertAdapter`` stays torch-free and only shells out here over the wire.
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
        # bf16 on GPU, fp32 on CPU (bf16 CPU matmul is slow/uneven).
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name not in mapping:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def _load(req: Request, *, for_logits: bool = False) -> tuple[Any, Any, Any]:
    """Build the ProkBERT model + tokenizer and move to device/dtype. Eval mode.

    Returns ``(model, tokenizer, device)``. Loaded with ``trust_remote_code`` (the custom
    ``prokbert`` arch); ``revision`` pins the weights commit and ``code_revision`` pins the
    ``neuralbioinfo/nbrg-transformers`` remote-code repo (both must be pinned).

    Model class depends on the op: ``embed``/``tokenize`` use ``AutoModel``
    (→ ``ProkBertModel``, no ``logits`` head); ``score_variant_llr`` sets ``for_logits=True``
    and uses ``AutoModelForMaskedLM`` (→ ``ProkBertForMaskedLM``, the masked-LM ``logits``
    head the LLR needs).
    """
    import torch  # noqa: F401  (ensures torch present before transformers import)
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    params = req.params
    device = _resolve_device(str(params.get("device", "auto")))
    dtype = _resolve_dtype(params.get("dtype"), device)
    model_dir = str(params["checkpoint"])
    tokenizer_dir = str(params.get("tokenizer") or model_dir)
    attn_impl = params.get("attn_implementation")  # None → model default
    trust = bool(params.get("trust_remote_code", True))
    revision = params.get("revision")  # None → default branch
    code_revision = params.get("code_revision")  # None → latest remote code

    _ = transformers  # imported for presence; ProkBERT's custom from_pretrained sets no dtype
    common: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        common["revision"] = str(revision)
    if code_revision:
        common["code_revision"] = str(code_revision)

    # ProkBERT's remote `models.py` overrides from_pretrained and forwards **kwargs straight
    # into the encoder's __init__, which rejects the transformers-5 `dtype`/
    # `attn_implementation` kwargs (TypeError). So load plainly and cast with `.to()` after.
    model_kwargs: dict[str, Any] = dict(common)
    if attn_impl:
        model_kwargs["attn_implementation"] = str(attn_impl)
    loader = AutoModelForMaskedLM if for_logits else AutoModel
    try:
        model: Any = loader.from_pretrained(model_dir, **model_kwargs)
    except TypeError:
        # Belt-and-suspenders: retry without attn_implementation if the custom __init__ chokes.
        model = loader.from_pretrained(model_dir, **common)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_dir, padding_side="right", **common
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, device


def _kmer_shift(tokenizer: Any) -> tuple[int, int]:
    """The LCA ``(k, shift)`` from the tokenizer (needed for token_spans + LLR window math)."""
    kmer = getattr(tokenizer, "kmer", None)
    shift = getattr(tokenizer, "shift", None)
    if kmer is None or shift is None:
        raise ValueError(
            "ProkBERT LCATokenizer is missing 'kmer'/'shift' attributes; cannot build "
            f"token_spans or masked-marginal windows (got kmer={kmer!r}, shift={shift!r})."
        )
    return int(kmer), int(shift)


def _encode_full_ids(tokenizer: Any, seqs: list[str], *, max_tokens: int) -> list[list[int]]:
    """Per-sequence full model-input ids: ``[CLS] + kmer_ids + [SEP]``, capped at ``max_tokens``.

    ProkBERT DNA is uppercase; inputs are upper-cased so lowercase never falls to ``[UNK]``.
    Truncation caps the *total* token count (specials included) so the input never exceeds the
    trained window (``max_position_embeddings``).
    """
    out: list[list[int]] = []
    for s in seqs:
        ids = list(
            tokenizer(
                s.upper(), add_special_tokens=True, truncation=True, max_length=int(max_tokens)
            )["input_ids"]
        )
        out.append(ids)
    return out


def _resolve_layers(layers_req: str | list[int] | None, n_states: int) -> list[int]:
    """Resolve ``last``/``all``/int/list to concrete ``hidden_states`` indices.

    ProkBERT returns ``n_layers+1`` states (the standard BERT layout); we index against the
    **actual** tuple length, never a hard-coded assumption. Negatives index from the end.
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

    Pooling positions: content = the k-mer tokens between ``[CLS]`` and ``[SEP]`` (full index
    ``1 … n_content``). ``pool_include_special=False`` (default) pools content only; ``True``
    includes ``[CLS]``/``[SEP]``. ``pool='cls'`` reads the ``[CLS]`` position. ``pool='none'``
    returns the per-content-token vectors + k-mer ``token_spans`` (``[CLS]``/``[SEP]``
    excluded), in original-string coordinates.

    Batches are **length-bucketed** (``_batching.bucketed_batches``) and every result is
    scattered back to its INPUT index — the padded rows a batch carries are masked out of
    both the attention and the pooling, so regrouping rows changes nothing but GEMM tiling.
    """
    import torch

    if pool not in ("mean", "max", "last", "cls", "none"):
        raise ValueError(
            f"ProkBERT embed pool must be mean|max|last|cls|none; got {pool!r}. "
            "(use 'cls' to read the [CLS] position.)"
        )

    kmer, shift = _kmer_shift(tokenizer)
    id_seqs = _encode_full_ids(tokenizer, seqs, max_tokens=max_tokens)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    payload: dict[str, np.ndarray] = {}
    pooled: list[np.ndarray | None] = [None] * len(id_seqs)
    sel: list[int] | None = None
    emb_dim = 0
    lengths = [len(ids) for ids in id_seqs]

    prog = BatchProgress(len(id_seqs), "prokbert/embed", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(
        lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [id_seqs[i] for i in idx_batch]
        lens = [len(ids) for ids in chunk]
        max_full = max(lens) if chunk else 1
        max_full = max(max_full, 1)
        input_ids = torch.full((len(chunk), max_full), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_full), dtype=torch.long, device=device)
        for r, ids in enumerate(chunk):
            if ids:
                input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attn[r, : len(ids)] = 1
        # `forward_hidden_states` hooks the last block and splices its RAW output over
        # hidden_states[-1].
        #
        # ProkBERT was PREDICTED not to need this — "post-LN encoder, the norm is inside the
        # block". The measurement says otherwise: rel_delta 3.14, cos 0.948. Its encoder is
        # pre-LN, with a final LayerNorm on the *encoder*, outside the blocks. The prediction
        # was wrong and the measurement is why we know.
        hs = forward_hidden_states(
            model,
            expected_depth=int(getattr(model.config, "num_hidden_layers", 0)) or None,
            where="prokbert/embed",
            input_ids=input_ids,
            attention_mask=attn,
        )  # list len n_layers+1, each [B, S, H]
        if sel is None:
            sel = _resolve_layers(layers_req, len(hs))
            emb_dim = int(hs[-1].shape[-1])
        stacked = torch.stack([hs[li] for li in sel], dim=2)  # [B, S, n_sel, H]

        if pool == "none":
            # Ragged output: ONE device→host copy for the batch, then slice rows on the host.
            stacked_cpu = stacked.float().detach().cpu().numpy().astype(np.float32)
            for r, i in enumerate(idx_batch):
                n_content = max(lens[r] - 2, 0)  # strip [CLS] (front) and [SEP] (back)
                payload[f"arr_{i}"] = stacked_cpu[r, 1 : 1 + n_content].copy()
                L = len(seqs[i])
                payload[f"span_{i}"] = np.asarray(
                    [(j * shift, min(j * shift + kmer, L)) for j in range(n_content)],
                    dtype=np.int64,
                ).reshape(-1, 2)
        else:
            # Reduce on device (pads never enter: every row is sliced to its own attended
            # length), then ONE device→host copy for the whole batch.
            rows: list[Any] = []
            for r in range(len(chunk)):
                k = lens[r]  # attended length ([CLS] + content + [SEP])
                n_content = max(k - 2, 0)
                content = stacked[r, 1 : 1 + n_content]  # [n_content, n_sel, H]
                if pool == "cls":
                    rows.append(stacked[r, 0].float())
                    continue
                sub = stacked[r, :k].float() if pool_include_special else content.float()
                if sub.shape[0] == 0:  # degenerate (empty content) → fall back to all attended
                    sub = stacked[r, :k].float()
                if sub.shape[0] == 0:  # still empty → empty input
                    sub = stacked[r, :1].float()
                if pool == "mean":
                    rows.append(sub.mean(dim=0))
                elif pool == "max":
                    rows.append(sub.amax(dim=0))
                else:  # last
                    rows.append(sub[-1])
            if rows:
                reduced_cpu = torch.stack(rows, dim=0).detach().cpu().numpy().astype(np.float32)
                for r, i in enumerate(idx_batch):
                    pooled[i] = reduced_cpu[r].copy()

        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()

    if sel is None:  # no inputs
        sel = []
    payload["layer_ids"] = np.asarray(sel, dtype=np.int64)
    payload["embedding_dim"] = np.asarray(emb_dim, dtype=np.int64)
    if pool == "none":
        payload["n"] = np.asarray(len(id_seqs), dtype=np.int64)
    else:
        assert all(p is not None for p in pooled), "bucketing dropped a sequence"
        payload["arrays"] = (
            np.stack(pooled, axis=0)  # type: ignore[arg-type]  # in INPUT order
            if pooled
            else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
        )
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"ProkBERT embed produced non-finite values in {key} — refusing.")
    return payload


def _affected_content_tokens(p: int, kmer: int, shift: int, n_content: int) -> list[int]:
    """Content token indices (0-based) whose k-mer window covers nucleotide position ``p``.

    Content token ``j`` covers the window ``[j·shift, j·shift+kmer)``, so it covers ``p`` iff
    ``j·shift ≤ p < j·shift + kmer`` ⇒ ``j ∈ [⌊(p−kmer)/shift⌋+1, ⌊p/shift⌋]`` clamped to
    ``[0, n_content−1]``. For shift=1 this is the familiar ``[p−kmer+1, p]``.
    """
    lo = max(0, math.floor((p - kmer) / shift) + 1)
    hi = min(n_content - 1, p // shift)
    return list(range(lo, hi + 1)) if lo <= hi else []


def _score_variant_llr(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    max_tokens: int,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Masked-marginal LLR per variant, generalized to ProkBERT's overlapping k-mers.

    For each mutated nucleotide position ``p`` of a variant, the affected content tokens
    (those whose k-mer window covers ``p``) are masked on the WT background and one forward
    pass yields their logit rows; the per-token contribution is
    ``logit[mut_kmer_j] − logit[wt_kmer_j]`` (shared masked context ⇒ the softmax normalizer
    cancels, so the raw logit difference *is* the k-mer log-probability ratio). The mutant/WT
    k-mer ids are read straight from re-tokenizing the (mutated) reference. Single-site LLR =
    Σ over affected tokens; multi-site = additive across sites (each site masked
    independently on the WT background). Any k-mer mapping to ``[UNK]`` is skipped+counted.

    Efficiency: one masked forward per **unique (reference, position)** pair — the mask set is
    base-independent — cached and reused across every variant touching that pair, batched. The
    masked forwards are **length-bucketed** (``_batching.bucketed_batches``), so a batch mixing
    a short and a long reference no longer pays the long one's width for every row; results are
    keyed by ``(ref_idx, position)``, never by batch position.
    """
    import torch

    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "ProkBERT tokenizer has no mask_token_id; cannot compute masked-marginal LLR "
            "(the MLM [MASK] token is required to mask each affected k-mer token)."
        )
    mask_id = int(mask_id)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_id = int(unk_id) if unk_id is not None else None
    kmer, shift = _kmer_shift(tokenizer)
    cls_offset = 1  # [CLS] leads; content token j sits at full input index j+1

    # Unique references (upper-cased) → their full model-input id sequences.
    ref_index: dict[str, int] = {}
    ref_strings: list[str] = []
    for it in items:
        ref = str(it["reference"]).upper()
        if ref not in ref_index:
            ref_index[ref] = len(ref_strings)
            ref_strings.append(ref)
    ref_full = _encode_full_ids(tokenizer, ref_strings, max_tokens=max_tokens)

    # Collect the unique (ref_idx, position) masked-forward jobs and their affected tokens.
    job_tokens: dict[tuple[int, int], list[int]] = {}
    for it in items:
        ridx = ref_index[str(it["reference"]).upper()]
        n_content = max(len(ref_full[ridx]) - 2, 0)
        for pos0, _wt, _mut in it["mutations"]:
            p = int(pos0)
            key = (ridx, p)
            if key in job_tokens:
                continue
            aff = _affected_content_tokens(p, kmer, shift, n_content)
            if not aff:
                raise ValueError(
                    f"mutated position {p} covers no content k-mer token for a reference of "
                    f"{n_content} content tokens (cap max_tokens={max_tokens}). The task skips "
                    "over-context references; this is the loud backstop."
                )
            job_tokens[key] = aff
    jobs = list(job_tokens.keys())

    # Run the masked forwards in length-bucketed batches; store per job {content_j: logit_row}.
    # Results are keyed by the job's (ref_idx, position) — read from `jobs[i]` for the INPUT
    # index `i` the bucketer hands back — so reordering rows cannot mis-attribute a row.
    logit_rows: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    job_lengths = [len(ref_full[r]) for (r, _p) in jobs]
    prog = BatchProgress(len(jobs), "prokbert/llr", unit="masked-forwards")
    n_done = 0
    for idx_batch in bucketed_batches(
        job_lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [jobs[i] for i in idx_batch]
        lens = [len(ref_full[r]) for (r, _p) in chunk]
        max_full = max(lens) if chunk else 1
        input_ids = torch.full((len(chunk), max_full), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_full), dtype=torch.long, device=device)
        for i, (r, _p) in enumerate(chunk):
            ids = list(ref_full[r])
            for j in job_tokens[(r, _p)]:
                ids[j + cls_offset] = mask_id  # mask every affected k-mer token
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[i, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, S, V]
        # Gather only the affected rows and copy them device→host ONCE for the whole batch
        # (the full [B, S, V] logit tensor is far too large to move, and a per-token .cpu()
        # was a sync per affected k-mer).
        rows_i: list[int] = []
        cols_j: list[int] = []
        for i, (r, _p) in enumerate(chunk):
            for j in job_tokens[(r, _p)]:
                rows_i.append(i)
                cols_j.append(j + cls_offset)
        gathered = (
            logits[rows_i, cols_j].float().detach().cpu().numpy()  # [n_affected, V]
            if rows_i
            else np.zeros((0, int(logits.shape[-1])), dtype=np.float32)
        )
        if not np.isfinite(gathered).all():
            raise ValueError("ProkBERT produced non-finite logits — refusing to score.")
        cursor = 0
        for r, _p in chunk:
            rows: dict[int, np.ndarray] = {}
            for j in job_tokens[(r, _p)]:
                rows[j] = gathered[cursor]
                cursor += 1
            logit_rows[(r, _p)] = rows
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()
    assert len(logit_rows) == len(jobs), "bucketing dropped a masked-forward job"

    # Mutant k-mer ids for the affected tokens. The naive form re-tokenizes the ENTIRE mutated
    # reference per (ref, pos, mut_base) — O(L) each × O(L) sites = O(L²) CPU work that
    # dominates the LLR path on kb-scale references. The LCA tokenizer is a purely *positional*
    # sliding window (content token j == the k-mer at nt offset j·shift), so tokenizing only the
    # affected nt window and splicing yields identical ids. Guarded, not assumed: per (ref, lo,
    # hi) we first re-tokenize the **WT** window and require it to reproduce the WT content ids
    # at exactly those indices; if it does not, we fall back to the full-sequence tokenization.
    window_trusted: dict[tuple[int, int, int], bool] = {}
    mut_cache: dict[tuple[int, int, str], dict[int, int]] = {}

    def window_content_ids(seq: str, lo: int, hi: int) -> list[int] | None:
        """Content ids for tokens ``lo..hi`` of ``seq``, tokenizing only the covering window."""
        sub = seq[lo * shift : min(hi * shift + kmer, len(seq))]
        content = _encode_full_ids(tokenizer, [sub], max_tokens=max_tokens)[0][1:-1]
        return content if len(content) == hi - lo + 1 else None

    def is_window_trusted(ridx: int, lo: int, hi: int) -> bool:
        key = (ridx, lo, hi)
        if key not in window_trusted:
            wt = ref_full[ridx][1:-1]
            got = window_content_ids(ref_strings[ridx], lo, hi)
            window_trusted[key] = got is not None and got == [
                int(x) for x in wt[lo : hi + 1]
            ]
        return window_trusted[key]

    def mut_kmer_ids(ridx: int, p: int, mut: str, aff: list[int]) -> dict[int, int]:
        """``{content_j: mutant k-mer id}`` for the affected tokens of one (ref, pos, base)."""
        key = (ridx, p, mut.upper())
        if key in mut_cache:
            return mut_cache[key]
        ref = ref_strings[ridx]
        mref = ref[:p] + mut.upper() + ref[p + 1 :]
        lo, hi = aff[0], aff[-1]  # `aff` is a contiguous run of content indices
        ids: dict[int, int] | None = None
        if is_window_trusted(ridx, lo, hi):
            win = window_content_ids(mref, lo, hi)
            if win is not None:
                ids = {j: int(win[j - lo]) for j in aff}
        if ids is None:  # untrusted/odd tokenizer → the exact, slow path
            content = _encode_full_ids(tokenizer, [mref], max_tokens=max_tokens)[0][1:-1]
            ids = {j: int(content[j]) for j in aff}
        mut_cache[key] = ids
        return ids

    scores = np.empty(len(items), dtype=np.float64)
    n_unk = 0
    for idx, it in enumerate(items):
        ridx = ref_index[str(it["reference"]).upper()]
        wt_content = ref_full[ridx][1:-1]
        total = 0.0
        for pos0, _wt, mut in it["mutations"]:
            p = int(pos0)
            aff = job_tokens[(ridx, p)]
            rows = logit_rows[(ridx, p)]
            mut_ids = mut_kmer_ids(ridx, p, str(mut), aff)
            for j in aff:
                wt_id = int(wt_content[j])
                mut_id = int(mut_ids[j])
                if unk_id is not None and (wt_id == unk_id or mut_id == unk_id):
                    n_unk += 1
                    continue
                total += float(rows[j][mut_id] - rows[j][wt_id])
        scores[idx] = total
    if not np.isfinite(scores).all():
        raise ValueError("ProkBERT masked-marginal LLR produced non-finite scores — refusing.")
    return {"scores": scores, "n_unk_tokens": np.asarray(n_unk, dtype=np.int64)}


def run(request_path: str) -> None:
    req = Request.read(request_path)
    if req.op not in ("tokenize", "embed", "score_variant_llr"):
        raise ValueError(
            f"ProkBERT runner implements 'tokenize', 'embed', and 'score_variant_llr', got "
            f"{req.op!r}. (ProkBertAdapter declares TOKENIZE + EMBEDDING + MASKED_MARGINAL_LLR; "
            "ProkBERT is a masked-LM encoder, so the causal SEQUENCE_LOGLIKELIHOOD / "
            "PER_TOKEN_LOGPROBS / LOGPROB_EMBEDDING are N/A.)"
        )
    batch_size = int(req.params.get("batch_size", 16))
    max_tokens = int(req.params.get("max_tokens", 1024))
    # Optional cap on `rows × padded_len` per batch; adapts the batch to the sequence length
    # instead of the hand-tuned batch_size magic number. None → batch_size alone bounds it.
    token_budget = req.params.get("token_budget")
    token_budget = int(token_budget) if token_budget else None
    scratch_dir = os.path.dirname(request_path)

    if req.op == "score_variant_llr":
        _ids, items = read_variant_jsonl(req.input_path)
        model, tokenizer, device = _load(req, for_logits=True)
        payload = _score_variant_llr(
            model,
            tokenizer,
            items,
            device=device,
            batch_size=batch_size,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, payload)
        Response(
            op=req.op,
            status="ok",
            output_path=req.output_path,
            meta={
                "n": len(items),
                "runner": "prokbert",
                "device": str(device),
                "n_unk_tokens": int(payload["n_unk_tokens"]),
            },
        ).write(os.path.join(scratch_dir, "response.json"))
        return

    _ids, seqs = read_sequences_jsonl(req.input_path)
    model, tokenizer, device = _load(req)
    meta: dict[str, object] = {"n": len(seqs), "runner": "prokbert", "device": str(device)}

    if req.op == "tokenize":
        tokens = _encode_full_ids(tokenizer, seqs, max_tokens=max_tokens)
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
            pool_include_special=bool(req.params.get("pool_include_special", False)),
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
        print("usage: python -m glmbench.runners.prokbert_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
