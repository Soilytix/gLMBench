"""gLM2 runner — the plain torch+transformers+einops path (TattaBio gLM2 encoder).

Invoked as ``<env-python> -m glmbench.runners.glm2_runner <request.json>`` in any env with
``torch`` + ``transformers`` + ``einops`` (the ``glmbench`` env after ``pip install -e
".[hf]"``). It loads a gLM2 checkpoint (``tattabio/gLM2_{150M,650M}``) via **``AutoModel``**
with ``trust_remote_code=True`` (the custom ``gLM2`` arch has no in-tree impl) and implements
two ops:

- ``tokenize`` — the token ids gLM2 consumes: the strand marker (when ``strand_prefix`` !=
  ``none``) followed by one nucleotide id per base of the **lowercased** sequence.
- ``embed`` — hidden states from one ``output_hidden_states=True`` pass, pooled.

**Masked-LM encoder, embeddings-only.** gLM2 is ``gLM2ForMaskedLM`` — bidirectional, not
causal — so this runner has no ``score_sequences`` / ``per_token_logprobs`` /
``logprob_embedding`` (the adapter declares only ``{TOKENIZE, EMBEDDING}``). ``AutoModel``
gives the base encoder (``last_hidden_state`` + ``hidden_states``, no ``logits``).

**gLM2 specifics (vs the BPE genomeocean runner this was copied from):**

- **Case-sensitive, mixed-modality tokenizer.** gLM2 encodes amino acids UPPERCASE and
  nucleotides **lowercase**; uppercase ``ACGT`` is silently read as amino acids. We
  **lowercase** every input (``_preprocess``). 1 nt/token (no multi-nt merges); ``<unk>``=3
  for non-acgt bases. No ``[CLS]…[SEP]`` wrapping, no BOS.
- **Strand marker.** Every genomic element is trained with a leading strand marker. We
  prepend one (``<+>``=33 default / ``<->``=34), so a bare contig is in-distribution. The
  marker is an ordinary vocab token (NOT in ``special_tokens_mask``); we track its leading
  position explicitly and **exclude** it from content pooling / ``token_spans`` by default.
- **Layer count.** gLM2's ``hidden_states`` tuple length is ``depth`` (not ``depth+1``); we
  resolve ``layers`` against the **actual** tuple length, never a ``+1`` assumption.
- **token_spans** are built in **original-string** coordinates (1 nt/token ⇒ ``(j, j+1)``),
  not from ``offset_mapping`` (which indexes the lowercased+marker-prefixed string).

This is a ``runners/*`` module — the one place allowed to import heavy deps.
The core ``GLM2Adapter`` stays torch-free and only shells out here over the wire.
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
    read_variant_jsonl,
    write_arrays_npz,
    write_json_payload,
)
from glmbench.runners._batching import bucketed_batches
from glmbench.runners._progress import BatchProgress

_STRAND_TOKEN = {"+": "<+>", "-": "<->"}


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
        # bf16 on GPU (gLM2's native dtype), fp32 on CPU (bf16 CPU matmul is slow/uneven).
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name not in mapping:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]


def _load(req: Request, *, for_logits: bool = False) -> tuple[Any, Any, Any, int | None]:
    """Build the gLM2 model + tokenizer and move to device/dtype. Eval mode.

    Returns ``(model, tokenizer, device, marker_id)`` where ``marker_id`` is the strand-marker
    token id (``<+>``/``<->``) or ``None`` when ``strand_prefix='none'``. Loaded with
    ``trust_remote_code`` (the custom gLM2 arch); ``revision`` pins the HF commit.

    Model class depends on the op (the documented choice — pay for the LM head only when it
    is needed): ``embed``/``tokenize`` use ``AutoModel`` (base encoder, no ``logits`` head);
    ``score_variant_llr`` sets ``for_logits=True`` and uses ``AutoModelForMaskedLM`` (the
    full ``gLM2ForMaskedLM`` with the masked-LM ``logits`` head the LLR needs).
    """
    import torch  # noqa: F401  (ensures torch present before transformers import)
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    params = req.params
    device = _resolve_device(str(params.get("device", "auto")))
    dtype = _resolve_dtype(params.get("dtype"), device)
    model_dir = str(params["checkpoint"])
    tokenizer_dir = str(params.get("tokenizer") or model_dir)
    attn_impl = params.get("attn_implementation")  # None → model default (sdpa)
    trust = bool(params.get("trust_remote_code", True))
    revision = params.get("revision")  # None → default branch

    common: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        common["revision"] = str(revision)

    # transformers >=5 renamed the from_pretrained dtype kwarg (torch_dtype → dtype).
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model_kwargs: dict[str, Any] = {dtype_key: dtype, **common}
    if attn_impl:
        model_kwargs["attn_implementation"] = str(attn_impl)
    loader = AutoModelForMaskedLM if for_logits else AutoModel
    model: Any = loader.from_pretrained(model_dir, **model_kwargs)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_dir, padding_side="right", **common
    )
    model.to(device)
    model.eval()

    strand_prefix = str(params.get("strand_prefix", "+"))
    marker_id: int | None = None
    if strand_prefix != "none":
        tok_str = _STRAND_TOKEN[strand_prefix]
        marker_id = int(tokenizer.convert_tokens_to_ids(tok_str))
        unk = getattr(tokenizer, "unk_token_id", None)
        if marker_id is None or (unk is not None and marker_id == unk):
            raise ValueError(
                f"gLM2 tokenizer has no strand marker {tok_str!r} in its vocab "
                f"(got id {marker_id}); cannot prepend it."
            )
    return model, tokenizer, device, marker_id


def _preprocess(seq: str, marker_id: int | None) -> str:
    """Lowercase the nucleotides (gLM2 reads uppercase as amino acids) and prepend the
    strand marker text when one is configured."""
    body = seq.lower()
    if marker_id is None:
        return body
    # The marker text is recovered from id inside _encode via the tokenizer; here we only
    # need the body — the marker id is prepended numerically after tokenization.
    return body


def _encode_ids(
    tokenizer: Any, seqs: list[str], marker_id: int | None, *, max_tokens: int
) -> list[list[int]]:
    """Per-sequence model-input ids: ``[marker_id] + nucleotide_ids`` (lowercased), capped.

    1 nt/token (no multi-nt merges) ⇒ one id per base. The leading marker (if any) counts
    toward ``max_tokens``; content is truncated to ``max_tokens - 1`` so the full input
    never exceeds the trained window.
    """
    out: list[list[int]] = []
    body_cap = int(max_tokens) - (1 if marker_id is not None else 0)
    for s in seqs:
        body = _preprocess(s, marker_id)
        ids = list(
            tokenizer(body, add_special_tokens=False, truncation=True, max_length=max(body_cap, 1))[
                "input_ids"
            ]
        )
        if marker_id is not None:
            ids = [marker_id, *ids]
        out.append(ids)
    return out


def _resolve_layers(layers_req: str | list[int] | None, n_states: int) -> list[int]:
    """Resolve ``last``/``all``/int/list to concrete ``hidden_states`` indices.

    gLM2 returns a ``hidden_states`` tuple of length ``n_states`` (== ``depth``); we index
    against the **actual** length, never a ``depth+1`` assumption. Negatives index from the end.
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
    marker_id: int | None,
    layers_req: str | list[int] | None,
    pool: str,
    batch_size: int,
    max_tokens: int,
    pool_include_special: bool,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Hidden-state embeddings (the layer convention), all layers from one forward pass.

    Pooling positions: content = the nucleotide tokens (everything after the leading strand
    marker). ``pool_include_special=False`` (default) pools over content only;
    ``True`` includes the marker. ``pool='marker'`` reads the strand-marker position.
    ``pool='none'`` returns the per-content-token vectors + unit-width ``token_spans``
    (marker excluded), in original-string coordinates.

    Batches are **length-bucketed** (``_batching.bucketed_batches``), so a batch no longer
    pays the longest sequence in the corpus for its shortest member. The bucketer returns
    INPUT indices in length order, so every result is scattered back by input index ``i``
    (never by position within the batch) — a mis-scatter here would silently label sequence
    A's embedding with sequence B's row.
    """
    import torch

    if pool not in ("mean", "max", "last", "marker", "none"):
        raise ValueError(
            f"gLM2 embed pool must be mean|max|last|marker|none; got {pool!r}. "
            "(gLM2 has no BOS/CLS; use 'marker' to read the strand-marker position.)"
        )
    if pool == "marker" and marker_id is None:
        raise ValueError("pool='marker' requires strand_prefix != 'none' (no marker to read).")

    id_seqs = _encode_ids(tokenizer, seqs, marker_id, max_tokens=max_tokens)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    marker_offset = 1 if marker_id is not None else 0

    payload: dict[str, np.ndarray] = {}
    pooled: list[np.ndarray | None] = [None] * len(id_seqs)
    sel: list[int] | None = None
    emb_dim = 0
    lengths = [len(ids) for ids in id_seqs]

    prog = BatchProgress(len(id_seqs), "glm2/embed", unit="seqs")
    n_done = 0
    for idx_batch in bucketed_batches(lengths, token_budget=token_budget, batch_size=batch_size):
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
        with torch.inference_mode():
            hs = model(
                input_ids=input_ids, attention_mask=attn, output_hidden_states=True
            ).hidden_states  # tuple len `depth`, each [B, S, H]
        if sel is None:
            sel = _resolve_layers(layers_req, len(hs))
            emb_dim = int(hs[-1].shape[-1])
        stacked = torch.stack([hs[li] for li in sel], dim=2)  # [B, S, n_sel, H]

        # Pooled ops build one [n_sel, H] vector per row on-device, then take ONE
        # device→host copy for the whole batch. pool='none' keeps its per-row copy (rows
        # have different lengths, so there is no single tensor to move).
        vecs: list[Any] = []
        vec_targets: list[int] = []
        for r, i in enumerate(idx_batch):
            k = lens[r]  # attended length (marker + content)
            n_content = max(k - marker_offset, 0)
            content = stacked[r, marker_offset : marker_offset + n_content]  # [n_content, n_sel, H]
            if pool == "none":
                payload[f"arr_{i}"] = content.float().detach().cpu().numpy().astype(np.float32)
                spans = np.asarray([(j, j + 1) for j in range(n_content)], dtype=np.int64).reshape(
                    -1, 2
                )
                payload[f"span_{i}"] = spans
                continue
            if pool == "marker":
                vecs.append(stacked[r, 0].float())
                vec_targets.append(i)
                continue
            sub = stacked[r, :k].float() if pool_include_special else content.float()
            if sub.shape[0] == 0:  # degenerate (empty content) → fall back to all attended
                sub = stacked[r, :k].float()
            if sub.shape[0] == 0:  # still empty → marker-only / empty input
                sub = stacked[r, :1].float()
            if pool == "mean":
                vecs.append(sub.mean(dim=0))
            elif pool == "max":
                vecs.append(sub.amax(dim=0))
            else:  # last
                vecs.append(sub[-1])
            vec_targets.append(i)
        if vecs:
            rows = torch.stack(vecs, dim=0).detach().cpu().numpy().astype(np.float32)
            for j, i in enumerate(vec_targets):
                pooled[i] = rows[j]  # keyed by INPUT index (bucketing reorders the batches)
        n_done += len(idx_batch)
        prog.update(n_done)
    prog.close()

    pooled_rows = [p for p in pooled if p is not None]
    if pool != "none":
        assert len(pooled_rows) == len(id_seqs), "bucketing dropped a sequence"

    if sel is None:  # no inputs
        sel = []
    payload["layer_ids"] = np.asarray(sel, dtype=np.int64)
    payload["embedding_dim"] = np.asarray(emb_dim, dtype=np.int64)
    if pool == "none":
        payload["n"] = np.asarray(len(id_seqs), dtype=np.int64)
    else:
        payload["arrays"] = (
            np.stack(pooled_rows, axis=0)
            if pooled_rows
            else np.zeros((0, len(sel), emb_dim), dtype=np.float32)
        )
    for key, arr in payload.items():
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(f"gLM2 embed produced non-finite values in {key} — refusing.")
    return payload


def _base_token_id(tokenizer: Any, base: str) -> int:
    """Token id for a single (lowercased) nucleotide base. Fails loudly on ``<unk>``.

    gLM2 reads nucleotides lowercase (uppercase ``ACGT`` are amino acids), so the WT/MUT
    bases from a variant string are lowercased before lookup. A base that maps to ``<unk>``
    is not a clean nucleotide token and would silently corrupt the logit read.
    """
    tid = int(tokenizer.convert_tokens_to_ids(base.lower()))
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        raise ValueError(
            f"gLM2 tokenizer maps base {base!r} to {tid} (<unk>={unk}); a masked-marginal "
            "LLR needs a clean single-nucleotide token id for both WT and MUT bases."
        )
    return tid


def _score_variant_llr(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    device: Any,
    marker_id: int | None,
    batch_size: int,
    max_tokens: int,
    token_budget: int | None = None,
) -> dict[str, np.ndarray]:
    """Masked-marginal LLR per variant: Σ_i (logit_mut_i − logit_wt_i) on the WT background.

    For every mutated position ``i`` of a variant, the reference is masked at the
    corresponding token index (``pos0 + marker_offset``) and a single forward pass yields
    the logit row; the contribution is ``logit[mut_id] − logit[wt_id]`` (the shared masked
    context cancels the softmax normalizer, so the raw logit difference *is* the
    log-probability ratio). Each mutated position is masked **independently** on the WT
    background (the additive single-site approximation) and summed.

    Efficiency: one masked forward per **unique (reference, position)** pair, cached and
    reused across every variant touching that pair, batched at ``batch_size``. The batching
    unit here is the **job** ``(ref_idx, token_index)``, not the variant — jobs are bucketed
    by their reference's token length, so a corpus mixing short and long references stops
    padding every job up to the longest one. Results are keyed by the job tuple in
    ``logit_rows``, which is order-free, so the bucketing cannot mis-attribute a logit row.
    """
    import torch

    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "gLM2 tokenizer has no mask_token_id; cannot compute masked-marginal LLR "
            "(the MLM [MASK] token is required to mask each mutated position)."
        )
    mask_id = int(mask_id)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    marker_offset = 1 if marker_id is not None else 0

    # Unique references → their (lowercased) model-input id sequences ([marker] + nt ids).
    ref_index: dict[str, int] = {}
    ref_strings: list[str] = []
    for it in items:
        ref = str(it["reference"]).lower()
        if ref not in ref_index:
            ref_index[ref] = len(ref_strings)
            ref_strings.append(ref)
    ref_ids = _encode_ids(tokenizer, ref_strings, marker_id, max_tokens=max_tokens)

    # Collect the unique (ref_idx, token_index) masked-forward jobs across all variants.
    job_set: dict[tuple[int, int], None] = {}
    for it in items:
        ridx = ref_index[str(it["reference"]).lower()]
        ids = ref_ids[ridx]
        for pos0, _wt, _mut in it["mutations"]:
            t = int(pos0) + marker_offset
            if t < 0 or t >= len(ids):
                raise ValueError(
                    f"masked position {pos0} (token index {t}) is out of range for a "
                    f"reference of {len(ids)} tokens (cap max_tokens={max_tokens}). The "
                    "task skips over-context references; this is the loud backstop."
                )
            job_set[(ridx, t)] = None
    jobs = list(job_set.keys())

    # Run the masked forwards in batches; store one logit row per job (mask position).
    # Bucket the JOBS by the length of the reference each one masks (that is the padded width
    # the job contributes), and read results back by the job tuple — never by batch position.
    logit_rows: dict[tuple[int, int], np.ndarray] = {}
    job_lengths = [len(ref_ids[r]) for (r, _t) in jobs]
    prog = BatchProgress(len(jobs), "glm2/llr", unit="masked-forwards")
    n_done = 0
    for idx_batch in bucketed_batches(
        job_lengths, token_budget=token_budget, batch_size=batch_size
    ):
        chunk = [jobs[j] for j in idx_batch]
        lens = [len(ref_ids[r]) for (r, _t) in chunk]
        max_full = max(lens) if chunk else 1
        input_ids = torch.full((len(chunk), max_full), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_full), dtype=torch.long, device=device)
        for i, (r, t) in enumerate(chunk):
            ids = list(ref_ids[r])
            ids[t] = mask_id  # mask exactly the target position on the WT background
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[i, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, S, V]
        # One device→host copy per batch: gather each row's own mask position first.
        pos = torch.tensor([t for (_r, t) in chunk], dtype=torch.long, device=logits.device)
        rows_cpu = (
            logits[torch.arange(len(chunk), device=logits.device), pos]  # [B, V]
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        if not np.isfinite(rows_cpu).all():
            raise ValueError("gLM2 produced non-finite logits — refusing to score.")
        for i, job in enumerate(chunk):
            logit_rows[job] = rows_cpu[i]
        n_done += len(chunk)
        prog.update(n_done)
    prog.close()

    # Cache base→id lookups (the WT/MUT bases are a tiny alphabet).
    base_id_cache: dict[str, int] = {}

    def base_id(b: str) -> int:
        key = b.lower()
        if key not in base_id_cache:
            base_id_cache[key] = _base_token_id(tokenizer, key)
        return base_id_cache[key]

    scores = np.empty(len(items), dtype=np.float64)
    for idx, it in enumerate(items):
        ridx = ref_index[str(it["reference"]).lower()]
        total = 0.0
        for pos0, wt, mut in it["mutations"]:
            t = int(pos0) + marker_offset
            row = logit_rows[(ridx, t)]
            total += float(row[base_id(str(mut))] - row[base_id(str(wt))])
        scores[idx] = total
    if not np.isfinite(scores).all():
        raise ValueError("gLM2 masked-marginal LLR produced non-finite scores — refusing.")
    return {"scores": scores}


def run(request_path: str) -> None:
    req = Request.read(request_path)
    if req.op not in ("tokenize", "embed", "score_variant_llr"):
        raise ValueError(
            f"gLM2 runner implements 'tokenize', 'embed', and 'score_variant_llr', got "
            f"{req.op!r}. (GLM2Adapter declares TOKENIZE + EMBEDDING + MASKED_MARGINAL_LLR; "
            "gLM2 is a masked-LM encoder, so the causal SEQUENCE_LOGLIKELIHOOD / "
            "PER_TOKEN_LOGPROBS / LOGPROB_EMBEDDING are N/A.)"
        )
    batch_size = int(req.params.get("batch_size", 8))
    max_tokens = int(req.params.get("max_tokens", 4096))
    token_budget = req.params.get("token_budget")
    token_budget = int(token_budget) if token_budget else None
    scratch_dir = os.path.dirname(request_path)

    if req.op == "score_variant_llr":
        _ids, items = read_variant_jsonl(req.input_path)
        model, tokenizer, device, marker_id = _load(req, for_logits=True)
        payload = _score_variant_llr(
            model,
            tokenizer,
            items,
            device=device,
            marker_id=marker_id,
            batch_size=batch_size,
            max_tokens=max_tokens,
            token_budget=token_budget,
        )
        write_arrays_npz(req.output_path, payload)
        Response(
            op=req.op,
            status="ok",
            output_path=req.output_path,
            meta={"n": len(items), "runner": "glm2", "device": str(device)},
        ).write(os.path.join(scratch_dir, "response.json"))
        return

    _ids, seqs = read_sequences_jsonl(req.input_path)
    model, tokenizer, device, marker_id = _load(req)
    meta: dict[str, object] = {"n": len(seqs), "runner": "glm2", "device": str(device)}

    if req.op == "tokenize":
        tokens = _encode_ids(tokenizer, seqs, marker_id, max_tokens=max_tokens)
        write_json_payload(req.output_path, tokens)
    else:  # embed
        payload = _embed(
            model,
            tokenizer,
            seqs,
            device=device,
            marker_id=marker_id,
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
        print("usage: python -m glmbench.runners.glm2_runner <request.json>", file=sys.stderr)
        return 2
    run(args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
