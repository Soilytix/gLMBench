"""The one place a runner turns "the last layer" into a tensor.

A ``transformers``-based runner that reads ``hidden_states[-1]`` and calls it the last layer
is right for some architectures and wrong for others. In a **pre-LN** decoder that tensor is
``norm(last_block_output)``, not the last block's output — the closing RMSNorm/LayerNorm has
already been applied. In a **post-LN** encoder the block's own output norm is *inside* the
block, so the same read is already the block output and nothing needs to change. The two cases are indistinguishable
from the tuple; they are distinguishable by measurement, which is what this module does.

Why it matters enough to hook a module rather than accept the difference: the normalised
read moved a DGEB score (``top_corr``) by −0.046 to −0.128 at the final layer, *growing with
depth*. RMSNorm divides each token by its own RMS, which a few enormous dimensions dominate;
every informative dimension is crushed toward zero and the pooled embedding collapses. It is
not a rounding difference.

Two entry points:

* :func:`last_block_capture` — a context manager that hooks the last transformer block so the
  runner can splice the **raw** block output into the hidden-states tuple. Used in the embed
  path of every HF runner.
* :func:`probe_last_layer` — the measurement: run one forward and report whether the hook
  and ``hidden_states[-1]`` are the same tensor. Its verdict is what licenses a runner to
  declare ``Readout.BLOCK_OUTPUT`` **without** a code change; an undocumented "we didn't
  change it" is not acceptable evidence.

This is a ``runners/*`` module — it may import torch. Core stays torch-free.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

# Attribute paths that hold the ordered transformer-block list, most specific first. The
# generic search below covers anything not named here; this list only makes the common cases
# unambiguous (a model can contain more than one ModuleList of identical children).
_KNOWN_BLOCK_PATHS: tuple[str, ...] = (
    "model.layers",  # Llama / Mistral (genomeocean)
    "layers",
    "encoder.layer",  # BERT-style encoders
    "bert.encoder.layer",
    "esm.encoder.layer",
    "model.encoder.layer",
    "backbone.blocks",  # StripedHyena (evo)
    "blocks",  # Evo 2 (arc)
    "model.blocks",
    "transformer.h",  # GPT-2 style
    "model.transformer.h",
)


def _resolve(model: Any, path: str) -> Any | None:
    obj: Any = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_blocks(
    model: Any, *, expected_depth: int | None = None, block_path: str | None = None
) -> tuple[str, Any]:
    """Return ``(dotted_path, ModuleList)`` for this model's transformer blocks.

    ``block_path`` names the list explicitly and skips the search — required for models whose
    depth axis is not one stack (NTv3's U-Net has three: ``core.conv_tower_blocks``,
    ``core.transformer_blocks``, ``core.deconv_tower_blocks``, and its final hidden state
    comes off the *deconv* tower).

    Otherwise: the known paths first, then a search over every ``nn.ModuleList`` whose
    children are all instances of one class. Raises rather than guessing when the search is
    ambiguous — a wrong block list would hook the wrong tensor and produce a plausible,
    silently incorrect embedding, the failure mode this module exists to remove.
    """
    import torch.nn as nn

    if block_path is not None:
        mod = _resolve(model, block_path)
        if mod is None or not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            raise ValueError(
                f"block_path {block_path!r} does not resolve to a non-empty ModuleList on "
                f"{type(model).__name__}."
            )
        return block_path, mod

    def _ok(mod: Any) -> bool:
        if not isinstance(mod, nn.ModuleList) or len(mod) == 0:
            return False
        if expected_depth is not None and len(mod) != expected_depth:
            return False
        return len({type(m) for m in mod}) == 1

    for path in _KNOWN_BLOCK_PATHS:
        mod = _resolve(model, path)
        if mod is not None and _ok(mod):
            return path, mod

    candidates = [(name, mod) for name, mod in model.named_modules() if _ok(mod)]
    if not candidates:
        raise ValueError(
            "could not locate the transformer-block ModuleList on this model "
            f"(tried {list(_KNOWN_BLOCK_PATHS)} and a generic search"
            + (f" for depth {expected_depth}" if expected_depth is not None else "")
            + "). Add its path to _KNOWN_BLOCK_PATHS rather than letting the runner fall back "
            "to hidden_states[-1], which is a different tensor in a pre-LN model."
        )
    if len(candidates) > 1:
        # Prefer the shallowest (outermost) one — nested ModuleLists inside a block would
        # otherwise win by accident. Still refuse if the tie is real.
        candidates.sort(key=lambda kv: (kv[0].count("."), kv[0]))
        if candidates[0][0].count(".") == candidates[1][0].count("."):
            raise ValueError(
                f"ambiguous transformer-block list: {[c[0] for c in candidates]}. Name the "
                "right one in _KNOWN_BLOCK_PATHS; guessing here would hook the wrong tensor."
            )
    return candidates[0]


def _tensor_of(obj: Any) -> Any:
    """Normalise a block's forward return to the hidden-state tensor.

    Blocks return a tensor (Llama), a tuple whose first element is one (BERT), or a dict
    (NTv3's ``SelfAttentionBlock`` returns ``{"embeddings", "attention_weights"}``). Anything
    else raises rather than being coerced — a hook that quietly returns the attention weights
    instead of the hidden state is exactly the class of silent substitution this module exists
    to stop.
    """
    if isinstance(obj, tuple):
        return obj[0]
    if isinstance(obj, dict):
        for key in ("embeddings", "hidden_states", "last_hidden_state"):
            if key in obj:
                return obj[key]
        raise ValueError(
            f"block forward returned a dict with keys {sorted(obj)} and none of them names "
            "the hidden state — add the key here rather than guessing."
        )
    return obj


@contextmanager
def last_block_capture(
    model: Any, *, expected_depth: int | None = None, block_path: str | None = None
) -> Any:
    """Hook the final transformer block; yield a dict that holds its **raw** output.

    Usage::

        with last_block_capture(model, expected_depth=n_blocks) as cap:
            hs = model(..., output_hidden_states=True).hidden_states
        raw_last = cap["out"]          # the block output, pre any model-level final norm

    The dict is filled during the forward and the hook is always removed, including on an
    exception. ``cap`` is empty if the hook never fired — the caller must treat that as an
    error, not fall back silently.
    """
    _path, blocks = find_blocks(model, expected_depth=expected_depth, block_path=block_path)
    cap: dict[str, Any] = {}
    handle = blocks[-1].register_forward_hook(
        lambda _m, _inp, out: cap.__setitem__("out", _tensor_of(out).detach())
    )
    try:
        yield cap
    finally:
        handle.remove()


def splice_raw_last(hidden_states: Any, cap: dict[str, Any], *, where: str) -> list[Any]:
    """Return ``hidden_states`` as a list with its final entry replaced by the raw block output.

    Loud if the hook did not fire: a missing capture means the block list we found is not on
    the forward path, and silently keeping ``hidden_states[-1]`` would put the post-final-norm
    tensor back on the board under an adapter version that declares the raw block output.
    """
    raw = cap.get("out")
    if raw is None:
        raise ValueError(
            f"{where}: the last-block forward hook never fired, so the raw block output is "
            "not available. Refusing to fall back to hidden_states[-1] — that is the "
            "post-final-norm tensor and would be recorded as if it were the block output."
        )
    out = list(hidden_states)
    if raw.shape != out[-1].shape:
        raise ValueError(
            f"{where}: hooked block output has shape {tuple(raw.shape)} but "
            f"hidden_states[-1] has {tuple(out[-1].shape)} — the hooked module is not the "
            "final block. Fix the block-path resolution; do not paper over it."
        )
    out[-1] = raw
    return out


def forward_hidden_states(
    model: Any, *, expected_depth: int | None, where: str, **forward_kwargs: Any
) -> list[Any]:
    """One forward → the hidden-states tuple **with the last entry on the target convention**.

    The three-line pattern every pre-LN HF runner needs, in one place: hook the last block,
    run the forward, splice the raw block output over ``hidden_states[-1]``. Entries
    ``0..L-1`` are untouched — they are already raw block outputs, which is why this fix moves
    only the *top* point of a sweep curve.

    Use it **only** where :func:`probe_last_layer` returned ``differs`` for the architecture
    (the adapter's ``readout_note`` records it). Where the probe said ``identical``, the tuple is
    already right and splicing a hook over it would be a no-op at best and, for a U-Net whose
    hook sees a pre-skip-add branch (NTv3), actively wrong.
    """
    import torch

    with last_block_capture(model, expected_depth=expected_depth) as cap:
        with torch.inference_mode():
            hs = model(output_hidden_states=True, **forward_kwargs).hidden_states
    return splice_raw_last(hs, cap, where=where)


def probe_last_layer(
    model: Any,
    input_ids: Any,
    *,
    attention_mask: Any | None = None,
    expected_depth: int | None = None,
    block_path: str | None = None,
    channels_first: bool = False,
    forward_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Measure ``hook(last_block)`` against ``hidden_states[-1]``.

    Returns a record with the measured deltas and a verdict:

    * ``identical`` — the two tensors agree to ``rtol`` (relative to the tensor's own RMS).
      This is the **post-LN** case: the block's output norm is inside the block, so the tuple
      already carries the block output and the runner needs no hook. The adapter may declare
      ``block_output`` with ``readout_evidence="probed"``, and *this record* is the evidence.
    * ``differs`` — the **pre-LN** case: ``hidden_states[-1]`` is ``norm(block_output)`` and
      the hook is the only way to reach the block output.

    ``rel_delta`` is normalised by the RMS of the raw tensor rather than by its max, because a
    pre-LN residual stream with massive activations has a max that a single dimension owns.
    """
    import torch

    kwargs: dict[str, Any] = {"output_hidden_states": True, **(forward_kwargs or {})}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with last_block_capture(
        model, expected_depth=expected_depth, block_path=block_path
    ) as cap:
        with torch.inference_mode():
            out = model(input_ids=input_ids, **kwargs)
        hs = out.hidden_states
    raw = cap.get("out")
    if raw is None:
        return {"verdict": "hook_did_not_fire", "n_hidden_states": len(hs)}
    if channels_first:
        # Conv towers carry [B, H, S]; the hidden-states tuple carries [B, S, H]. Same
        # tensor, different layout — transpose to compare like with like.
        raw = raw.transpose(1, 2)

    top = hs[-1]
    shapes_match = tuple(raw.shape) == tuple(top.shape)
    rec: dict[str, Any] = {
        "n_hidden_states": len(hs),
        "expected_depth": expected_depth,
        "raw_shape": list(raw.shape),
        "top_shape": list(top.shape),
        "shapes_match": shapes_match,
    }
    if not shapes_match:
        rec["verdict"] = "shape_mismatch"
        return rec

    a = raw.float()
    b = top.float()
    rms = float(a.pow(2).mean().sqrt())
    max_abs = float((a - b).abs().max())
    rec.update(
        {
            "raw_rms": rms,
            "top_rms": float(b.pow(2).mean().sqrt()),
            "max_abs_delta": max_abs,
            "rel_delta": (max_abs / rms) if rms > 0 else float("inf"),
            "cosine": float(
                torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
            ),
        }
    )
    rec["verdict"] = "identical" if rec["rel_delta"] <= rtol else "differs"
    return rec
