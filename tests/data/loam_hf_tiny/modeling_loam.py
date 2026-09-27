"""``LoamModel`` / ``LoamForCausalLM`` — a self-contained HF port of the LOAM backbone.

**This file is copied verbatim into every export directory** so the artifact
loads under ``trust_remote_code=True`` on a box that has only ``transformers``
installed. It must therefore **never** import from the ``loam`` package. That
self-containment is also a drift risk, which a parity test catches: the port
is checked against the live ``loam.models.LOAMModel`` for bitwise-equal logits
across the whole ``{pre,peri} x {gpas} x {qk_norm} x {MHA,GQA} x {tied}``
variant matrix, on every CI run.

Three LOAM features have no stock HF equivalent, which is why this is a custom
architecture rather than a re-badged ``LlamaForCausalLM``:

1. **Peri-LN** (``norm_placement="peri"``, arXiv:2502.02732) —
   ``y = x + Norm(Module(Norm(x)))``. A Llama block has no normalization on the
   branch output before the residual add. Silently dropping those ``2*L`` norms
   produces a model that runs, scores plausibly, and is wrong.
2. **qk-norm applied AFTER RoPE.** Qwen3 — the one stock family with qk-norm —
   normalizes *before* RoPE. Same parameter shapes, different function. This is
   the single most likely silent bug in the port, and a negative control tests it.
3. **GPAS** (``x - SiLU(alpha) * sg(x)``, arXiv:2506.22049) has no stock analogue.

Two further details are load-bearing and easy to "clean up" into a bug:

- ``LoamRMSNorm`` upcasts to fp32, multiplies by the gain **while still in fp32**,
  and only then downcasts. HF's ``LlamaRMSNorm`` downcasts first
  (``self.weight * hidden_states.to(input_dtype)``). The two agree in fp32 and
  differ measurably in bf16.
- RoPE is the LLaMA **half-rotation** convention (``cat(freqs, freqs)`` +
  ``rotate_half``), not the GPT-NeoX interleaved one.

**Scope.** The export supports autoregressive generation: a KV cache,
padded batches in either direction, and ``.generate()`` through ``GenerationMixin``.
It still does *not* do intra-document packing masks, sequence parallelism or FIM,
and each of those is refused loudly rather than half-supported — a
quietly-ignored ``attention_mask`` would corrupt every downstream number without a
single warning.

**The generation contract, stated before anyone edits it**:

- The cache holds K **after RoPE and after qk-norm**, and V **after ``v_proj`` but
  BEFORE ``repeat_kv``**; both are repeated on read, every step. Caching pre-RoPE K
  and re-rotating at read time is also correct *if* the offset is right, and is the
  same family of mistake as qk-norm-before-RoPE: it yields a model that generates
  fluent, plausible, wrong DNA.
- RoPE positions come from ``position_ids``, which come from ``cache_position``
  under a cache and from ``cumsum(attention_mask) - 1`` under padding — never from
  a bare ``arange`` over the *current* chunk.
- ``F.scaled_dot_product_attention(..., is_causal=True)`` aligns its mask to the
  TOP-LEFT, so at ``q_len=1, k_len=N`` it masks away nearly the whole cache and
  attends to position 0. It is therefore correct ONLY when ``q_len == k_len``.
  Every other shape builds an explicit mask.

The no-cache, no-mask forward is kept on a separate branch that is *bitwise* the
code the parity tests measured (``attn_mask=None, is_causal=True``), so adding generation
cannot perturb the parity record.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_loam import LoamConfig

logger = logging.get_logger(__name__)

try:  # transformers v5: initializers that no-op on already-loaded parameters.
    from transformers import initialization as _init
except ImportError:  # pragma: no cover - transformers v4
    from torch.nn import init as _init  # type: ignore[no-redef]

__all__ = [
    "LoamPreTrainedModel",
    "LoamModel",
    "LoamForCausalLM",
    "LoamRMSNorm",
    "LoamRotaryEmbedding",
    "LoamAttention",
    "LoamMLP",
    "LoamDecoderLayer",
    "LoamGPASGate",
]

_SUPPORTED_ATTN = ("sdpa", "eager")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class LoamRMSNorm(nn.Module):
    """Port of ``loam.models.blocks.norm.LoamRMSNorm``.

    Operation order is part of the contract: upcast -> normalize -> **multiply by
    the gain in fp32** -> downcast. Reordering the last two steps changes bf16
    results measurably.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x32).to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LoamRotaryEmbedding(nn.Module):
    """Port of ``loam.models.blocks.positional.RotaryEmbedding``.

    Tables are precomputed to ``max_seq_len`` in fp32 and registered as
    **non-persistent** buffers — they are absent from the checkpoint by design
    and rebuilt here from ``rope_theta`` + ``max_position_embeddings``.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE; got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq, cos, sin = self._tables(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _tables(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        # Half-rotation layout: duplicate the halves. NOT GPT-NeoX interleaving.
        emb = torch.cat((freqs, freqs), dim=-1)
        return inv_freq, emb.cos(), emb.sin()

    @torch.no_grad()
    def reset_tables(self) -> None:
        """Recompute the tables in place.

        ``from_pretrained`` builds the module with weight initialization
        suppressed and then fills parameters from the checkpoint. These buffers
        are ``persistent=False``, so they are in no checkpoint and nothing fills
        them — without this they arrive as whatever was in the allocation, and
        every logit comes out NaN. HF re-initializes its own rotary buffers from
        ``_init_weights`` for exactly this reason; so does :class:`LoamPreTrainedModel`.
        """
        inv_freq, cos, sin = self._tables(self.inv_freq.device)
        for name, value in (("inv_freq", inv_freq), ("cos_cached", cos), ("sin_cached", sin)):
            buf = getattr(self, name)
            if buf.is_meta:  # pragma: no cover - device_map / meta-init loads
                self.register_buffer(name, value.to(buf.dtype), persistent=False)
            else:
                buf.copy_(value.to(buf.dtype))

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len {seq_len} exceeds RoPE cache ({self.max_seq_len})."
            )
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: ``(-x2, x1)``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. ``q,k``: ``[B,H,S,D]``; ``cos,sin``: ``[max_seq,D]``.

    The tables are cast to the query dtype rather than left in fp32. Native LOAM
    holds them as ordinary float buffers, so ``model.to(bf16)`` casts them and the
    rotation runs in bf16; keeping them fp32 here would both diverge from that and
    promote q/k to fp32 while v stayed bf16 — which SDPA rejects outright. The
    cast is exact: ``bf16(cos_fp32)`` is precisely what the native model holds.
    """
    cos = cos[position_ids].unsqueeze(1).to(q.dtype)  # [B, 1, S, D]
    sin = sin[position_ids].unsqueeze(1).to(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, n_kv, S, D] -> [B, n_kv * n_rep, S, D]``."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    x = x.unsqueeze(2).expand(B, n_kv, n_rep, S, D)
    return x.reshape(B, n_kv * n_rep, S, D)


class LoamGPASGate(nn.Module):
    """``x - SiLU(alpha) * sg(x)`` with one learnable scalar per layer.

    ``SiLU(0) == 0`` exactly, so at ``init=0.0`` this is bit-identical to the
    identity — not merely close.
    """

    def __init__(self, init: float = 0.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - F.silu(self.alpha) * x.detach()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha.detach().flatten()[0].item():.4g}"


class LoamMLP(nn.Module):
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LoamAttention(nn.Module):
    """Grouped-query attention. **RoPE first, then qk-norm** (see module docstring)."""

    def __init__(
        self,
        config: LoamConfig,
        attn_implementation: str = "sdpa",
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if attn_implementation not in _SUPPORTED_ATTN:
            raise ValueError(
                f"attn_implementation={attn_implementation!r} is not supported by the "
                f"LOAM export; expected one of {list(_SUPPORTED_ATTN)}. The native "
                "LOAM forward calls F.scaled_dot_product_attention, so 'sdpa' is the "
                "parity-faithful choice and 'eager' exists as its numerical cross-check."
            )
        self.attn_implementation = attn_implementation
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        bias = config.use_bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = LoamRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LoamRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output_attentions and self.attn_implementation != "eager":
            raise NotImplementedError(
                "output_attentions=True requires attn_implementation='eager'. The sdpa "
                "path calls the fused F.scaled_dot_product_attention, which never "
                "materializes the attention probabilities — there is nothing to return. "
                "Load with attn_implementation='eager' (gated equivalent to sdpa by #H18, "
                "and slower)."
            )
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(hidden_states)
            .view(B, S, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, S, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # THE TRAP: qk-norm goes here, AFTER RoPE. Qwen3 does it before.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # The cache holds K post-RoPE AND post-qk-norm, V pre-`repeat_kv`. Both
        # halves are load-bearing and both are tested.
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.kv_groups)
        v = repeat_kv(v, self.kv_groups)

        # `is_causal=True` aligns SDPA's mask top-left, which is the right answer
        # only when the query and key lengths agree. See the module docstring.
        square = q.shape[-2] == k.shape[-2]
        if self.attn_implementation == "sdpa":
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=attn_mask is None and square,
                scale=self.scaling,
            )
        else:
            attn, probs = _eager_attention(q, k, v, self.scaling, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(attn)
        return (out, probs) if output_attentions else out


def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-written softmax attention — the numerical cross-check for sdpa.

    Returns ``(output, probabilities)``. The probabilities are the *only* reason
    this path can serve ``output_attentions=True`` at all; sdpa never materializes
    them.

    With ``attn_mask=None`` this reproduces the pre-generation behaviour exactly:
    a square lower-triangular mask built here. With a mask it honours that mask
    instead, which is how the padded and cached paths reach the ``eager`` kernel.
    """
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    if attn_mask is None:
        Lq, Lk = q.shape[-2], k.shape[-2]
        if Lq != Lk:
            raise ValueError(
                f"_eager_attention got q_len={Lq} != k_len={Lk} with no mask; the "
                "implicit causal mask is only defined for a square attention."
            )
        attn_mask = torch.ones((Lq, Lk), dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v), probs


class LoamDecoderLayer(nn.Module):
    """LLaMA-style block with the Peri-LN and GPAS variants of v1.5.

    ``attn_out_norm`` / ``mlp_out_norm`` are registered **only** under
    ``norm_placement="peri"``, and ``gpas`` only when GPAS is on — mirroring the
    native conditional registration. An unconditionally-registered no-op norm
    would still add ``2*L*d_model`` parameters and break state-dict
    compatibility with every existing checkpoint.
    """

    def __init__(
        self,
        config: LoamConfig,
        attn_implementation: str = "sdpa",
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.norm_placement = config.norm_placement

        self.input_layernorm = LoamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LoamAttention(
            config, attn_implementation=attn_implementation, layer_idx=layer_idx
        )
        self.post_attention_layernorm = LoamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LoamMLP(config.hidden_size, config.intermediate_size, bias=config.use_bias)

        if config.norm_placement == "peri":
            self.attn_out_norm = LoamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_out_norm = LoamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gpas = LoamGPASGate(init=config.gpas_init) if config.gpas_enabled else None

    # Set by `PreTrainedModel._set_gradient_checkpointing`, which walks the module
    # tree and writes to any module that ALREADY has this attribute — declaring it
    # here is what opts the block in. Declared rather than inherited from
    # `transformers.modeling_layers.GradientCheckpointingLayer` on purpose: that
    # class is recent and its location has moved, and this file has to import
    # cleanly on transformers 4.x as well.
    gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        peri = self.norm_placement == "peri"

        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x, cos, sin, position_ids,
            attn_mask=attn_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )
        if output_attentions:
            x, attn_probs = x
        if peri:
            x = self.attn_out_norm(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        if peri:
            x = self.mlp_out_norm(x)
        x = x + residual

        if self.gpas is not None:
            x = self.gpas(x)
        # A bare tensor unless probabilities were asked for. Returning a 1-tuple
        # unconditionally, as stock HF blocks do, would change what every forward
        # hook on `model.layers.{i}` captures — the residual-stream taps
        # and the every-activation parity test (which asserts an *exact*
        # module-output count) both read those hooks.
        return (x, attn_probs) if output_attentions else x


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LoamPreTrainedModel(PreTrainedModel):
    config_class = LoamConfig
    config: LoamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LoamDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize a module — via the **flag-aware** initializers, not raw
        ``tensor.normal_``.

        transformers v5 runs this pass *after* the checkpoint is loaded, to fill
        whatever the checkpoint did not cover. The initializers in
        ``transformers.initialization`` no-op on any tensor already marked
        ``_is_hf_initialized``; a raw ``module.weight.data.normal_()`` does not
        consult the flag and therefore silently overwrites every loaded weight
        with fresh noise. That failure is invisible — the model loads with zero
        missing keys and produces plausible-looking garbage.
        """
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            _init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                _init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, LoamRMSNorm):
            _init.ones_(module.weight)
        elif isinstance(module, LoamGPASGate):
            _init.constant_(module.alpha, float(getattr(self.config, "gpas_init", 0.0)))
        elif isinstance(module, LoamRotaryEmbedding):
            # Non-persistent buffers: never in a checkpoint, always ours to fill.
            module.reset_tables()

    # -- input guards: refuse what this export deliberately does not do --

    def _resolve_attn_implementation(self) -> str:
        impl = getattr(self.config, "_attn_implementation", None) or "sdpa"
        if impl not in _SUPPORTED_ATTN:
            raise ValueError(
                f"attn_implementation={impl!r} is not supported by the LOAM export; "
                f"expected one of {list(_SUPPORTED_ATTN)}."
            )
        return impl

    def _check_forward_inputs(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> None:
        if attention_mask is not None:
            _check_padding_mask(attention_mask)
        limit = self.config.max_position_embeddings
        max_pos = int(position_ids.max())
        if max_pos >= limit:
            raise ValueError(
                f"position {max_pos} reaches or exceeds max_position_embeddings={limit}, "
                "which is the size of the precomputed RoPE cos/sin cache. Split the "
                "input into windows of at most that length, or stop generating there."
            )


def _check_padding_mask(attention_mask: torch.Tensor) -> None:
    """Validate a 2-D padding mask. Left, right and interior padding are all honoured.

    **This changed in export version 1.1.0.** The forward-only export
    took *no* mask into the kernel and therefore had to refuse anything but right
    padding by name: real tokens would otherwise attend to pads and sit at shifted
    RoPE positions. Batched ``.generate()`` conventionally LEFT-pads, so that
    refusal had to go — but the fix is to honour the mask, not to relax the check.
    The mask now reaches SDPA as an explicit 4-D boolean and ``position_ids`` are
    derived from it (``cumsum - 1``), so every real token sits where it would sit
    in an unbatched forward, which a test checks.

    What is still refused: a pre-expanded 4-D mask (the caller is doing the model's
    job and the two conventions would silently disagree), and a row with no real
    tokens at all (nothing to generate from, and its softmax has no finite entry).

    Outputs AT pad positions remain garbage — finite, because the diagonal is kept
    unmasked so a fully-masked query row cannot produce ``NaN``, but meaningless.
    Every LOAM scoring path already drops them.
    """
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2-D [batch, seq]; got shape {tuple(attention_mask.shape)}. "
            "The LOAM export builds its own 4-D causal mask and does not accept a "
            "pre-expanded one."
        )
    m = attention_mask.to(torch.bool)
    if not bool(m.any(dim=1).all()):
        bad = int(torch.nonzero(~m.any(dim=1))[0, 0])
        raise ValueError(
            f"attention_mask row {bad} is entirely padding: it contains no real token. "
            "Every row must have at least one."
        )


def build_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return the 4-D boolean ``[B, 1, q_len, kv_len]`` mask, or ``None``.

    ``None`` means "let SDPA build the square causal mask itself" and is returned
    for exactly one case — no padding and ``q_len == kv_len`` — which is the
    unpadded, uncached forward the parity tests measured. Keeping that
    case on a distinct branch is deliberate: it stays bitwise the code those tests
    ran against, so generation support cannot perturb the parity record.

    Otherwise the mask is ``causal & padding``:

    - **causal** — a query at absolute position ``kv_len - q_len + i`` may see keys
      ``0..kv_len - q_len + i``. The offset is what makes a 1-token decode step
      attend to the whole cache instead of to position 0.
    - **padding** — broadcast over queries from the 2-D mask.

    The diagonal is forced on. It is already causal-legal for every real query, so
    this changes nothing for them; it only keeps a *pad* query row from having an
    all-``-inf`` softmax, which would come back ``NaN`` and pollute the batch's
    finiteness checks. Those rows are garbage either way and callers drop them.
    """
    if attention_mask is None and q_len == kv_len:
        return None

    offset = kv_len - q_len
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset  # [q_len, 1]
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)  # [1, kv_len]
    mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

    if attention_mask is not None:
        pad = attention_mask.to(torch.bool)[:, None, None, :]  # [B, 1, 1, kv_len]
        mask = mask & pad
        diag = (k_pos == q_pos).unsqueeze(0).unsqueeze(0)
        mask = mask | diag

    return mask.expand(
        attention_mask.shape[0] if attention_mask is not None else 1, 1, q_len, kv_len
    )


class LoamModel(LoamPreTrainedModel):
    """``embed_tokens -> L x LoamDecoderLayer -> norm``."""

    def __init__(self, config: LoamConfig) -> None:
        super().__init__(config)
        impl = self._resolve_attn_implementation()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LoamDecoderLayer(config, attn_implementation=impl, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LoamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = LoamRotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = True if return_dict is None else bool(return_dict)
        use_cache = self.config.use_cache if use_cache is None else bool(use_cache)

        # Gradient checkpointing discards the activations it will recompute, and a
        # KV cache is precisely a set of activations kept across calls. Silently
        # keeping both would either leak memory or feed the recomputation a cache
        # that has already moved on. HF resolves it the same way, loudly.
        checkpointing = self.training and any(
            layer.gradient_checkpointing for layer in self.layers
        )
        if checkpointing and use_cache:
            logger.warning_once(
                "use_cache=True is incompatible with gradient checkpointing; setting "
                "use_cache=False. Nothing is silently dropped — the forward is "
                "unchanged, only the cache is not built."
            )
            use_cache = False

        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, S = x.shape[0], x.shape[1]

        if attention_mask is not None:
            _check_padding_mask(attention_mask)
            # An all-ones mask carries no information, and the tokenizer emits one
            # on every single-sequence call. Dropping it here keeps that overwhelmingly
            # common case on the `is_causal=True` fast path that the parity tests
            # measured, instead of routing it through mask construction for nothing.
            if bool(attention_mask.all()):
                attention_mask = None

        if past_key_values is None and use_cache:
            past_key_values = DynamicCache(config=self.config)
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_len + S

        if cache_position is None:
            cache_position = torch.arange(past_len, kv_len, device=x.device)

        if position_ids is None:
            if attention_mask is None:
                position_ids = cache_position.unsqueeze(0).expand(B, S)
            else:
                # Left padding shifts every real token; `cumsum - 1` puts the first
                # real token of each row at position 0, which is what an unbatched
                # forward would do. Pads land on a clamped duplicate and are dropped.
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)[:, -S:]

        self._check_forward_inputs(attention_mask, position_ids)

        # `self.rotary(n)` is called with exactly `S` on the unpadded, uncached path,
        # so the hooked (cos, sin) tensors keep the shapes the parity test compares.
        cos, sin = self.rotary(int(position_ids.max()) + 1)
        attn_mask = build_attention_mask(
            attention_mask, q_len=S, kv_len=kv_len, device=x.device
        )

        all_hidden: list[torch.Tensor] = []
        all_attn: list[torch.Tensor] = []
        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(x)
            if layer.gradient_checkpointing and self.training:
                # `layer.__call__`, not `layer.forward`: __call__ is what runs the
                # module's forward hooks, and the mech-interp capture reads exactly
                # those. Positional hidden state, per the reentrant-checkpoint rule.
                x = layer._gradient_checkpointing_func(
                    layer.__call__, x, cos, sin, position_ids, attn_mask,
                    past_key_values, output_attentions,
                )
            else:
                x = layer(
                    x, cos, sin, position_ids,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                x, probs = x
                all_attn.append(probs)
        x = self.norm(x)
        if output_hidden_states:
            # HF convention: the LAST element is POST-final-norm, not the last
            # block's residual. LOAM's residual stream carries ~1e7-scale
            # activations, so the two are very different objects — read the
            # block-module outputs (hooks on `model.layers.{i}`) for residuals.
            all_hidden.append(x)

        hidden_states = tuple(all_hidden) if output_hidden_states else None
        attentions = tuple(all_attn) if output_attentions else None
        cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v for v in (x, cache, hidden_states, attentions) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=cache,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class LoamForCausalLM(LoamPreTrainedModel, GenerationMixin):
    """``LoamModel`` + a tied-or-untied ``lm_head`` and the standard causal LM loss.

    ``GenerationMixin`` must come AFTER ``LoamPreTrainedModel`` in the MRO (HF
    raises otherwise) and is what makes ``can_generate()`` true; without it
    ``.generate()`` does not exist at all, which is what the forward-only export
    shipped. See the module docstring for the cache contract it relies on.
    """

    # transformers v5 spells this as target -> source. Only consulted when
    # `config.tie_word_embeddings` is true, which mirrors LOAM's `tie_embeddings`.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LoamConfig) -> None:
        super().__init__(config)
        self.model = LoamModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = True if return_dict is None else bool(return_dict)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,)
            if outputs.past_key_values is not None:
                out = out + (outputs.past_key_values,)
            if outputs.hidden_states is not None:
                out = out + (outputs.hidden_states,)
            if outputs.attentions is not None:
                out = out + (outputs.attentions,)
            return ((loss,) + out) if loss is not None else out
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
