"""``LoamConfig`` — the HuggingFace config for an exported LOAM checkpoint.

**This file is copied verbatim into every export directory** (together with
``modeling_loam.py``) so the artifact loads under ``trust_remote_code=True`` on a
box that has only ``transformers`` installed. It therefore must not import
anything from the ``loam`` package.

Field names follow HF conventions on the left, LOAM's ``ModelConfig`` on the
right::

    vocab_size                <- vocab_size (derived from the tokenizer)
    hidden_size               <- d_model
    num_hidden_layers         <- n_layers
    num_attention_heads       <- n_heads
    num_key_value_heads       <- n_kv_heads
    head_dim                  <- head_dim (explicit: 28 heads x 64 != a power of
                                 two, and some HF paths assume hidden/heads)
    intermediate_size         <- d_ffn
    max_position_embeddings   <- max_seq_len
    rope_theta                <- rope_theta
    rms_norm_eps              <- rms_norm_eps
    qk_norm                   <- qk_norm
    use_bias                  <- use_bias
    tie_word_embeddings       <- tie_embeddings
    norm_placement            <- norm_placement   ("pre" | "peri")
    gpas_enabled / gpas_init  <- gpas.enabled / gpas.init

``norm_placement`` and ``gpas_enabled`` change the *trained function*, not just
the parameter count: a checkpoint is not portable across them and the native
loader hard-errors on a mismatch. They are ordinary config fields here for the
same reason they are ordinary config fields in LOAM — so that the artifact
records what the model actually is.

The ``loam_*`` fields are provenance. They are carried through
``config.json`` and never read by the forward pass.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

__all__ = ["LoamConfig"]

# Bumped when the exported *format* changes in a way a consumer could observe.
#   1.0.0 — forward-only export.
#   1.1.0 — generation: KV cache, padded batches, `.generate()`.
#           `generation_config.json` carries DNA-sane sampling defaults and turns
#           the cache on, so a 1.0.0 directory and a 1.1.0 directory built from the
#           same checkpoint differ in more than the modeling code.
LOAM_HF_EXPORT_VERSION = "1.1.0"


class LoamConfig(PretrainedConfig):
    """Configuration for :class:`LoamModel` / :class:`LoamForCausalLM`."""

    model_type = "loam"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 58,
        hidden_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        use_bias: bool = False,
        norm_placement: str = "pre",
        gpas_enabled: bool = False,
        gpas_init: float = 0.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        use_cache: bool = False,
        # --- provenance (carried, never used by the forward) ---------------
        loam_config_hash: str | None = None,
        loam_tokenizer_hash: str | None = None,
        loam_checkpoint_step: int | None = None,
        loam_checkpoint_id: str | None = None,
        loam_token_count: int | None = None,
        loam_source_config: str | None = None,
        loam_export_version: str = LOAM_HF_EXPORT_VERSION,
        **kwargs,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )

        if head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"head_dim was not given and hidden_size ({self.hidden_size}) is not "
                    f"divisible by num_attention_heads ({self.num_attention_heads}); "
                    "set head_dim explicitly."
                )
            head_dim = self.hidden_size // self.num_attention_heads
        else:
            head_dim = int(head_dim)
            if head_dim * self.num_attention_heads != self.hidden_size:
                raise ValueError(
                    f"head_dim ({head_dim}) * num_attention_heads "
                    f"({self.num_attention_heads}) = {head_dim * self.num_attention_heads} "
                    f"!= hidden_size ({self.hidden_size}). LOAM's q_proj/o_proj are "
                    "square, so these must agree."
                )
        self.head_dim = head_dim
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE; got {self.head_dim}")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        self.intermediate_size = int(
            _default_d_ffn(self.hidden_size) if intermediate_size is None else intermediate_size
        )
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.qk_norm = bool(qk_norm)
        self.use_bias = bool(use_bias)

        if norm_placement not in ("pre", "peri"):
            raise ValueError(
                f"norm_placement must be 'pre' or 'peri'; got {norm_placement!r}"
            )
        self.norm_placement = norm_placement
        self.gpas_enabled = bool(gpas_enabled)
        self.gpas_init = float(gpas_init)
        self.initializer_range = float(initializer_range)

        # DEFAULT-OFF ON PURPOSE, and not a leftover from the forward-only export.
        # HF's convention (`LlamaConfig.use_cache=True`) means every plain forward
        # allocates and fills a KV cache. LOAM's actual consumers — gLMBench scoring,
        # the mech-interp capture, the SAE trainer — run forward-only at 8192 context
        # in large batches, where that cache is tens of GB of VRAM bought for nothing
        # and thrown away. `generation_config.json` sets `use_cache: true`, so
        # `.generate()` and `pipeline()` are cached regardless; an explicit
        # `use_cache=True` on a forward still works. The knob is on where it is used.
        self.use_cache = bool(use_cache)

        self.loam_config_hash = loam_config_hash
        self.loam_tokenizer_hash = loam_tokenizer_hash
        self.loam_checkpoint_step = loam_checkpoint_step
        self.loam_checkpoint_id = loam_checkpoint_id
        self.loam_token_count = loam_token_count
        self.loam_source_config = loam_source_config
        self.loam_export_version = loam_export_version

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def _default_d_ffn(d_model: int) -> int:
    """``8/3 * d_model`` rounded up to a multiple of 256 (the LLaMA-2 rule).

    Mirrors ``loam.models.loam._default_d_ffn``. Duplicated rather than imported
    because this file must stand alone inside an export directory.
    """
    raw = int(round(8 * d_model / 3))
    return ((raw + 255) // 256) * 256
