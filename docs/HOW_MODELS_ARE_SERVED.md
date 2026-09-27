# How models are served to the benchmark

**Audience:** anyone *running* a model through gLMBench who wants to know which environment,
checkpoint and conventions each one needs, and why two models' "layer 12" are not the same
kind of number. For *adding a new* model see [ADD_A_MODEL.md](ADD_A_MODEL.md); for setting up
the environments see [INSTALL.md](INSTALL.md).

**The models that ship:**

| Model | adapter | Specs | One-line summary |
| --- | --- | --- | --- |
| **LOAM** 25M / 100M / 340M / 624M | `loam-hf` | `specs/loam/LOAM-*.yaml` | Decoder-only causal LM, 1 nt/token, released on the Hub as `Soilytix/LOAM-*` with its own modeling code. |
| **Evo2-7B** | `evo2-arc` | `specs/external/evo2-arc-7b-resid.yaml` | StripedHyena 2 causal LM (Arc Institute), run through the `evo2` pip package in its own env. Named-layer taps. |
| **Evo 1.5** (7B, 8k) | `evo` | `specs/external/evo-1.5-8k.yaml` | StripedHyena causal LM, byte-level (1 nt/token), Hugging Face `trust_remote_code`. Needs flash-attn. |
| **GenomeOcean** 100M / 500M | `genomeocean` | `specs/external/genomeocean-{100M,500M}.yaml` | Stock `MistralForCausalLM` with a BPE tokenizer (variable nt/token). |
| **gLM2** 150M / 650M | `glm2` | `specs/external/glm2-{150M,650M}.yaml` | Mixed-modality masked-LM **encoder** (TattaBio). Embeddings + masked-marginal variant scoring. |
| **ProkBERT** mini / mini-c | `prokbert` | `specs/external/prokbert-{mini,mini-c}.yaml` | BERT-style masked-LM **encoder** with an overlapping k-mer tokenizer. |
| **NTv3** 100M | `ntv3` | `specs/external/ntv3-100M.yaml` | Nucleotide Transformer v3, a **U-Net** masked-LM encoder. Inputs pad to a multiple of 128. |
| echo | `echo` | `specs/echo-{full,score-only}.yaml` | Model-free test double; the smoke test and the simplest adapter template. |

Not to be confused: **Evo2-7B** (`evo2-arc`) and **Evo 1.5** (`evo`) are different models,
different adapters and different environments.

If you have hit `ModuleNotFoundError: No module named 'glmbench'` from a runner, a
`trust_remote_code` crash, or a sweep whose layer ids do not line up with another model's,
this doc is for you.

---

## 0. Which tensor is "a layer"? (read this before comparing two models)

Every adapter **declares** which tensor `embed(layers=...)` returns. The declaration is on the
adapter class (`readout`, `readout_evidence`, `readout_note`), is echoed into `describe()`, and
is written into every result record, so a stored number always says what it was read from.

The convention: **layer `i` is the output of transformer block `i`**, the raw residual stream
after the block's adds, before any model-level final normalisation. Layer 0 is the embedding
output. For an `L`-block model the ids run `0..L`, and `layers="last"` is the raw output of
the last block, **not** `norm(last_block_output)`.

This matters because a pre-LN model's `hidden_states[-1]` is usually the *final-normed*
state. A per-token RMSNorm/LayerNorm divides each token by its own RMS; when a few residual
dimensions are very large, that crushes every informative dimension, and a pooled embedding
of the normed state can probe very differently from the raw one. It is not a rounding
difference, so every runner below returns the raw block output and says how.

| adapter | readout | how the runner gets there | evidence |
|---|---|---|---|
| `loam-hf` | `block_output` | forward hooks on the token embedding and on every block; the final RMSNorm is not applied to embeddings | by construction |
| `evo2-arc` | **set by the spec** | `blocks.N` → `block_output` (the shipped spec); `blocks.N.pre_norm` → `normed_block_output`; `blocks.N.mlp.l3` → `branch_tap` | by construction |
| `evo` | `block_output` | the HF wrapper returns `hidden_states=None`, so the runner hooks the backbone blocks directly | by construction |
| `genomeocean` | `block_output` | hooks the last block: in this pre-LN Mistral, `hidden_states[-1]` is the final-normed state | probed |
| `prokbert` | `block_output` | hooks the last block: ProkBERT's encoder is pre-LN with a final LayerNorm outside the blocks | probed |
| `glm2` | `block_output` | no hook needed: gLM2's closing norm lives in `lm_head`, which the `AutoModel` load drops | probed |
| `ntv3` | `block_output` | no hook: the hidden-states tuple already holds the state *after* each U-Net skip add, while a hook on a deconv block would see the branch *before* it | probed |
| `echo` | `block_output` | synthesizes its own layer axis | by construction |

`by_construction` means the runner builds the tensor itself (an explicit hook or a named tap).
`probed` means a hook on the last block was measured against `hidden_states[-1]` on the real
checkpoint with `glmbench.runners._readout.probe_last_layer`, which returns `identical` (the
tuple already carries the block output) or `differs` (it carries the normed state). Two of
these went against the rule of thumb: ProkBERT *looks* like a post-LN BERT and needed the hook;
NTv3 looked like it needed one, and there the hook would have been wrong. Measure; do not
assume.

The four `readout` values are three different objects and one qualifier: `block_output` (the
raw trunk; the target), `normed_block_output` (the trunk with a per-tap norm),
`post_final_norm` (the trunk with the model's closing norm) and `branch_tap` (an attention or
MLP branch output *before* the residual add: a proposed edit to the stream, not a state of it).

**Layer ids are not comparable across models.** Every adapter above returns raw block outputs,
but the layer *axis* differs:

| adapter | taps in a full sweep | what id `k` is |
|---|---|---|
| `loam-hf` | `n_layers + 1` (6 / 9 / 13 / 17 for 25M / 100M / 340M / 624M) | 0 = token embeddings, `k` = block `k` |
| `evo` | 33 | 0 = input to block 0, `k` = output of block `k − 1` of 32 |
| `genomeocean` | `n_layers + 1` (13 / 15) | standard `hidden_states` layout, last one hooked |
| `prokbert` | 7 | standard BERT layout |
| `glm2` | `n_layers` (30 / 33) | `hidden_states[0]` is **block 0's output**; the embedding output is not exposed |
| `ntv3` | 20 | mixed resolution: 7 conv-down states, 6 torso states at `L/128`, 7 deconv-up states; `last` is the final deconv state at nucleotide resolution |
| `evo2-arc` | the length of the spec's `layer_names` (17) | a **position in `layer_names`**, not a block number |

So the paper compares models at their last layer and at their best layer, never at "layer
*k*". [CAVEATS.md](CAVEATS.md) covers how to read a best-layer number.

**The prefix token is part of the readout too.** An embedding depends on what causal context
the forward saw, not only on which tensor was read. `loam-hf` always runs `[BOS, t1..tn]` and
drops position 0 (BOS is context, never pooled); the Evo runners embed the bare sequence;
GenomeOcean and ProkBERT wrap `[CLS] … [SEP]`; gLM2 prepends a strand marker. Serve a causal
model with a different prefix and layer 0 still matches exactly (an embedding lookup does not
see its neighbours) while every later block diverges, which looks like a conversion bug and is
not one.

**`last` is a slice of `all`.** Every adapter guarantees `embed(layers="last")` is exactly
`embed(layers="all")` at the last id. [`adapters/reuse.py`](../src/glmbench/adapters/reuse.py)
relies on it: tasks that read the same corpus share one all-layer forward pass, so the
single-layer rows come free with the sweep.

---

## 1. The one idea that explains everything

**The benchmark core never imports a model.** The core (the part that loads tasks, scores
metrics and writes the leaderboard) is kept free of torch and every model library, so that
models with mutually incompatible dependency sets can live in one suite.

A task gets predictions through a three-layer hand-off:

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ BENCHMARK CORE  (no torch, no model)                                     │
  │   a task asks: "embed these sequences, every layer"                      │
  │        │                                                                 │
  │        ▼                                                                 │
  │   ADAPTER  (adapters/<model>.py — still core, still torch-free)          │
  │   - writes the request to FILES:  input.jsonl  +  request.json           │
  │   - launches the RUNNER with the model environment's Python              │
  └────────┬────────────────────────────────────────────────────────────────┘
           │  (a subprocess)
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ RUNNER  (runners/<model>_runner.py — runs where the model's deps live)   │
  │   - reads request.json + input.jsonl                                     │
  │   - loads the model and runs forward passes                              │
  │   - writes results to FILES:  output.npz  +  response.json               │
  └────────┬────────────────────────────────────────────────────────────────┘
           │  (the adapter reads the files back and validates them)
           ▼
       results flow back to the task → metrics → leaderboard
```

The adapter and runner **never share memory**; they talk over files (the wire protocol,
[`adapters/wire.py`](../src/glmbench/adapters/wire.py)). Two consequences:

1. **Every model call is batched.** A task hands the adapter all its sequences at once; the
   adapter does one round trip, and the runner batches internally.
2. **The runner can run in a different environment from the core.** That is what lets Evo2
   (the `evo2` package, its own env) and the Hugging Face models (the `glmbench` env) share one
   benchmark.

---

## 2. How a runner is launched and what it does

Every spec has a `runner` block. All shipped specs use `backend: local`:

| `runner` field | Meaning |
| --- | --- |
| `backend: local` | The adapter runs `<python_exe> -m glmbench.runners.<model>_runner <request.json>` as a plain subprocess. |
| `python_exe` | The Python of the environment that holds the model's deps (and `glmbench`). Shipped specs read it from `${GLMBENCH_PYTHON:-python}`, or `${GLMBENCH_EVO2_PYTHON:-python}` for Evo2. |
| `gpus` | `"0"` / `"auto"` → CUDA, `"cpu"` → CPU. Choose the physical card with `CUDA_VISIBLE_DEVICES`. |
| `scratch_dir` | Where the wire files go; `null` → a temp dir. |
| `extra` | Adapter-specific knobs (`batch_size`, `token_budget`, …). |

Every shipped runner works the same way once started: it imports torch and the model library,
loads the checkpoint, and computes hidden states and log-probabilities in-process with forward
passes and hooks. There is no second subprocess and no vendor CLI to parse.

A `docker` backend also exists for a model that needs its own image (the runner then executes
inside the container, with the scratch dir and checkpoint bind-mounted). No shipped spec uses
it. If you do, note that `gpus` is passed to `docker run --gpus`, where a bare `"0"` means
*zero* GPUs; write `'"device=0"'`.

---

## 3. Per-model cheat sheet

| Model | adapter | Environment | Checkpoint | Pinned revision(s) |
| --- | --- | --- | --- | --- |
| **LOAM** | `loam-hf` | `glmbench` | Hub `Soilytix/LOAM-{25M,100M,340M,624M}` (gated) or a local copy | weights verified by sha256 against the export's `loam_export.json` |
| **Evo2-7B** | `evo2-arc` | `evo2` | `arcinstitute/evo2_7b_base` (model name `evo2_7b_base`) | weights `074097e9dc788e8bfe045d6495b9f6153a7c6bfc` |
| **Evo 1.5** | `evo` | `glmbench` + flash-attn | `evo-design/evo-1.5-8k-base` | weights `99a9a4df722662b03d2a79b1770c3150421aa9e9`; code `togethercomputer/evo-1-131k-base` @ `c206aab77ae5967a069c4200ecb1858588528c9d` (the `1.1_fix` branch) |
| **GenomeOcean** | `genomeocean` | `glmbench` | `pGenomeOcean/GenomeOcean-{100M,500M}` | 100M `7e558533874dd3ee11ef229d68a2844a88a0c3e2`; 500M `473f8de940f8549539b9bcb3abe6149202b85b4e` |
| **gLM2** | `glm2` | `glmbench` (needs `einops`) | `tattabio/gLM2_{150M,650M}` | 150M `93c88529115476b27bdd85da14311779114a8a64`; 650M `08754cba59a1f97d517f873fad6c672d2b1abdc7` |
| **ProkBERT** | `prokbert` | `glmbench` | `neuralbioinfo/prokbert-{mini,mini-c}` | mini `feb2520a43cd9cdb5b3d8477e47209dbcb55d1dc` + code `neuralbioinfo/nbrg-transformers` @ `366d9336b6b89d7c1feddf56d9b01be8065ccc43`; mini-c `417ebe73d355bc68b01d4fbe9dc2b0cb274b86e0` (code in the same repo, same commit) |
| **NTv3** | `ntv3` | `glmbench` | `InstaDeepAI/NTv3_100M_pre` (gated) | weights `5c685dca15891f5c5b80e0c930e23b87a217e441`; code `InstaDeepAI/ntv3_base_model` @ `0ecff3637f0d3ba5b686d1095083218157c2ca34` |

Every Hub model is fetched once into your Hugging Face cache (`HF_HOME`). The external specs
pin the commits above, and for these `hf_revision` specs the weights commit is what the model
hash is keyed to, so a spec reproduces its hash on any machine.

The capabilities each adapter serves:

| adapter | tokenize | score_sequences | per_token_logprobs | logprob_embedding | embed | masked_marginal_llr |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `loam-hf` | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `evo2-arc` | | ✓ | | ✓ | ✓ | |
| `evo` | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `genomeocean` | ✓ | ✓ | ✓ (per BPE token) | | ✓ | |
| `glm2`, `prokbert`, `ntv3` | ✓ | | | | ✓ | ✓ |

(`glmbench models list` prints the class-level set, which for `loam-hf` omits
`logprob_embedding`: the adapter adds it per checkpoint, when the tokenizer is the
single-nucleotide one, as it is for every LOAM release.)

A task that needs a capability the adapter lacks returns a first-class `N/A` naming it, never
a crash and never a silent zero. `rnagym-dms` asks for `sequence_loglikelihood` and falls back
to `masked_marginal_llr`, so every model above lands a real RNAGym row; the chosen path is
recorded as `scoring_method` in the result and footnoted on the leaderboard. The embedding
tasks (essentiality, EC, and their sweeps) run on every adapter.

---

## 4. Each model in plain language

### LOAM (`loam-hf`)

The four released LOAM models are decoder-only causal LMs with a single-nucleotide tokenizer.
Each Hub repo is a Hugging Face model directory (`model_type: "loam"`) that ships its own
`modeling_loam.py`, loaded with `trust_remote_code`. The adapter is
[`adapters/loam_hf.py`](../src/glmbench/adapters/loam_hf.py), the runner
[`runners/loam_hf_runner.py`](../src/glmbench/runners/loam_hf_runner.py).

- **Env:** the `glmbench` env (torch + transformers). No flash-attn.
- **Checkpoint:** `model.checkpoint` is a Hub repo id or a local directory. The shipped specs
  resolve `${GLMBENCH_LOAM_HF_ROOT:-Soilytix}/LOAM-<size>`: unset, that is the Hub id, fetched
  with `huggingface_hub.snapshot_download` at `model.revision` (the repos are gated: accept the
  terms on the Hub and log in first), hashed from the download, and loaded by repo id at the
  downloaded commit, which every record names as `describe.hub_commit`. Set the variable to a
  directory holding `LOAM-<size>/` to score a local copy. The adapter refuses a directory whose `config.json` is not `model_type: "loam"`,
  and weights whose sha256 disagrees with the export's own `loam_export.json`.
- **Conventions the adapter fixes** (each has a wrong value that yields a plausible, wrong
  number, so none is a spec knob):
  - A **BOS** token is prepended to every input, because the paper's scoring code did. It is
    context only: never a prediction target, never pooled. LOAM itself was trained with no
    token before a sequence, which is the model card's convention for general use; the
    difference is small but not zero (see [USING_LOAM.md](USING_LOAM.md)).
  - The usable context is **8,191 nt**: `max_position_embeddings` (8,192) minus the BOS slot.
    Tasks window their inputs to it, and the runner refuses anything longer.
  - Embedding layer `i` is the **raw output of block `i`** (layer 0 = the token embeddings),
    captured with forward hooks. The final RMSNorm is **not** applied to embeddings: the LOAM
    residual stream carries a few very large dimensions, and a per-token RMSNorm divides every
    informative dimension by them. Log-probabilities *do* go through the final norm, because
    that is the model's own forward.
- **Numerics.** The runner reproduces the arithmetic the paper's LOAM rows were scored with,
  not only their semantics:
  - weights are loaded in fp32, then cast to the compute dtype (`dtype: auto` → bf16 on GPU,
    fp32 on CPU);
  - inputs are right-padded `[BOS, …]` batches run through `model.model` with **no attention
    mask**. That is exact, not an approximation: under the causal mask a real token attends
    only to positions at or before it, and every pad sits after every real token;
  - embedding and per-token log-prob batches are length-bucketed under a **token budget of
    16,384** (batch × padded length); sequence scores use length-sorted chunks of `batch_size`
    (8);
  - log-softmax runs in fp32;
  - mean pooling runs in the compute dtype (`pool_dtype: compute`). `pool_dtype: float32`
    pools in fp32 instead: more accurate, but it **changes the model hash** and its numbers are
    not the paper's.
- **Model hash:** the weight files (`*.safetensors`), the architecture fields of
  `config.json`, `dtype`, `pool_dtype`, and `loam-hf@<version>`. README, licence and model
  card files are not hashed, so the same weights hash identically from a Hub download or a
  local copy.
- **Capabilities:** `tokenize`, `score_sequences`, `per_token_logprobs`, `logprob_embedding`
  (one log-prob per nucleotide) and `embed` (any or all layers in one pass), plus a fused
  embedding + log-prob pass (`embed_logprob`).

To use LOAM outside the benchmark with the same conventions, see [USING_LOAM.md](USING_LOAM.md).

### Evo2-7B (`evo2-arc`)

Evo2 runs through the Arc Institute `evo2` pip package (0.6.0, with its `vtx` kernel package
and flash-attn), in the separate `evo2` environment. The runner builds `Evo2('evo2_7b_base')`
from `arcinstitute/evo2_7b_base` at the commit the spec's `hf_revision` digest names
(`074097e9…`), so the weights loaded are the weights the model hash names. They land in your
Hugging Face cache on first use (`runner.extra.hf_home` overrides the cache dir).

**A tap, not a truncation.** The package runs the whole 32-block stack and registers forward
hooks on named submodules on the way past:

```python
outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=["blocks.28"])
```

`outputs[0]` is the full-depth logits and `embeddings[...]` the tapped activation, from one
pass. Consequences:

- **`layer_names` affects embeddings only.** `rnagym-dms` and the per-nucleotide log-probs
  read the final layer and LM head whatever taps are configured.
- **The layer sweeps work**: every name in `model.layer_names` comes back from one pass.
- **Layer ids are positions in `model.layer_names`, not block numbers.** Evo2 has no integer
  layer space of its own, so the adapter reports each name's index in the configured list, and
  `layers="last"` is the last entry. The shipped spec taps every other block plus the last
  (`blocks.0, blocks.2, …, blocks.30, blocks.31`: 17 taps), so id 5 is block 10. The id→name
  map is in `describe()["layer_id_to_name"]`, and so in every result record.
- **`layer_names` is part of the model hash.** Two specs that differ only by tap are different
  rows, so reading block 28 can never overwrite a result read at block 31.
- **The tap kind sets the readout.** `blocks.N` is the block's own output, the trunk after the
  residual add (`block_output`, what the shipped spec uses). `blocks.N.pre_norm` is the
  *normalised* trunk entering block N (`normed_block_output`). `blocks.N.mlp.l3`, the layer
  Arc's README recommends for embeddings, is the MLP output *before* the residual add
  (`branch_tap`). A spec that mixes kinds is refused, and so is a name that does not resolve
  (the runner calls `get_submodule` before spending a forward pass). Names are arch-specific:
  the 7B has 32 blocks, the 1B 25.
- **Scoring.** The package's `prepend_bos` prepends `tokenizer.eod_id`, not a distinct BOS; for
  Evo2's character tokenizer `eod_id == eos_id == 0`, so `score_prefix: bos` and `eos` are the
  same token. The runner computes the reduction itself with an explicit pad mask, because the
  package's own `score_sequences` averages in pad predictions for the shorter members of a
  mixed-length batch.
- **No attention mask, and none needed.** Every mixer in the stack is causal, so right-padding
  never reaches a real token. The runner length-buckets batches under `token_budget` (8,192).
- **Pin one GPU.** `vtx` places the model itself and will shard it across every visible card:
  always set `CUDA_VISIBLE_DEVICES` to one GPU.
- **Capabilities:** `score_sequences`, `embed`, `logprob_embedding`, all from the same
  forward. `tokenize` and `per_token_logprobs` are `N/A`.

### Evo 1.5 (`evo`)

`evo-design/evo-1.5-8k-base` is a 7B **StripedHyena** causal LM (29 gated-Hyena convolution
layers and 3 attention layers, at indices 8, 16 and 24) with an 8,192-token context and a
byte-level tokenizer. It is a different model and a different serving path from Evo2.

- **`trust_remote_code` is forced, and the code is cross-repo.** There is no in-tree
  StripedHyena class. The modeling code lives in `togethercomputer/evo-1-131k-base`, so the
  spec pins both `revision` (weights) and `code_revision` (the commit of that repo's `1.1_fix`
  branch). It loads on transformers 5.12.1 with one harmless `use_return_dict` deprecation
  warning.
- **flash-attn is required.** The remote attention block instantiates flash-attn's `MHA` with no
  SDPA or eager fallback, so the three attention layers fail without it. It is the only model in
  the `glmbench` env that needs flash-attn ([INSTALL.md](INSTALL.md) covers the build). The
  FFT-convolution kernels are off in this config, so `flashfftconv` is not needed. Do **not**
  `pip install evo-model`: it pins an old transformers and flash-attn ≤ 2.7.4.
- **Byte tokenizer, reimplemented.** `id = clamp(ord(char), 32, 511)`, vocab 512, no special
  tokens and no wrapping. The runner does this one line itself, because the tokenizer
  `auto_map` is inconsistent between the two repos.
- **Embeddings via hooks.** Evo's HF wrapper returns `hidden_states=None`. The runner captures
  layer 0 as the *input* to `backbone.blocks[0]` with a forward pre-hook (the embedding is
  applied through `embedding_layer.embed()`, not `__call__`, so a hook on the embedding layer
  never fires) and layer `i` as the output of `backbone.blocks[i-1]`: 33 taps in one pass.
- **Scoring** follows `evo/scoring.py`: prepend `eod` (id 0) as causal context, then the
  standard causal shift, so every nucleotide is a target and `logprob_embedding` has exactly
  `len(seq)` values.
- **dtype is effectively bf16.** The remote code casts most parameters to bf16 while loading,
  and flash-attn's attention accepts only fp16/bf16, so a spec asking for `float32` still runs
  in bf16. Check `p.dtype` on the built model rather than trusting the spec.
- **Memory and batching.** The weights are about 14 GB in bf16. An all-layer pass (the sweep
  tasks) holds every hooked layer on the GPU and then stacks them, which needs a second full
  copy; the spec's `token_budget: 8192` (one full-length row) keeps that within 48 GB of GPU
  memory.
  Evo 1.5's outputs are not bitwise batch-invariant in practice, so keep the spec's
  `batch_size` / `token_budget` if you want the paper's last digits.
- **Capabilities:** `tokenize`, `score_sequences`, `per_token_logprobs`, `logprob_embedding`,
  `embed`.

### GenomeOcean (`genomeocean`)

JGI's GenomeOcean (100M: 12 layers × 768; 500M: 14 layers × 1536) is a causal LM with a plain
`MistralForCausalLM` architecture and a DNABERT-2-style **BPE** tokenizer (vocab 4096,
variable nucleotides per token) that wraps each sequence as `[CLS] … [SEP]`.

- **Load stock, not `trust_remote_code`.** The repos ship a bundled `modeling_mistral.py`
  written for transformers 4.38.2 that calls the removed `DynamicCache.from_legacy_cache` and
  crashes on transformers ≥ 5. The weights are plain Mistral, so the spec sets
  `trust_remote_code: false` and the in-tree implementation runs them.
- **Context.** The models were trained at 1,024 tokens (`max_position_embeddings` = 32,768 is
  the RoPE ceiling, not the window). Tasks window by nucleotides, so the spec sets
  `max_context: 5000` nt (about 1,024 BPE tokens) and the runner hard-caps tokenization at
  `max_tokens: 1024`.
- **Scoring.** `[CLS]` is causal context; the loss is taken over real BPE tokens only (never
  `[CLS]`, `[SEP]` or `[PAD]`). The tokenizer pads left by default; the runner right-pads
  internally so scores do not depend on batch composition. `per_token_logprobs` is per **BPE
  token**, not per nucleotide, and `token_spans` come from the fast tokenizer's offset mapping.
- **Embeddings:** any or all layers from one `output_hidden_states` pass, with the last block
  hooked. `pool_include_special: true` (the spec default) matches the vendor's mean pool, which
  masks only padding and so includes `[CLS]` and `[SEP]`.
- `attn_implementation: sdpa` by default; `flash_attention_2` is optional. No GenomeOcean pip
  package or vLLM is needed.
- **Capabilities:** `tokenize`, `score_sequences`, `per_token_logprobs`, `embed`.
  `logprob_embedding` is `N/A`: one log-prob per nucleotide needs 1 nt/token.

### gLM2 (`glm2`)

TattaBio's gLM2 (150M: dim 640, depth 30; 650M: dim 1280, depth 33) is a bidirectional
**masked-LM encoder** (`gLM2ForMaskedLM`) trained on interleaved protein (per amino acid) and
intergenic DNA (per nucleotide) elements.

- **`trust_remote_code: true` is forced, and it needs `einops`.** There is no in-tree gLM2.
  The bundled code does `from einops import …`, so the runner env needs `einops` (it is in the
  `[hf]` extra). It uses SDPA; no flash-attn.
- **Case matters.** gLM2 encodes amino acids UPPERCASE and nucleotides **lowercase**. Uppercase
  `ACGT` tokenizes as the amino acids A, C, G, T: no error, just wrong embeddings. The runner
  lowercases every input and prepends a **strand marker** (`<+>` by default; `strand_prefix`:
  `"+"`, `"-"` or `"none"`), since gLM2 never saw nucleotides without one. The marker is
  excluded from pooling (`pool_include_special: false`). There is no `n` token: ambiguous bases
  become `<unk>`.
- **Context:** 4,096 tokens at 1 nt/token, minus the marker: `max_context: 4095` nt,
  `max_tokens: 4096`.
- **Layers:** the `hidden_states` tuple has `depth` entries, and `hidden_states[0]` is block 0's
  output, so a full sweep has `n_layers` taps, not `n_layers + 1`. The runner resolves `layers`
  against the actual tuple length. Pools: `mean`, `max`, `last`, `marker`, `none`.
- **Variant scoring: masked-marginal LLR.** An encoder has no causal likelihood, so
  `score_sequences` is `N/A`. The runner instead loads the masked-LM head and, for each mutated
  position, masks that position on the wild-type background and takes
  `logit(mut) − logit(wt)`; a multi-site variant sums its single-site terms (the ESM /
  ProteinGym convention; substitutions only). One forward per unique (reference, position) is
  cached across variants.
- **Capabilities:** `tokenize`, `embed`, `masked_marginal_llr`.

### ProkBERT (`prokbert`)

NBRG's ProkBERT (6 layers, 384 dim, about 20M parameters) is a BERT-style **masked-LM
encoder**. Its encoder is pre-LN with a final LayerNorm outside the blocks, which is why the
runner hooks the last block. What makes it its own adapter is the tokenizer.

- **Overlapping k-mer "LCA" tokenizer.** `prokbert-mini` uses k = 6, shift = 1: a sequence of
  `L` nt becomes `L − 5` overlapping 6-mer tokens, wrapped `[CLS] … [SEP]` (`[PAD]`=0,
  `[UNK]`=1, `[CLS]`=2, `[SEP]`=3, `[MASK]`=4; non-ACGT windows → `[UNK]`). `prokbert-mini-c`
  uses k = 1 (1 nt/token). The tokenizer is slow (no offset mapping), so `token_spans` are built
  from k and shift; for mini they overlap and their union covers the sequence. `per_token`
  means per k-mer token.
- **Masked-marginal LLR over overlapping k-mers.** A single-nucleotide substitution changes up
  to k tokens, and the vocabulary is 6-mers. The runner masks every content token whose window
  covers the mutated position (one forward) and sums `logit(mutant k-mer) − logit(wild-type
  k-mer)` over them. For mini-c this reduces to the per-position rule gLM2 uses.
- **`trust_remote_code: true` is forced.** For `prokbert-mini` the `auto_map` points at a
  separate repo, `neuralbioinfo/nbrg-transformers`, so the spec pins both `revision` and
  `code_revision`. `prokbert-mini-c` hosts its code in its own repo, so its `code_revision` is
  the weights commit.
- **Do not `pip install prokbert`.** The package pins `transformers<=5.3.0` and pulls
  `torchvision`; installed into a shared env it downgrades transformers and leaves a broken
  torchvision/torch pair. The remote code alone runs on transformers 5.12.1 (one harmless
  `get_extended_attention_mask` deprecation warning).
- **Its `from_pretrained` rejects standard kwargs.** The custom class forwards `**kwargs` into
  its base encoder's `__init__`, which raises a `TypeError` on the transformers-5 `dtype` and
  `attn_implementation` arguments. The runner loads without them and casts with
  `.to(device, dtype)` afterwards.
- **Context:** `max_position_embeddings` = 1,024 is a hard ceiling. The specs set
  `max_context: 1020` nt and `max_tokens: 1024`.
- **Capabilities:** `tokenize`, `embed` (7 hidden states), `masked_marginal_llr`.

### NTv3 (`ntv3`)

InstaDeep's Nucleotide Transformer v3 (`NTv3_100M_pre`, 106.5M) is a **U-Net masked LM**:
conv tower (7 downsampling stages) → 6-layer transformer torso → deconv tower (7 upsampling
stages) → LM head. The Hub repo is gated.

- **Inputs must be a multiple of 128, and the torso ignores the attention mask.** The U-Net
  downsamples by 2⁷ = 128, so any other length crashes in the conv tower (`Calculated output
  size … too small`: a loud failure, not a wrong answer). The torso runs with
  `attention_mask=None`, so pad tokens reach real positions through attention, and padding a
  batch to its longest member makes a sequence's output depend on its batch-mates. The runner
  instead pads **each sequence to its own next multiple of `pad_multiple`** (128) and batches
  only sequences of equal padded length, which keeps scores and embeddings batch-invariant.
- **Keep the parameters in fp32.** NTv3 manages its own mixed precision through per-block
  `*_compute_dtype` config fields and assumes fp32 parameters; casting them to bf16 crashes
  its conv path on GPU (`Input type … and weight type … should be the same`). The runner never
  casts, and the spec says `dtype: float32`.
- **`trust_remote_code: true` is forced, with cross-repo code** in `InstaDeepAI/ntv3_base_model`:
  the spec pins both `revision` and `code_revision`. It loads on transformers 5.12.1 with no
  extra deps. `attn_implementation` is not honoured (the torso's attention is hand-written).
- **Single-nucleotide tokenizer, no wrapping.** Vocab 11 (`<unk>`=0, `<pad>`=1, `<mask>`=2,
  `<cls>`=3, `<eos>`=4, `<bos>`=5, `A`=6, `T`=7, `C`=8, `G`=9, `N`=10); nothing is added to a
  sequence, other characters become `<unk>`, and there is no lowercasing or marker.
- **Mixed-resolution hidden states.** `output_hidden_states` returns 20 states at different
  lengths (conv-down `L … L/128`, torso `L/128`, deconv-up `L/128 … L`). `layers="last"` is the
  final deconv state, at nucleotide resolution (dim 768). Each layer pools over its own non-pad
  span; `pool="none"` needs all selected layers at one resolution and says so loudly otherwise.
- **Context:** the model card claims up to about 1 Mb, but memory binds first. The spec sets
  `max_context: 12288` nt (96 × 128).
- **Capabilities:** `tokenize`, `embed`, `masked_marginal_llr` (per-position, as gLM2).

---

## 5. The pitfalls that bite everyone

Almost all "why won't my model run" failures are environment or spec issues, not benchmark
bugs.

**1. `ModuleNotFoundError: No module named 'glmbench'` from the runner.** `runner.python_exe`
points at a Python that does not have gLMBench installed (with the shipped specs, usually
`python` resolving to the wrong interpreter). Set `GLMBENCH_PYTHON` (or `GLMBENCH_EVO2_PYTHON`)
to the environment's interpreter.

**2. A model's bundled `trust_remote_code` crashes on your transformers (GenomeOcean).** Some
repos ship modeling code via `auto_map` pinned to an old transformers. If the architecture is
a stock one (`MistralForCausalLM`, `LlamaForCausalLM`, …), load it with
`trust_remote_code: false`: the in-tree implementation runs the same weights on any version.

**3. The opposite case: a custom architecture (gLM2, ProkBERT, NTv3, Evo 1.5, LOAM).** With no
in-tree class, `trust_remote_code: true` is forced. Watch for extra remote-code deps (gLM2 needs
`einops`), and pin `revision` (and `code_revision` when the `auto_map` points at another repo)
so you are not silently pulling new remote code.

**4. gLM2 returns garbage on UPPERCASE DNA.** Its tokenizer reads uppercase as amino acids.
The `glm2` runner lowercases; if you call the model directly, lowercase first.

**5. A vendor pip package you must not install (ProkBERT, Evo 1.5).** `prokbert` and
`evo-model` both pin old transformers versions (and `evo-model` an old flash-attn). The remote
code runs without them. If a vendor library is truly needed, give it its own environment.

**6. A custom `from_pretrained` that rejects standard kwargs (ProkBERT).** Load plainly, then
`.to(device, dtype)`.

**7. A required length multiple plus an ignored attention mask (NTv3).** Batch-max padding is
then not batch-invariant. Pad each sequence to its own length multiple and batch equal lengths.
For any new conv or U-Net model, check both before trusting batch-max padding.

**8. Parameters that must stay fp32 (NTv3).** Load natively; do not cast.

**9. A mandatory custom kernel with no prebuilt wheel for your system (Evo 1.5, Evo2).**
flash-attn's prebuilt wheels need a recent system C library; where they fail to import, build
from source ([INSTALL.md](INSTALL.md)).
For Evo2, `vtx` imports flash-attn without declaring it, so `pip install evo2` alone gives an
env that installs cleanly and fails on `import evo2`.

**10. Evo2 spread across every GPU.** `vtx` shards the model over all visible cards. Set
`CUDA_VISIBLE_DEVICES` to one.

**11. Comparing `hidden_states[-1]` across models.** For a pre-LN model it is the final-normed
state, not the last block's output (section 0). Use the adapters, or apply the same readout by hand.

**12. Serving LOAM with your own code.** The model card's general-use conventions (no BOS,
8,192 nt of context, final-normed embeddings) give sensible numbers that are not the paper's.
The `loam-hf` adapter applies the paper's conventions; [USING_LOAM.md](USING_LOAM.md) compares
the two.

---

## 6. Debugging a serving problem

When a run fails, the error carries the failing command and the runner's captured output. To
dig deeper:

- **Keep the scratch dir** with `glmbench run … --keep-scratch`. It holds everything the run
  produced:
  - `request.json` + `input.jsonl`: exactly what was asked;
  - `output.npz` + `response.json`: what came back, if it got that far;
  - `runner.stdout` / `runner.stderr`: the runner process's output.
- **Dry-run the command** with `glmbench run … --dry-run` (and a throwaway `--results` dir). It
  prints the exact runner command without executing it, so the tasks then report `ERROR`; it
  is how to check `python_exe` before a real run, and how the adapter unit tests check
  commands without a GPU.
- **Check the environment**, not the benchmark: run `<python_exe> -c "import torch,
  transformers, glmbench"` (plus `einops`, `flash_attn` or `evo2` as the model needs).
- **Resume is on by default.** A re-run skips tasks that already have an `OK` result for the
  same model hash; pass `--no-resume` to recompute them.

---

## 7. Quick reference

Each line runs the cheapest benchmark (`rnagym_prok`, zero-shot RNAGym). Swap in
`--benchmark loam_paper` for the paper's five task rows ([REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md)).
Choose the GPU with `CUDA_VISIBLE_DEVICES`.

```bash
# LOAM (glmbench env; gated Hub repo, log in first)
glmbench run --model specs/loam/LOAM-25M.yaml --benchmark rnagym_prok --results results/

# Evo2-7B (evo2 env; ONE visible GPU)
export GLMBENCH_EVO2_PYTHON=/path/to/envs/evo2/bin/python
CUDA_VISIBLE_DEVICES=0 glmbench run --model specs/external/evo2-arc-7b-resid.yaml --benchmark rnagym_prok --results results/

# Evo 1.5 (glmbench env + flash-attn)
glmbench run --model specs/external/evo-1.5-8k.yaml --benchmark rnagym_prok --results results/

# GenomeOcean, gLM2, ProkBERT, NTv3 (glmbench env; NTv3 is gated)
glmbench run --model specs/external/genomeocean-100M.yaml --benchmark rnagym_prok --results results/
glmbench run --model specs/external/glm2-650M.yaml       --benchmark rnagym_prok --results results/
glmbench run --model specs/external/prokbert-mini.yaml   --benchmark rnagym_prok --results results/
glmbench run --model specs/external/ntv3-100M.yaml       --benchmark rnagym_prok --results results/

# the board for everything under results/
glmbench leaderboard --benchmark rnagym_prok --results results/
```

`glmbench models list` shows each adapter with the capabilities it declares; `glmbench tasks
list` shows each task with the capabilities it requires.

---

## 8. TL;DR

- The core never imports a model; the **adapter** (core, torch-free) talks to a **runner**
  (the model's environment) over **files**.
- Every shipped runner loads its model in-process, in the environment `runner.python_exe`
  names: `glmbench` for everything except Evo2, which uses `evo2`.
- Every adapter returns the **raw block output** as a layer and declares it; layer *ids* still
  mean different things across models, so compare last layer to last layer and best to best.
- Most "it won't run" problems are the wrong `python_exe`, a `trust_remote_code` / transformers
  mismatch, a missing flash-attn, or a gated repo you have not been granted. Use
  `--keep-scratch` and `--dry-run` to pin them down.
