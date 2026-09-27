# Adding a model

A model is a localized change: **drop one adapter into `src/glmbench/adapters/`, drop
its runner into `src/glmbench/runners/`, register the adapter, add one import line, and
write a spec**. Nothing in the runner orchestrator (`run.py`), the tasks, or the
leaderboard changes. The extensibility test in `tests/unit/test_phase8.py` proves
a brand-new adapter appears in `glmbench models list` and lands a row on the board
without touching the runner.

## The two halves: adapter (core, torch-free) + runner (the model's env)

The **execution boundary is a subprocess per model**. Models with mutually incompatible
dependency sets (the Hugging Face models on transformers 5.x, Evo2 on the Arc `evo2` package
in its own env) coexist because the benchmark core never imports any of them. Each model is
two files:

| File | Runs in | May import torch? | Job |
| --- | --- | --- | --- |
| `adapters/<name>.py` | the **core** env | **No** (the torch-free import test) | assemble wire requests, build the runner backend, validate responses, compute the model hash |
| `runners/<name>_runner.py` | the **model's own** env | **Yes** | read the wire request, run the model, write the wire response |

The adapter and runner talk over **files** via the wire protocol (`adapters/wire.py`):
the adapter writes a `Request` (op + params + a JSONL of sequences), the runner writes a
`Response` (status + a path to an `.npz`/JSON output). Heavy tensors **always** go via
`.npz` referenced by path — never inline in JSON.

## The adapter contract (`adapters/base.py`)

```python
class ModelAdapter(ABC):
    name: str
    adapter_version: str                       # pinned; participates in the model hash
    CAPABILITIES: frozenset[Capability] = frozenset()

    readout: Readout | None                    # WHICH TENSOR embed() returns
    readout_evidence: str                      # "probed" | "by_construction" | "unverified"
    readout_note: str | None                   # the qualifier, written into every record
    sweepable: bool = True                     # can it reach a non-final layer at all?
    not_sweepable_reason: str | None = None

    @classmethod
    def expected_sweep_taps(cls, describe: dict) -> int | None: ...   # taps in a full sweep

    @classmethod
    def from_spec(cls, spec, *, keep_scratch=False, dry_run=False) -> "ModelAdapter": ...
    def model_hash(self) -> str: ...           # must NOT require running the model
    def describe(self) -> dict: ...            # human-readable config echo

    # capability-gated, batch-first; default bodies raise CapabilityNotImplemented
    def tokenize(self, sequences) -> list[list[int]]: ...
    def score_sequences(self, sequences, *, reduction="mean") -> list[float]: ...
    def per_token_logprobs(self, sequences) -> list[np.ndarray]: ...
    def logprob_embedding(self, sequences) -> list[np.ndarray]: ...     # one log-prob per nt
    def embed(self, sequences, *, layers, pool) -> EmbeddingResult: ...
    def score_variant_llr(self, items: list[dict]) -> list[float]: ...   # MASKED_MARGINAL_LLR
```

Rules:

- An adapter **"has" a capability iff it overrides the method *and* lists it in
  `CAPABILITIES`**. Gating reads the *declaration* only (`capabilities()`), so a
  half-built adapter that overrode a method but forgot to declare it is treated as **not
  having** the capability (fail-closed). Declare only what your runner really computes —
  the rest surface as a correct first-class `N/A`, never a crash.
- A masked/encoder LM with **no causal head** should declare `MASKED_MARGINAL_LLR`
  (mapped to the adapter method `score_variant_llr`) instead of leaving variant-effect
  tasks as `N/A` — see the encoder note in the recipe.
- All capability methods are **batch-first**: take a list, return a list. The adapter
  micro-batches internally; it never exposes per-item round-trips to a task.
- For embeddings, honor the `EmbeddingResult` contract (`adapters/base.py`), including
  `token_spans` when `pool="none"` so tokens map back to nucleotides regardless of
  tokenizer granularity.
- **Declare `readout` — WHICH tensor `embed()` returns.** Layer `i` is the **output of
  transformer block `i`**, pre any model-level final normalisation; layer 0 is the embedding
  output. In a pre-LN model `hidden_states[-1]` is the final-normed state, not the last block's
  output, and the two probe differently, so this is not a rounding question. The four values
  are three different objects and one qualifier — `block_output` (the raw trunk; the target),
  `normed_block_output` (the trunk with a *per-tap* norm), `post_final_norm` (the trunk with
  the model's *closing* norm — a pre-LN decoder's `hidden_states[-1]`), and `branch_tap` (an
  attention/MLP output *before* the residual add: a proposed edit, not a state).
  Section 0 of [HOW_MODELS_ARE_SERVED.md](HOW_MODELS_ARE_SERVED.md) lists what every shipped
  adapter declares.
- **Say how you know** via `readout_evidence`. `probed` means you measured a hook on the last
  block against `hidden_states[-1]` on the real checkpoint —
  `glmbench.runners._readout.probe_last_layer(model, input_ids)` does this and returns
  `identical` or `differs` — and `readout_note` records the verdict; `by_construction` means
  your runner *builds* the tensor itself (an explicit hook, a named tap) and `readout_note`
  says where. `unverified` is the visible "nobody checked" state. **Do not skip the probe
  because your model "looks post-LN"**: ProkBERT looked post-LN and needed the hook (its
  encoder is pre-LN with a final LayerNorm outside the blocks), and NTv3 looked like it needed
  one and did not. `runners/_readout.py` also has `last_block_capture`, the hook the HF runners
  use to splice the raw last-block output into the hidden-states tuple.
- **The PREFIX TOKEN is part of the readout too.** Two adapters can agree about *which tensor*
  and still disagree about *which forward pass*. `loam-hf` runs `[BOS, t1..tn]` and returns
  positions 1..n; serve the same weights without the BOS and **layer 0 matches bitwise** (an
  embedding lookup does not see its neighbours) while every block after it diverges. That
  reads exactly like a numerical bug in a conversion and is not one. If your model has a
  sibling served another way (the vendor's own code, a different adapter), run them against
  each other before believing either.
- **Declare sweepability and the sweep size.** `sweepable = False` (with a reason) if your
  serving path can only reach the final layer — "cannot be swept" and "was not swept" are
  opposite claims about a model. `expected_sweep_taps(describe)` defaults to `n_layers + 1`;
  **override it if that is wrong for your architecture.** Three shipped adapters legitimately
  differ: gLM2's tuple omits the embedding output (full = `n_layers`), NTv3's depth axis is a
  U-Net (`2·num_downsamples + num_layers`), and Evo2's is its spec's `layer_names`.
- **`embed(layers="last")` must be exactly `embed(layers="all")` at the last id.**
  `adapters/reuse.py` serves a single-layer task from a sweep's cached all-layer forward on
  that basis, so a violation makes a fused run disagree with an unfused one.

## Runner backends (`adapters/runner_backend.py`)

Pick one in the spec's `runner.backend`:

- **`local`** → `LocalSubprocessRunner(python_exe=<the model's env python>)`. Runs
  `"<python_exe> -m glmbench.runners.<name>_runner <request.json>"`. Every shipped model
  uses it: a sibling conda env or venv that has `glmbench` plus the model's deps.
- **`docker`** → `DockerRunner(image=…)`. `docker run … <image> python -m
  glmbench.runners.<name>_runner …`, bind-mounting a shared scratch dir (and the
  gLMBench `src` + the checkpoint) at the **same absolute path on both sides** so wire
  paths resolve. For a model that needs its own image; `adapters/glm2.py` shows how an
  adapter builds either backend off `spec.runner.backend` — that choice lives in the
  adapter, not a forked runner.

Both backends support `dry_run` (print the exact command, run nothing — used in tests),
capture the runner's full stdout/stderr to scratch and surface it in `RunnerError` on
non-zero exit (no silent failures), and keep scratch on failure / `--keep-scratch`.

## Model hash (`adapters/hashing.py`)

```
model_hash = "glmb:" + sha256(weights_digest ‖ canonical_config_json ‖ name@version)[:24]
```

The three parts are the **weights digest**, the adapter's **hashed config** (canonical JSON of
whatever architecture and serving fields the adapter decides change the numbers, e.g.
`loam-hf` hashes the architecture fields of `config.json` plus `dtype` and `pool_dtype`), and
**`name@adapter_version`** from the adapter class.

Computing it must **not** require running the model (so the board can be keyed before
any GPU time). Pick a `weights_digest.strategy` in the spec:

- **`file_sha256`** — sha256 over the checkpoint (multi-file → sha256 of the sorted
  per-file sha256s). Use it for local weights; `loam-hf` uses it (over the `*.safetensors`
  files only). A `weights_digest_cache` sidecar lets a giant checkpoint be digested once, not
  on every run.
- **`hf_revision`** — the Hugging Face repo commit (every external spec; avoids reading
  gigabytes).
- **`declared`** — a user-supplied digest string (escape hatch; logged as untrusted — the
  `echo` specs use it).

**`adapter_version` in a spec must equal the adapter class's `adapter_version`**; a mismatch
is an error. The hash uses the class attribute, so when a change to your runner changes what
it returns (a different readout, a different scoring convention), bump the class version and
your specs together: the hash moves and the new results land on a new row instead of silently
replacing the old ones.

Note what is **not** in that formula: **the checkpoint's path**. Identity is the weights
*content* (or the HF commit) plus the architecture read from the config's *contents*. Moving
a checkpoint to a different directory — or a different machine — preserves its hash and its
leaderboard row.

## Spec portability — never hardcode a machine's paths

Specs are committed and must run on **any** machine, so the loader expands **`${VAR}`** and
**`${VAR:-default}`** in every string of the YAML before validation. An unset variable with no
default is a **loud error naming the variable** — never a literal `${VAR}` that resurfaces
later as a baffling `FileNotFoundError`.

The shipped specs use three variables ([INSTALL.md](INSTALL.md)):

| Variable | Names |
| --- | --- |
| `GLMBENCH_PYTHON` | the runner env's python (`runner.python_exe`); default `python` |
| `GLMBENCH_EVO2_PYTHON` | the `evo2` env's python, for the Evo2 spec; default `python` |
| `GLMBENCH_LOAM_HF_ROOT` | a directory holding local copies of the LOAM releases; unset → the Hub org `Soilytix` |

Give each a `:-default` that works for most users. Only `$`**`{...}`** is expanded — a bare
`$FOO` passes through untouched, so docker args and `gpus: '"device=0"'` can never be mangled.

**The most portable spec carries no path at all:** an HF repo id + a pinned `revision`
(`weights_digest.strategy: hf_revision`) reproduces itself on any machine straight from the
pin. Prefer that whenever the weights are public, and put local weights behind a variable
with `file_sha256`.

## Recipe

1. **Copy an adapter + runner.** Two templates cover most cases:
   - **`echo`** (`adapters/echo.py` + `runners/echo_runner.py`) — the simplest complete
     adapter: every capability, the full wire round trip, no model and no torch. Read it first
     to learn the shape; its runner computes deterministic fake outputs, so you can replace
     them op by op.
   - **`loam_hf`** (`adapters/loam_hf.py` + `runners/loam_hf_runner.py`) — a complete
     Hugging Face-format adapter and runner, **the default for a causal LM you load with
     `transformers`**. It resolves a Hub repo id or a local directory, hashes and verifies the
     weight files, hashes the architecture fields of `config.json`, captures raw block outputs
     with forward hooks, length-buckets batches under a token budget, and serves every causal
     capability. Its tests are the ones to copy (step 5).

   For a specific shape, the shipped adapters are worked examples: a BPE tokenizer with
   offset-mapped spans → `genomeocean`; a masked-LM encoder with masked-marginal scoring →
   `glm2` (1 nt/token), `prokbert` (overlapping k-mers) or `ntv3` (U-Net, per-sequence
   padding); a custom architecture whose wrapper discards hidden states → `evo`; a model served
   by its own pip package in a separate env → `evo2_arc`.
2. **In the adapter:** set `name` / `adapter_version`, declare `CAPABILITIES` and `readout`,
   implement `from_spec` (read the spec, build the runner backend, read the architecture
   config torch-free for the hash + `max_context`), `model_hash`, `describe`, and the
   capability methods (each builds a wire `Request`, runs it via the backend, validates the
   `Response`).
3. **In the runner:** read the `Request`, import your model (torch is fine *here*), run
   it, write the `Response` (`.npz` for arrays/scores). Keep torch a **lazy** import if
   any pure parsing/command-building helper should stay unit-testable off-GPU. Traps we hit
   onboarding the shipped models:
   - *Stock architecture, broken bundled code (GenomeOcean):* if the model is a standard
     architecture (`MistralForCausalLM`, `LlamaForCausalLM`, …), load it with
     `trust_remote_code=False`. A repo's bundled `auto_map` code is often pinned to an old
     `transformers` and crashes on current versions; the in-tree implementation runs the
     same weights and is version-robust.
   - *Tokenizer granularity:* don't assume 1 nt/token. A BPE/k-mer tokenizer needs
     `token_spans` from the fast tokenizer's offset mapping (or built from k and shift for a
     slow one), makes `per_token_logprobs` *per token* (not per nt), and makes
     `logprob_embedding` a clean `N/A`. `max_context` is consumed by tasks as
     **nucleotides** — convert from any token-based window, and keep a token cap in the
     runner as the backstop.
   - *Encoder / masked-LM models (gLM2, ProkBERT, NTv3):* a bidirectional MLM has **no
     causal log-likelihood**, so `SEQUENCE_LOGLIKELIHOOD` / `PER_TOKEN_LOGPROBS` /
     `LOGPROB_EMBEDDING` stay clean `N/A`. To still score variant effects, declare
     `MASKED_MARGINAL_LLR` and implement `score_variant_llr(items)`: for each
     `{"reference": <WT DNA>, "mutations": [[pos0, "WT", "MUT"], ...]}` item, mask position
     `i` in the reference and return `Σ_i (logit_mut_i − logit_wt_i)` (substitutions only —
     indels skipped and counted; multi-site is the additive single-site approximation, the
     ESM/ProteinGym convention). `rnagym-dms` declares this as an ordered fallback to
     `SEQUENCE_LOGLIKELIHOOD`, so an MLM lands a real row instead of `N/A`. Resolve
     `embed(layers=…)` against the **actual** `len(hidden_states)` — custom encoders don't
     always return `n_layers+1` states (gLM2 returns `depth`). Overlapping k-mer tokenizers
     (shift < k) make `token_spans` windows whose *union* covers the sequence, and make the
     LLR mask *all* tokens that cover the mutated nucleotide.
   - *Custom architectures:* no in-tree class means `trust_remote_code: true` is forced. Watch
     for extra remote-code deps (gLM2 needs `einops`) and input-format assumptions (gLM2 reads
     UPPERCASE as amino acids, so lowercase DNA first). When the `auto_map` uses the
     `other-repo--module.Class` form, the code lives in a *separate* Hub repo: pin **both** the
     weights `revision` and a `code_revision`.
   - *A vendor pip package you must not install (ProkBERT, Evo 1.5):* `prokbert` and
     `evo-model` pin old `transformers` versions and break a shared env. Prefer the
     remote-code path; if the vendor library is truly needed, give it a **dedicated** env.
   - *A custom `from_pretrained` that rejects standard kwargs (ProkBERT):* it forwards
     `**kwargs` into its base `__init__`, which raises `TypeError` on the transformers-5
     `dtype` / `attn_implementation` kwargs. Load plainly and `.to(device, dtype)` after.
   - *Mandatory custom kernels (Evo 1.5):* an attention layer with no SDPA/eager fallback
     makes flash-attn a hard runtime dependency, and its prebuilt wheels may not match your
     OS (see [INSTALL.md](INSTALL.md)). Do the **load probe first** — "does it import and
     run one forward here?" is the unknown worth resolving before writing the adapter.
   - *Hidden states discarded by the wrapper (Evo 1.5):* serve `embed` via forward **hooks**
     on the backbone blocks; an embedding applied through a custom method (not `__call__`)
     needs a forward-**pre**-hook on block 0 to capture layer 0.
   - *Length multiples and ignored masks (NTv3):* check whether the model needs input lengths
     of a fixed multiple and whether it honours the attention mask before trusting batch-max
     padding. If it ignores the mask, pad each sequence to its own length and batch equal
     lengths only.
4. **Register + import:**

   ```python
   # in adapters/<name>.py
   from glmbench import registry

   @registry.register("adapter", "<name>")
   class YourAdapter(ModelAdapter): ...
   ```

   ```python
   # in adapters/__init__.py — import for the registration side effect
   from .your_adapter import YourAdapter  # noqa: F401
   ```
5. **Test it** without the real model: build a tiny random model of your architecture, save it
   to a temp dir, and drive it through the real wire boundary with
   `python_exe=sys.executable` on CPU. `tests/unit/test_loam_hf.py` (torch-free: registration,
   readout, context, hash) and `tests/unit/test_loam_hf_runner.py` (every op against a direct
   computation, including that `last` is exactly a slice of `all`) are the pattern. Also run
   `tests/unit/test_phase0.py` (the core stays torch-free) and `tests/unit/test_phase8.py`
   (extensibility).
6. **Write a spec** under `specs/`:

   Public HF weights — **the portable case, no filesystem path anywhere**:

   ```yaml
   adapter: <name>
   adapter_version: "0.1.0"            # must equal YourAdapter.adapter_version
   display_name: "<label>"             # leaderboard label only; not hashed
   model:
     checkpoint: "<org>/<repo>"        # HF repo id — resolves on any machine
     revision: "<commit-sha>"          # pin it: reproducibility + the hash key
   weights_digest:
     strategy: hf_revision
     value: "<commit-sha>"
   runner:
     backend: local
     python_exe: "${GLMBENCH_PYTHON:-python}"
     gpus: "0"
     extra: {}                         # adapter-specific knobs
   ```

   Local weights — path goes behind a variable, **never hardcoded**:

   ```yaml
   model:
     checkpoint: "${MY_WEIGHTS_ROOT}/<subdir>"     # unset → a loud error naming MY_WEIGHTS_ROOT
   weights_digest:
     strategy: file_sha256
     value: null
   runner:
     backend: local                    # local | docker
     python_exe: "${GLMBENCH_PYTHON:-python}"      # local
     image: null                       # docker
     gpus: "0"                         # docker: '"device=0"' — bare "0" means ZERO GPUs
     scratch_dir: null
     extra: {}
   ```

   Unknown top-level and `runner` keys are rejected at load. Knobs under `model` and
   `runner.extra` are read by your adapter, so reject unknown ones there too: a typo'd knob
   should fail loudly, not silently do nothing.
7. **Run it:**

   ```bash
   glmbench run --model specs/<name>.yaml --benchmark rnagym_prok --results results/
   glmbench leaderboard --benchmark rnagym_prok --results results/ \
                        --out results/rnagym_prok/LEADERBOARD.md
   ```

`glmbench models list` now shows your adapter + its declared capabilities; the run lands
a row keyed by its model hash; the board renders it next to every other model.

## Keep the core torch-free

`tests/unit/test_phase0.py` forbids importing torch from core (`config/`, `tasks/`,
`leaderboard/`, `registry.py`, `cli.py`, **and `adapters/*.py`**). Only `runners/*` may
import torch. If a pure helper in the runner (FASTA writing, command building, output
parsing) should be testable in torch-free CI, import torch lazily inside the function
that needs it (the Evo runners do this). The test suite asserts `import glmbench` does not
drag torch into `sys.modules`.

## Before you open a pull request

- [ ] Adapter is torch-free; all model code lives in the runner.
- [ ] `CAPABILITIES` declares exactly what the runner computes (fail-closed).
- [ ] Capability methods are batch-first (list in, list out).
- [ ] `model_hash()` is computable **without** running the model.
- [ ] `adapter_version` in every spec equals the class's.
- [ ] `from_spec` builds the right runner backend off `spec.runner.backend`.
- [ ] Wire responses are validated (length, shape, NaN) before returning to a task.
- [ ] `readout` is declared, with `readout_evidence` and a `readout_note` that cites the
      probe verdict or the code that makes it true. **Run `probe_last_layer` before
      declaring** — the rule of thumb has been wrong in both directions.
- [ ] `sweepable` + `expected_sweep_taps()` are right for this architecture (the default is
      `n_layers + 1`; three shipped adapters legitimately differ).
- [ ] `embed(layers="last")` is **exactly** `embed(layers="all")` sliced at the last id.
- [ ] Registered + imported in `adapters/__init__.py`; a spec exists.
- [ ] No edits to `run.py`, any task, or the leaderboard.
