# gLMBench Task Overview

A one-page reference to every benchmark task documented in `task_*.md`. For each task:
what the model sees (**input**), what the adapter is asked to return (**adapter call**),
how it is scored (**metric**), and what the task emits (**output**).

Every task uses the model as a **frozen feature extractor** — no fine-tuning. A task makes
**exactly one batched adapter call** across all its data, then does CPU-side probing/scoring.
Before comparing numbers across models, read [CAVEATS.md](CAVEATS.md).

## Capabilities — what each adapter method computes, and from which activations

Two families of signal come out of a transformer, and the tasks split cleanly along them:

- **Hidden-state activations** (the residual stream after each block) → geometry. Used by
  `EMBEDDING`. These are the raw `d_model`-wide vectors *inside* the network; no vocabulary
  projection, no probabilities.
- **LM-head logits** (the residual stream projected to vocabulary size `V`, then softmaxed) →
  likelihood. Used by `SEQUENCE_LOGLIKELIHOOD`, `LOGPROB_EMBEDDING`, and `MASKED_MARGINAL_LLR`.
  These read the model's *output distribution over bases*, not its internal geometry.

### `EMBEDDING` — `embed(seqs, layers, pool)` → `EmbeddingResult`

Extracts **hidden-state activations** and pools them. One forward pass per sequence, with the
model's own prefix token if it uses one (LOAM prepends BOS):

1. **Forward & capture.** Token-embed the input, then run every block once, *capturing the
   state after each block*. Layer axis convention: **layer 0** = the embedding-layer output
   (post token-embedding, pre block 0) where the model exposes it; **layer `i`** = the output
   of block `i`, **pre any model-level final normalisation**. `layers="last"` keeps only the
   final block; `layers="all"` keeps the whole axis; an explicit list indexes that axis
   (negatives allowed). Every adapter declares which tensor it returns (`readout`), and every
   public adapter returns all requested layers from one forward pass. The tap axis differs by
   model (gLM2 has no embedding tap, NTv3 is a U-Net, Evo2's taps are named in its spec); see
   [CAVEATS.md](CAVEATS.md#3-readout-conventions-what-layer-i-is-in-each-model).
   A pooled embedding is **not** layer-invariant, which is why the layer-sweep tasks exist.
2. **Drop the prefix and other special tokens**, so pooling sees only real sequence positions
   (a spec can keep them: GenomeOcean's does, to match its authors' mean-pool).
3. **Pool** over the token axis → the returned feature:
   - `mean` / `max` — reduce the token axis to one `[n_layers, H]` vector per sequence (mean is
     the benchmark default; it averages the per-position hidden states).
   - `last` / `eos` — the final real token's hidden state; `bos` — the BOS position's.
   - `none` — the full per-token tensor `[n_tokens, n_layers, H]` **plus** `token_spans`
     (half-open `(nt_start, nt_end)` per token into the original nucleotide string, so a per-base
     task can align tokens→nucleotides even under a multi-nucleotide tokenizer).
   Not every adapter supports every pool; the tasks in this repo use `mean`.
   Returns `[n_seq, n_layers, H]` (reducing pools). **No softmax, no logits** — these are the
   internal activations themselves.

### `SEQUENCE_LOGLIKELIHOOD` — `score_sequences(seqs, reduction)` → one float/seq

Reads the **LM-head logits** to score how probable the model finds the actual sequence:

1. One causal forward → `logits` of shape `[B, S, V]`.
2. `log_softmax` over the vocab axis of `logits[:, :-1, :]` (each position predicts the *next*
   token) → per-position log-probabilities over all bases.
3. **Gather the realized next token** at each position (teacher forcing): `tok_lp[t] =
   log p(x_t | x_<t)` — the model's log-probability of the base that actually occurs. BOS and
   padding positions are excluded.
4. **Reduce** over real positions: `reduction="mean"` → mean per-token causal log-likelihood
   `ℓ(x) = (1/L) Σ_t log p(x_t | x_<t)` (length-normalised, the benchmark default);
   `reduction="sum"` → the unnormalised total. One scalar per sequence; higher = the model finds
   the sequence more probable. For a multi-nucleotide tokenizer (GenomeOcean's BPE) the mean is
   over tokens, not nucleotides.

### `LOGPROB_EMBEDDING` — `logprob_embedding(seqs)` → one length-`N` array/seq

The **reduction-free form of `score_sequences`**: identical forward + gather, but instead of
reducing it keeps the whole per-position vector `z_t = log p(x_t | x_<t)`, length `N` = number of
nucleotides. (`−z_t` is the *surprisal* of base `t`.) Because it shares the exact
prefix/BOS/masking of scoring, `mean(z) == score_sequences(reduction="mean")` per sequence to fp
tolerance. **Single-nucleotide tokenizers only** (1 token = 1 nt, so per-token ≡ per-nt); a
k-mer or BPE token's log-prob is a joint over several bases, not separable per-nt conditionals.
Declared by `loam-hf`, `evo` and `evo2-arc`; none of the tasks in this repo requires it.

### `MASKED_MARGINAL_LLR` — `score_variant_llr(variants)` → one float/variant

The variant-effect path for **masked-LM (bidirectional) encoders** (gLM2, NTv3, ProkBERT),
which have no causal factorization and so cannot serve `score_sequences`. It also reads
**LM-head logits**, but under masking rather than autoregressively. For a variant with mutated
positions `M` on reference (WT) sequence `r`:

1. For each mutated position `i`, replace the WT base at `i` with the model's `[MASK]` token on
   the otherwise-unchanged WT background, and run **one forward**.
2. Read the logits at position `i` over the vocabulary and take the difference for that site:
   `logit(mut_i) − logit(wt_i)`. Because both bases are read from the *same* masked context, the
   softmax normalizer cancels — the raw logit difference **is** the log-likelihood ratio
   `log p(mut_i | r_{\i}) − log p(wt_i | r_{\i})`.
3. **Sum over the mutated positions**: `LLR = Σ_{i∈M} [logit(mut_i) − logit(wt_i)]`. Multi-site
   variants use the additive single-site approximation (each position masked independently on the
   WT background; no joint masking). **Substitutions only** — indels are skipped-and-counted.
   Higher = the model prefers the mutant, the same sign convention as the AR score, which is what
   lets one metric rank AR and MLM models together. (ProkBERT-mini's tokens are overlapping
   6-mers, so a substitution touches up to 6 tokens; its runner masks all of them.)

**Summary table:**

| Capability | Adapter method | Activations read | Returns |
|---|---|---|---|
| `EMBEDDING` | `embed(seqs, layers, pool)` | hidden states (per block) | `[n_seq, n_layers, H]` pooled vectors |
| `SEQUENCE_LOGLIKELIHOOD` | `score_sequences(seqs, reduction)` | LM-head logits (causal) | one log-likelihood scalar/seq |
| `LOGPROB_EMBEDDING` | `logprob_embedding(seqs)` | LM-head logits (causal) | per-nt log-prob vector `log p(x_t\|x_<t)` |
| `MASKED_MARGINAL_LLR` | `score_variant_llr(variants)` | LM-head logits (masked) | Σ-of-sites logit-difference LLR/variant |

**Which adapter declares which** (`loam-hf` adds `LOGPROB_EMBEDDING` for single-nucleotide
exports, which the released LOAM models are):

| Adapter | Models | `EMBEDDING` | `SEQUENCE_LOGLIKELIHOOD` | `MASKED_MARGINAL_LLR` | `LOGPROB_EMBEDDING` |
|---|---|:---:|:---:|:---:|:---:|
| `loam-hf` | LOAM-25M / 100M / 340M / 624M | ✅ | ✅ | — | ✅ |
| `evo2-arc` | Evo2-7B | ✅ | ✅ | — | ✅ |
| `evo` | Evo 1.5 | ✅ | ✅ | — | ✅ |
| `genomeocean` | GenomeOcean-100M / 500M | ✅ | ✅ | — | — |
| `glm2` | gLM2-150M / 650M | ✅ | — | ✅ | — |
| `prokbert` | ProkBERT-mini / mini-c | ✅ | — | ✅ | — |
| `ntv3` | NTv3-100M | ✅ | — | ✅ | — |
| `echo` | model-free test double | ✅ | ✅ | ✅ | ✅ |

`echo` declares everything by default; a spec can narrow it (`specs/echo-score-only.yaml`
declares only `sequence_loglikelihood`).

---

## Quick index

| Task | Benchmark | Capability | Scale | Metric (primary) |
|---|---|---|---|---|
| BacBench Gene Essentiality | `bacbench` (+`bacbench_layer_sweep`) | `EMBEDDING` | Gene | macro-mean AUROC |
| DGEB EC Classification (DNA) | `dgeb_ec` (both rows) | `EMBEDDING` | Gene | macro-F1 |
| RNAGym Prokaryotic DMS | `rnagym_prok` | `SEQUENCE_LOGLIKELIHOOD` / `MASKED_MARGINAL_LLR` | Variant (protein CDS) | macro-Spearman |
| Dummy (smoke) | `dummy_bench` | `SEQUENCE_LOGLIKELIHOOD` / `EMBEDDING` | Toy | mean_score / mean_embed_norm |

The benchmark **`loam_paper`** runs the five rows the LOAM paper reports: both essentiality
rows, both EC rows and RNAGym. Within it the single-layer rows cost nothing extra: each corpus
is embedded once at every layer, and the last-layer row is read from that pass.

---

## 1. BacBench Gene Essentiality

*Does a frozen encoder separate genes the cell can't live without from dispensable ones?*

- **Input**: each gene = its CDS + 128 bp upstream promoter (strand-aware), 55 genomes across
  train/val/test (the validation split is not used). Binary label `essential ∈ {0,1}`. Genes
  over the model's `max_context` are chunked into overlapping windows.
- **Adapter call** (2 rows, each one batched call over every window in train+test):
  - **base** (`bacbench-essentiality`): `embed(windows, layers="last", pool="mean")`
    → one `[H]` vector/window; windows meaned back per gene.
  - **layer-sweep** (`bacbench-essentiality-layer-sweep`): `embed(windows, layers="all", pool)`
    → `[n_layers, H]`/window; probes each layer independently, keeps best by AUROC.
- **Scoring**: `StandardScaler` + `sklearn.LogisticRegression` probe on train; per-test-genome
  **AUROC/AUPRC**; single-class genomes excluded. Macro mean/median across genomes.
- **Output**: `macro_mean_auroc` (primary), `macro_median_auroc`, `macro_mean/median_auprc`,
  pooled confusion (`precision/recall/specificity/f1/accuracy`, `tp/fp/tn/fn` at thr 0.5).
  Layer-sweep adds `best_layer` + per-layer AUROC/AUPRC. Metadata: `per_genome` breakdown,
  embed timing, probe hyperparams.
- **Notes**: sklearn re-derivation of the upstream torchmetrics (the vendored upstream script is
  kept as a reference, not run). Model-free k-mer floor: **0.6531** (k = 5), so read a score as
  headroom above it.

## 2. DGEB EC Classification (DNA)

*Do genes catalysing the same reaction (EC class) cluster enough for a linear probe?*

- **Input**: 640 enzyme CDSs (512 train / 128 test), 128 EC classes (DNA modality only). Length
  102–14,739 nt; over-context handled by toggleable strategy (`window` = overlapping
  mean-pool, default; `truncate` = central slice).
- **Adapter call** (`EMBEDDING`): one batched `embed(windows, layers=<layer|all>, pool="mean")`;
  windows meaned per gene. Base task uses one layer (`"last"`); layer-sweep uses `layers="all"`.
- **Scoring**: **vendored DGEB `logRegClassificationEvaluator`** (called char-for-char, not
  re-implemented): `LogisticRegression(max_iter=1000)` → `f1_score(average="macro")`. Optional
  `StandardScaler` (default **off**, matching DGEB).
- **Output**: `f1` (primary, macro-F1), `accuracy`, sizes. Layer-sweep adds `best_layer` +
  per-layer scores.
- **Notes**: metric is a true vendor with SHA256 pin + equivalence test. Model-free k-mer
  floor: **0.0191** (k = 4), close to chance for 128 classes, so EC is the cleanest test of
  learned representation here.

## 3. RNAGym Prokaryotic DMS

*Does a zero-shot variant-effect score correlate with measured mutational fitness?*

- **Input**: 11 prokaryotic protein-coding **deep-mutational-scanning** assays, 30,658 variant
  sequences + experimental `DMS_score`. WT rows dropped. U→T normalised (adapters see DNA).
- **Adapter call** (one batched call, branch by negotiated capability):
  - **AR path** (`SEQUENCE_LOGLIKELIHOOD`, primary): `score_sequences(seqs, reduction="mean")`
    → mean per-token causal log-likelihood `ℓ(x)`. Over-context variants skipped-and-counted.
  - **MLM fallback** (`MASKED_MARGINAL_LLR`, for bidirectional encoders): `score_variant_llr(variants)`
    → additive single-site masked-marginal LLR (Σ over mutated positions of `logit_mut − logit_wt`).
    **Substitutions only**; indels skipped-and-counted. Reference (WT) derived + verified per assay.
- **Scoring**: **vendored RNAGym `performance_fitness.py`** — per assay: abs-Spearman, symmetric
  median-split AUC, abs-MCC. Macro = plain mean over the scored assays (all same RNA type).
- **Output**: `macro_spearman` (primary), `macro_auc`, `macro_mcc`, per-assay `{DMS_ID}_spearman`.
  Metadata: `per_assay` table, `scoring_method` (AR vs MLM), `n_scored/n_missing` (split into
  over-context + unscoreable).
- **Notes**: method-agnostic metric, but AR and MLM scores are different estimators — rank
  within a scoring method first ([CAVEATS.md](CAVEATS.md#4-rnagym-causal-log-likelihood-vs-masked-marginal-llr)).
  Never silently truncates. No k-mer floor: a frequency vector is not a likelihood model.

## 4. Dummy (smoke tests)

*Infrastructure smoke tests — exercise the full run/leaderboard/capability path with no real data or GPU.*

- **`dummy`** (`SEQUENCE_LOGLIKELIHOOD`): 5 hard-coded toy sequences → `score_sequences(seqs, reduction="mean")`
  → 5 log-likelihoods. Output: `mean_score` (primary), `min_score`, `max_score`. Validates the
  end-to-end lifecycle + leaderboard.
- **`dummy-embed`** (`EMBEDDING`): 2 toy sequences → `embed(seqs, layers="last", pool="mean")`,
  shape `[2,1,H]`; L2-norm each. Output: `mean_embed_norm`. Validates **capability negotiation** —
  a score-only adapter yields a clean first-class `N/A` (named missing capability), not a crash.
- Run with `EchoAdapter` (deterministic, model-free) for the fastest possible CI check.

---

## Cross-cutting conventions

- **One GPU pass per task**: all sequences/windows across every split go in a single batched
  adapter call; regrouping/probing/scoring is CPU-side. Tasks that share a corpus (a row and its
  layer sweep) share that pass when they run in one benchmark.
- **Loud, never silent**: over-context sequences are counted+logged (never cropped silently);
  NaN embeddings, broken splits, single-class data → `ERROR`, never a misleading zero.
- **Normalisation** happens once in each loader: strip whitespace, uppercase, `U → T`.
- **Manifests are sha256-pinned**: any data drift raises loudly.
- **Metric provenance** — two styles:
  - *True vendor* (DGEB EC, RNAGym): upstream metric copied char-for-char and **called**,
    with SHA256 pin + equivalence test.
  - *Re-derivation* (BacBench): upstream torchmetrics re-implemented with sklearn (rank-equivalent);
    vendor copy kept as reference, not executed.
- **Layer sweeps** return all layers in one forward pass on every public adapter. The best
  layer is chosen on the test split and degenerate taps (pooled RMS ≤ 1e-12) are excluded
  first ([CAVEATS.md](CAVEATS.md#2-the-best-layer-is-chosen-on-the-test-split)).
