# Reading gLMBench numbers: five caveats

Five things to know before comparing two numbers from this benchmark, whether they come from
the LOAM paper ([`paper/reference_scores.csv`](../paper/reference_scores.csv), which holds
every tap of both sweeps and every RNAGym assay) or from your own run. Each one changes how a
bar should be read. None of them changes a number.

One rule covers all five: **ranks are within a task and within a readout.** The three tasks
report different metrics (macro-mean AUROC, macro-F1, macro-Spearman) on different scales with
different floors. Never average them, and never put a last-layer and a best-layer value in the
same ranking.

---

## 1. The model-free k-mer floor

The **k-mer floor** is the score a task gives to a feature vector built only from nucleotide
counts, with no model at all. [`scripts/kmer_floor.py`](../scripts/kmer_floor.py) produces it by
running the unmodified task classes over normalised k-mer frequency vectors (k = 1…6): the same
8,192-nt windowing, the same probe and the same vendored metric as any model row. The floor is
the best k. It runs on CPU in minutes and is deterministic; the paper's values are in
[`paper/kmer_floor.json`](../paper/kmer_floor.json).

| Task | Floor | Best k | Paper range, last layer → best layer |
|---|---:|---:|---|
| BacBench essentiality (macro-mean AUROC) | **0.6531** | 5 | 0.5565 – 0.7703 → 0.5758 – 0.7880 |
| DGEB EC, DNA (macro-F1) | **0.0191** | 4 | 0.0056 – 0.3008 → 0.0056 – 0.3939 |
| RNAGym prokaryotic DMS | not applicable | — | — |

RNAGym has no floor because it reads the LM head: a frequency vector is not a likelihood model,
so there is no model-free score to report. (A k-th-order Markov background model would give
one; that is a different baseline and is not implemented.) A layer sweep inherits its
single-layer task's floor, because the composition vector is the same at every tap.

Three ways to read a bar against its floor:

- **At or below the floor**, the model contributed nothing a counter could not. ProkBERT-mini-c
  scores 0.5565 (last layer) and 0.5758 (best layer) on essentiality, both below 0.6531, and
  0.0056 on EC, below 0.0191.
- **Above the floor**, the gap is the headroom, and the headroom is the only part of the score
  that is evidence about the model. On essentiality a counter already reaches 0.6531, so the
  best score in the paper sits only 0.135 above it: LOAM-340M's best layer (0.7722) is +0.119
  above the floor, Evo2-7B's (0.7880) +0.135, NTv3-100M's last layer (0.6789) +0.026.
- **A floor near zero** makes the task a clean test of learned representation. EC's floor is
  barely above what random guessing among 128 classes gives (about 1/128 ≈ 0.008), which is
  why EC separates the models most sharply.

## 2. The best layer is chosen on the test split

A `-layer-sweep` row fits one probe per tap on the train split, scores every tap on the test
split, and reports the tap with the highest **test** score. No held-out split is used for the
choice: EC has none, and essentiality's validation split is not embedded. A best-layer value is
therefore a **retrospective upper bound**, not a number fixed before looking at the test set.

Two consequences:

- **The bias grows with the number of taps and with the noise per tap.** Evo 1.5 and gLM2-650M
  sweep 33 taps; LOAM-25M sweeps 6. EC has 128 test genes, one per class, so a single flipped
  prediction moves macro-F1 by roughly 0.01, and the maximum over 33 noisy taps is more
  optimistic than the maximum over 6.
- **Last layer and best layer answer different questions**, and can differ a lot. On EC,
  Evo 1.5 moves from 0.0234 at its last layer to 0.3318 at its best tap, Evo2-7B from 0.0513 to
  0.3721, and LOAM-624M from 0.2974 to 0.3939. Report the two side by side, as the paper does;
  `reference_scores.csv` carries both, and every tap in between.

**Degenerate taps are excluded before the argmax.** A tap whose pooled RMS is at or below
1e-12 is numerically empty, and whatever it scores is float noise, so it cannot be chosen
([`tasks/_depth.py`](../src/glmbench/tasks/_depth.py),
[`diagnostics/depth_health.py`](../src/glmbench/diagnostics/depth_health.py)). Frozen taps (a
state equal to the previous tap's) are *not* excluded: their score is real, just redundant.
Each sweep row records its choice under `best_layer_selection` in the result metadata, with the
excluded taps and the tap the plain argmax would have picked. No tap was excluded in any of the
paper's 13 records.

## 3. Readout conventions: what "layer i" is in each model

gLMBench defines one tensor for every model: **tap i is the output of a block, before any
model-level final normalisation**, and `layers="last"` is the raw output of the final block.
Each adapter declares how it gets there (`readout`, `readout_evidence` and `readout_note` in
every record); all 13 paper records declare `block_output`.

This is not what `output_hidden_states=True` returns for every model. In decoders that end in a
final norm (LOAM, GenomeOcean, most Llama- or Mistral-style models) the last entry of
`hidden_states` is the final-normalised state, a different tensor with different pooled
geometry. To reproduce a gLMBench LOAM embedding outside the benchmark, follow
[USING_LOAM.md](USING_LOAM.md). How each model gets to its raw block output:

- **LOAM** (`loam-hf`): forward hooks on the token embedding and on every block; the final
  RMSNorm is never applied to embeddings.
- **GenomeOcean**: a Mistral-style pre-LN decoder, so the runner hooks the last block instead
  of reading `hidden_states[-1]`.
- **ProkBERT**: a pre-LN BERT encoder with a final LayerNorm outside the blocks, so the runner
  hooks the last block here too. The rule of thumb that BERT-style encoders are post-LN, with
  nothing to strip, does not hold for it.
- **gLM2**: the closing norm lives in the LM head, which the embedding load drops, so the hidden
  states are already raw block outputs.
- **NTv3**: a U-Net, read from its own hidden-state tuple, which holds each up-sampling state
  after the skip add (a hook on the block would see the branch before it). The states are at
  mixed resolution; the last one is at nucleotide resolution.
- **Evo 1.5**: its wrapper returns no hidden states, so the runner hooks the residual stream.
- **Evo2-7B** (`evo2-arc`): the taps are named in the spec. The paper's spec reads the residual
  stream at `blocks.{0,2,…,30,31}`. Other names give different tensors and a different model
  hash: `blocks.N.mlp.l3` (Arc's README suggests `blocks.28.mlp.l3` for embeddings) is an MLP
  branch output before the residual add, and `blocks.N.pre_norm` is a normalised read.

**Tap indices are not comparable across models.** Each model's tap axis is its own:

| Model | Taps | Tap 0 | Last tap |
|---|---:|---|---|
| LOAM-25M / 100M / 340M / 624M | 6 / 9 / 13 / 17 | embedding output | last block |
| GenomeOcean-100M / 500M | 13 / 15 | embedding output | last block |
| ProkBERT-mini / mini-c | 7 | embedding output | last block |
| Evo 1.5 | 33 | embedding output | last block |
| gLM2-150M / 650M | 30 / 33 | output of the **first block** (no embedding tap) | last block |
| NTv3-100M | 20 | full resolution; the axis runs down to 1/128 and back | final up-sampling state |
| Evo2-7B | 17 | `blocks.0` (output of the first block) | `blocks.31` |

For Evo2-7B a tap is **an index into the spec's 17-name list, not a block number**: its best
essentiality tap, 5, is `blocks.10`, and its best EC tap, 10, is `blocks.20`. Each Evo2 record
carries the map as `layer_id_to_name`. Compare tap *values* across models, not tap indices.

## 4. RNAGym: causal log-likelihood vs. masked-marginal LLR

RNAGym scores every variant zero-shot, by one of two methods chosen by what the model can do.
The record's `scoring_method` says which one produced the number.

| Scoring | What it computes | Paper models |
|---|---|---|
| sequence log-likelihood | mean per-token causal log-likelihood of the whole variant sequence | LOAM ×4, GenomeOcean ×2, Evo 1.5, Evo2-7B |
| masked-marginal LLR | Σ over mutated sites of logit(mut) − logit(wt), each site masked on the wild-type background | ProkBERT ×2, NTv3-100M, gLM2 ×2 |

Both feed the same vendored RNAGym metric: per-assay Spearman against measured fitness, averaged
over the assays. That puts them on one axis, but they are different estimators, and a gap
between a causal and a masked model mixes model quality with scoring method:

- The causal score predicts each base from the bases before it only, so the mutated base itself
  is predicted from its upstream context alone, which is short for a mutation near the start of
  a gene. The masked score sees both sides of each site, but treats a multi-site variant as a
  sum of independent sites.
- For GenomeOcean the mean runs over BPE tokens, and a substitution can change how the sequence
  is segmented. For ProkBERT-mini's overlapping 6-mers, one substitution touches up to 6 tokens,
  and all of them are masked.
- The per-assay Spearman is taken in absolute value (the vendored metric), so a score that is
  anti-correlated with fitness counts as much as a correlated one.

Rank within a scoring method first. The paper reports LOAM-624M (0.3176) and Evo 1.5 (0.3182),
both causal, as a tie.

## 5. Context limits and over-context handling

Every adapter declares a context limit, `max_context`, in nucleotides, and no task feeds a
model a longer input. Longer inputs are handled per task:

- **Essentiality** splits a gene longer than the context into overlapping windows of that
  length (32-nt overlap; windows under 32 nt dropped unless that leaves none), embeds each, and
  averages them into one vector per gene. The 163,944 train + test genes become 164,630 windows
  at an 8k context.
- **EC** does the same by default (`window`); `GLMBENCH_DGEB_EC_OVERFLOW=truncate` keeps the
  central `max_context` slice instead. The record counts the overflowing sequences as
  `n_train_overflow` / `n_test_overflow`.
- **RNAGym** never truncates or windows. A variant (causal scoring) or a reference (masked
  scoring) longer than the context is skipped and counted as `n_over_context`, and an assay with
  nothing scored drops out of the macro.
- **The k-mer floor** windows at 8,192 nt.

What that meant for the paper's models (all counts are in the records):

| Model | Context (nt) | Essentiality windows | EC over-context, train / test | RNAGym variants skipped |
|---|---:|---:|---:|---:|
| LOAM-25M … 624M | 8,191 | 164,630 | 4 / 1 | 0 |
| Evo 1.5, Evo2-7B | 8,192 | 164,630 | 4 / 1 | 0 |
| NTv3-100M | 12,288 | 164,367 | 2 / 0 | 0 |
| GenomeOcean-100M / 500M | 5,000 | 165,290 | 16 / 9 | 0 |
| gLM2-150M / 650M | 4,095 | 166,039 | 31 / 13 | 0 |
| ProkBERT-mini / mini-c | 1,020 | 259,403 | 301 / 77 | 697 |

- **LOAM's 8,191 nt** is its 8,192 positions minus the BOS token every input starts with. Most
  other limits are the trained window in tokens, converted to nucleotides: 8,192 for the Evo
  models, 1,024 BPE tokens for GenomeOcean (a conservative 5,000 nt), 4,096 tokens minus gLM2's
  strand marker, 1,024 tokens for ProkBERT. NTv3's 12,288 nt is a memory bound, not its trained
  limit.
- **ProkBERT's RNAGym macro covers 10 assays, not 11.** Its 1,020-nt window cannot hold the
  1,770-nt reference of Q837P4_ENTFA_Meier_2023, so all 697 of that assay's variants are
  skipped. Every other paper model scored all 30,658 variants.
- **A short context changes the input, not only the model.** A 1,020-nt model sees a long gene
  as many fragments averaged together, where an 8k model sees it whole, and that difference is
  part of what the essentiality and EC rows measure.

---

## The three comparisons the paper's claims rest on

When you reproduce the paper, check that these gaps survive, not only that each number is
close ([REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md)):

- **EC** (best layer): LOAM-624M 0.3939 vs. Evo2-7B 0.3721 (Δ 0.022).
- **Essentiality** (best layer): LOAM-340M 0.7722 vs. Evo 1.5 0.7779 vs. Evo2-7B 0.7880.
- **RNAGym**: LOAM-624M 0.3176 vs. Evo 1.5 0.3182 (Δ 0.0006, reported as a tie).

The first two are best-layer values, so the second caveat above applies to both. Per-task
details are in [TASKS_OVERVIEW.md](TASKS_OVERVIEW.md); per-model serving details are in
[HOW_MODELS_ARE_SERVED.md](HOW_MODELS_ARE_SERVED.md).
