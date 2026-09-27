# Task: RNAGym Prokaryotic DMS

> _Predict mutational fitness effects: score variants across 11 prokaryotic protein-coding deep-mutational-scanning (DMS) assays — by causal sequence log-likelihood (AR models) or masked-marginal LLR (MLM encoders, fallback) — and correlate model scores with measured fitness using the vendored RNAGym metric (method-agnostic abs-Spearman)._

**Registry key**: `rnagym-dms`
**Required capability**: `SEQUENCE_LOGLIKELIHOOD`, or `MASKED_MARGINAL_LLR` (MLM fallback)
**Benchmarks**: `rnagym_prok`, `loam_paper`

---

## Source

Adapted from **RNAGym** (Notin et al., "RNAGym: Large-scale Benchmarks for RNA Fitness and
Structure Prediction", bioRxiv 2025.06.16.660049; MIT licence).
The 11 prokaryotic protein-coding DMS assays and the scoring metric are taken
directly from that dataset. The upstream metric code (`performance_fitness.py`)
is vendored verbatim so gLMBench calls it rather than reimplements it. Individual assays keep
their primary-publication citations (e.g. Firnberg et al. 2014 and Jacquier et al. 2013 for
BLAT_ECOLX).

---

## Code and files

| File | Role |
|---|---|
| `src/glmbench/tasks/rnagym_dms.py` | Task class: `RnagymDmsTask`. Implements `prepare / run / score`. |
| `src/glmbench/tasks/rnagym_data.py` | Data loader: reads CSVs, enforces manifest sha256, normalises sequences (U→T); parses mutants and derives each assay's reference. |
| `src/glmbench/tasks/metrics.py` | Thin wrapper: calls vendored upstream; zero extra maths. |
| `src/glmbench/tasks/_vendor/rnagym/performance_fitness.py` | **Verbatim copy** of upstream metric authority. `calculate_metrics()` and `calculate_RNA_types_averages_with_se()` live here. |
| `src/glmbench/tasks/_vendor/rnagym/{NOTICE,SHA256SUMS,LICENSE}` | Upstream licence + sha256 of vendored file. |
| `src/glmbench/tasks/data/rnagym/reference_sheet_final.csv` | RNAGym's assay index (committed, MIT); the 11 assays are selected from it. |
| `src/glmbench/tasks/data/rnagym/prok_coding_manifest.yaml` | Per-file sha256 checksums, assay counts. |
| `src/glmbench/tasks/data/rnagym/processed_DMS_files/` | 11 DMS CSV files (fetched on demand, gitignored). |
| `src/glmbench/benchmarks/rnagym_prok.yaml` | Benchmark definition that includes only this task. |
| `scripts/fetch_rnagym_prok.py` | One-time fetch script. |

---

## The 11 assays

All 11 assays are prokaryotic protein-coding RNA (`mRNA-coding, PROK`), selected
programmatically from the reference sheet (`RNA_TYPE == "mRNA-coding"` and
`ASSAY_DESCRIPTION == "PROK"`), never from a hard-coded list.

| DMS ID | Organism | Variants | CDS length |
|---|---|---|---|
| BLAT_ECOLX_Firnberg_2014 | *E. coli* | 4 783 | 861 bp |
| BLAT_ECOLX_Jacquier_2013 | *E. coli* | 989 | 861 bp |
| CCDB_ECOLI_Adkar_2012 | *E. coli* | 1 176 | 306 bp |
| IF1_ECOLI_Kelsic_2016 | *E. coli* | 1 367 | 219 bp |
| RNC_ECOLI_Weeks_2023 | *E. coli* | 4 277 | 681 bp |
| F7YBW8_MESOW_Ding_2023 | *M. opportunistum* | 7 922 | 282 bp |
| Q837P4_ENTFA_Meier_2023 | *E. faecalis* | 697 | 1 770 bp |
| MLAC_ECOLI_MacRae_2023 | *E. coli* | 4 007 | 636 bp |
| ESTA_BACSU_Nutschel_2020 | *B. subtilis* | 2 172 | 639 bp |
| PSAE_PICP2_Tsuboyama_2023_1PSE | *Picosynechococcus* sp. PCC 7002 | 1 616 | 204 bp |
| BCHB_CHLTE_Tsuboyama_2023_2KRU | *C. tepidum* | 1 652 | 156 bp |

**Total: 30 658 variants.**

Wild-type rows (empty/NaN mutant field) are always dropped before scoring.

---

## What is being measured

The task tests whether a model's **zero-shot variant-effect score** correlates with
experimental DMS fitness. No training, no fine-tuning: the model scores from its
pre-training objective alone. The metric is **method-agnostic abs-Spearman** — it
rank-correlates whatever score the model produces against the measured fitness, exactly
the AR-vs-MLM comparison ProteinGym/RNAGym make.

Two scoring methods feed this one metric, selected by capability negotiation:

- **AR causal log-likelihood** (`SEQUENCE_LOGLIKELIHOOD`, the primary). The model assigns a
  causal log-likelihood `ℓ(x)` to each variant sequence.
- **MLM masked-marginal LLR** (`MASKED_MARGINAL_LLR`, the fallback for bidirectional encoders
  like gLM2 that have no causal likelihood). For a variant with mutated positions `M` on
  reference (WT) `r`:

  $$\mathrm{LLR} = \sum_{i \in M}\big[\log p(x_i{=}\mathrm{mut}_i \mid r_{\setminus i}) - \log p(x_i{=}\mathrm{wt}_i \mid r_{\setminus i})\big]$$

  where `r_\i` is the reference with position `i` replaced by `[MASK]`. Both terms share the
  same masked context, so the softmax normalizer cancels and each site reduces to the raw
  logit difference `logit_mut − logit_wt`. Higher = prefers the mutant (same sign as the AR
  score). **Substitutions only** (indels are skipped-and-counted); multi-site uses the
  additive single-site masked-marginal approximation (no joint masking). For a tokenizer with
  overlapping k-mers (ProkBERT-mini), a substitution touches up to k tokens and all of them
  are masked.

Each assay provides a set of single-mutant (or multi-mutant) variant sequences and their
measured fitness values; rank-correlation with the experimental values is the measured signal.
The chosen path is recorded in the result metadata as `scoring_method` and disclosed on the
leaderboard (CSV column + per-task footnote).

| Adapter | Scoring path |
|---|---|
| `loam-hf`, `evo2-arc`, `evo`, `genomeocean` | causal log-likelihood |
| `glm2`, `prokbert`, `ntv3` | masked-marginal LLR |
| `echo` | causal log-likelihood (it declares both; the primary wins) |

---

## Lifecycle in detail

### 1. `prepare()`

- Loads the 11 CSVs via `rnagym_data.load_assays()`.
- Verifies per-file sha256 checksums against the manifest (raises loudly on mismatch).
- Enforces column contract: `{mutant, sequence, DMS_score}`.
- Drops wild-type rows, checks all scores are finite floats.
- Normalises: strip whitespace, uppercase, `U → T` (done once here; adapters see DNA).

### 2. `run(adapter)`

`run` branches on the negotiated capability (`self._chosen_capabilities`, set by `evaluate`
before `run`). Both branches build one flat list across all 11 assays and make **exactly one
batched adapter call**, then scatter scores back per assay (preserving original order).

**AR path** (`SEQUENCE_LOGLIKELIHOOD`, primary):

- Determines `max_context`: explicit task arg → adapter's declared `max_context` → None.
- Partitions every assay's variant sequences into **in-context** and **over-context** sets.
- Calls once:

  ```
  scores = adapter.score_sequences(flat_seqs, reduction="mean")
  ```

  `reduction="mean"` returns the mean per-token causal log-likelihood:

  $$\ell(x) = \frac{1}{L}\sum_{t=1}^{L} \log p_\theta(x_t \mid x_{<t})$$

  For a multi-nucleotide tokenizer (GenomeOcean's BPE) the mean runs over tokens, and a
  substitution can change how the sequence is segmented.

**MLM fallback path** (`MASKED_MARGINAL_LLR`):

- Parses each mutant string (`rnagym_data.parse_mutant`, 1-based→0-based, `U→T`-normalized).
- **Derives + verifies the reference (WT)** per assay (`rnagym_data.assay_reference`): reverts
  each variant's mutations and asserts all variants reconstruct the identical WT and that
  `sequence[pos0] == mut` (loud `AssayContractError` otherwise) — the loader still drops WT rows.
- Skips-and-counts **unscoreable** variants (indel/non-substitution) and over-`max_context`
  references; calls once:

  ```
  scores = adapter.score_variant_llr([{"reference": ref, "mutations": [[pos0, wt, mut], ...]}, ...])
  ```

- Both branches log (never crash on) skipped variants; they contribute to `n_missing` /
  `n_missing_predictions` (`metadata` breaks this into `n_over_context` + `n_unscoreable`).

### 3. `score()`

For each assay **i**, let:
- $\hat{y}_i$ = model scores for the scored variants
- $y_i$ = experimental DMS fitness values for the same variants

Three statistics are computed via the vendored upstream:

**Spearman correlation** (absolute value):

$$\rho_i = \left|\text{Spearman}(\hat{y}_i,\, y_i)\right|$$

**AUC** (median-split, symmetric):

Binarise fitness at its median; compute AUC of $\hat{y}_i$ predicting the high-fitness class:

$$\text{AUC}_i = \max\bigl(\text{AUC}(\hat{y}_i, b_i),\; 1 - \text{AUC}(\hat{y}_i, b_i)\bigr)$$

The symmetry makes the metric invariant to which half of the distribution is
called "positive".

**MCC** (absolute value):

Binarise both $\hat{y}_i$ (at its median) and $y_i$ (at its median); compute
Matthews Correlation Coefficient and take the absolute value.

**Macro aggregation** across assays (via `metrics.macro()`):

Because all 11 assays are the same RNA type (`mRNA-coding`), the type-mean macro
degenerates to a plain mean:

$$\text{macro\_spearman} = \frac{1}{11} \sum_{i=1}^{11} \rho_i$$

$$\text{macro\_auc} = \frac{1}{11} \sum_{i=1}^{11} \text{AUC}_i, \qquad \text{macro\_mcc} = \frac{1}{11} \sum_{i=1}^{11} \text{MCC}_i$$

Assays where all variants exceed `max_context` contribute NaN to the per-assay
table and are excluded from the macro (pandas skipna), so the mean then runs over fewer
assays.

---

## Output

### Metrics (for the leaderboard)

| Key | Description |
|---|---|
| `macro_spearman` | **Primary metric.** Macro-mean Spearman across the scored assays. |
| `macro_auc` | Macro-mean AUC (median-split symmetric). |
| `macro_mcc` | Macro-mean MCC (absolute). |
| `{DMS_ID}_spearman` | Per-assay Spearman (11 keys). |
| `blat_ecolx_spearman` | Shorthand for the first BLAT\_ECOLX assay. |
| `n_assays`, `n_missing_predictions` | Assay count and skipped variants. |

### Metadata (stored, not ranked)

- `per_assay`: list of dicts — `dms_id`, `rna_type`, `n_variants`, `n_missing`, `spearman`, `auc`, `mcc`
- `n_scored`, `n_missing`, `n_over_context`, `n_unscoreable`, `scoring_method`, `reduction`,
  `max_context`, `task_version`

---

## Data location

| What | Where |
|---|---|
| Manifest (sha256 + counts) | `src/glmbench/tasks/data/rnagym/prok_coding_manifest.yaml` |
| Reference sheet | `src/glmbench/tasks/data/rnagym/reference_sheet_final.csv` (committed) |
| Processed DMS CSVs | `src/glmbench/tasks/data/rnagym/processed_DMS_files/` (gitignored; fetched) |
| Tiny CI fixture | `tests/data/rnagym_tiny/` (committed; three synthetic toy assays, not RNAGym data) |
| Fetch script | `scripts/fetch_rnagym_prok.py` |

To fetch the full data (one-time):

```bash
python scripts/fetch_rnagym_prok.py                      # Hugging Face mirror (default)
python scripts/fetch_rnagym_prok.py --source official    # the RNAGym zip from marks.hms.harvard.edu
```

The default backend downloads each assay from the Hugging Face mirror `Marks-lab/RNAgym`, and
the reference sheet comes from the RNAGym GitHub repository (`MarksLab-DasLab/RNAGym`); both are
read at a pinned commit (see the fetch script). The two backends may emit byte-different CSVs,
so the manifest's sha256 check is what confirms you have the paper's data. The fixture in
`tests/data/rnagym_tiny/` is always present and is used by unit tests without requiring a
fetch.

---

## How to run

```bash
# 1. Fetch the full DMS data (one-time; ~18 MB)
python scripts/fetch_rnagym_prok.py

# 2. Run on a specific model
glmbench run \
  --model specs/loam/LOAM-100M.yaml \
  --benchmark rnagym_prok \
  --results results/

# Dry-run: print the runner command without executing
glmbench run \
  --model specs/loam/LOAM-100M.yaml \
  --benchmark rnagym_prok \
  --dry-run

# Build the leaderboard from all stored results
glmbench leaderboard \
  --benchmark rnagym_prok \
  --results results/ \
  --out results/rnagym_prok/LEADERBOARD.md
```

**Capability availability by adapter:**

| Task | `loam-hf` | `evo2-arc` | `evo` | `genomeocean` | `glm2` | `prokbert` | `ntv3` | `echo` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `rnagym-dms` | ✅ AR | ✅ AR | ✅ AR | ✅ AR | ✅ MLM | ✅ MLM¹ | ✅ MLM | ✅ AR |

¹ ProkBERT's 1,020-nt context cannot hold Q837P4_ENTFA_Meier_2023's 1,770-nt reference, so
that assay's 697 variants are skipped and its macro covers 10 assays. Every other model in the
paper scores all 30,658 variants.

---

## Reading the numbers

- **Rank within a scoring method first.** AR and MLM scores are different estimators; the
  shared metric puts them on one axis but does not make them the same measurement
  ([CAVEATS.md](CAVEATS.md#4-rnagym-causal-log-likelihood-vs-masked-marginal-llr)).
- **The Spearman is absolute.** A score anti-correlated with fitness counts as much as a
  correlated one (the vendored upstream metric).
- **There is no k-mer floor for this task**: a frequency vector is not a likelihood model.
- The paper's per-assay values for every model are in
  [`paper/reference_scores.csv`](../paper/reference_scores.csv).

---

## Design notes

- **One batched call**: All 30,658 variants are sent in a single `score_sequences()` (or
  `score_variant_llr()`) call. Runner spin-up is expensive; per-assay round-trips would be
  impractical.
- **Never silent truncation**: Sequences that exceed `max_context` are excluded and
  counted in `n_missing`; they never get silently cropped to fit.
- **Metric authority is vendored**: The Spearman/AUC/MCC code is a byte-identical
  copy of the upstream. gLMBench calls it; it does not reimplement it. This is the
  single biggest correctness safeguard.
- **`reduction="mean"` vs `reduction="sum"`**: The task default is `"mean"` (mean
  per-token log-likelihood), which normalises for sequence length. This matches the
  standard zero-shot fitness prediction practice and the upstream RNAGym evaluation.
