# Task: BacBench Gene Essentiality

> _Predict whether bacterial genes are essential for survival: encode each gene's DNA, extract a fixed-length representation, fit a logistic-regression probe, and report per-genome AUROC/AUPRC aggregated to a macro mean/median._

This document covers two related tasks that share the same data and lifecycle but
differ in which layers they read from the model:

| Registry key | Capability | What it probes |
|---|---|---|
| `bacbench-essentiality` | `EMBEDDING` | Frozen hidden-state embeddings at the last layer |
| `bacbench-essentiality-layer-sweep` | `EMBEDDING` | All layers in one forward pass; picks the best |

Benchmarks: `bacbench`, `bacbench_layer_sweep`, and `loam_paper` (both rows, sharing one
forward pass).

---

## Source

Adapted from **BacBench** (Wiatrak et al., "BacBench: Evaluating Genomic Language Models for
Bacteria"; Apache-2.0).

BacBench frames bacterial gene essentiality as a DNA language model probe: encode
each gene as a DNA sequence, extract a fixed-length representation from the model,
fit a logistic regression classifier on a train split, and evaluate on held-out
genomes. Performance is reported per-genome (AUROC, AUPRC) and aggregated to a
macro mean/median.

The upstream pipeline uses PyTorch-Lightning + torchmetrics. Because those are not
torch-free, gLMBench re-derives the metrics with `scikit-learn`
(`LogisticRegression`, `roc_auc_score`, `average_precision_score`). The vendor
copy (`_vendor/bacbench_essentiality/run_train_cls.py`, upstream commit `64b7d458`) is the
reference implementation — kept for documentation, not executed.

---

## Code and files

| File | Role |
|---|---|
| `src/glmbench/tasks/bacbench_essentiality.py` | Both task classes: `BacBenchEssentialityTask` (plus the shared windowing and probe helpers) and `BacBenchEssentialityLayerSweepTask`. |
| `src/glmbench/tasks/bacbench_essentiality_data.py` | Data loader: reads per-split CSVs, enforces manifest sha256, normalises sequences. |
| `src/glmbench/tasks/bacbench_essentiality_metrics.py` | sklearn-based AUROC/AUPRC and confusion-matrix helpers. |
| `src/glmbench/tasks/_vendor/bacbench_essentiality/run_train_cls.py` | Verbatim copy of upstream training script (reference only). |
| `src/glmbench/tasks/data/bacbench_essentiality/manifest.yaml` | Per-file sha256 checksums, genome/gene counts per split. |
| `src/glmbench/tasks/data/bacbench_essentiality/` | Train / validation / test CSV files (one CSV per split; fetched, gitignored). |
| `scripts/fetch_bacbench_essentiality.py` | One-time fetch: builds the per-gene sequences and the CSVs. |
| `src/glmbench/benchmarks/bacbench.yaml` | Benchmark: `bacbench-essentiality`. |
| `src/glmbench/benchmarks/bacbench_layer_sweep.yaml` | Benchmark: `bacbench-essentiality-layer-sweep`. |

---

## Data

**Source:** HuggingFace dataset `macwiatrak/bacbench-essential-genes-dna` (Apache-2.0), read
at a pinned commit (see the fetch script). Its labels come from the Database of Essential Genes
(DEG, CC-BY-4.0) and its sequences from NCBI GenBank/RefSeq. The genus-disjoint splits are the
dataset's native splits.

**Three splits, 55 genomes total:**

| Split | Genomes | Genes | Essential genes | Essential % |
|---|---|---|---|---|
| Train | 37 | 136 278 | 14 167 | 10.4% |
| Validation | 9 | 29 269 | 3 886 | 13.3% |
| Test | 9 | 27 666 | 3 573 | 12.9% |

The validation split is fetched but not embedded — it exists in the upstream pipeline to tune
classification thresholds, which gLMBench does not need since it uses a fixed
logistic regression probe.

**Per-gene sequence construction** (BacBench DNA-LM recipe):

Each gene is represented as its CDS plus 128 bp of upstream promoter sequence,
extracted strand-aware from the reference genome (no reverse complement). This matches the
exact recipe used by the upstream BacBench paper and ensures apples-to-apples comparison.

**Normalisation**: strip whitespace, uppercase, `U → T` (once, in the loader).

**Labels**: Binary — `1` = essential, `0` = non-essential. A gene is essential if
it was found to be lethal when knocked out under standard growth conditions.

---

## What is being measured

The task tests whether a frozen DNA language model encoder captures the biological
signal of gene essentiality — i.e., whether genes the cell cannot live without
look meaningfully different in embedding space from dispensable genes.

No fine-tuning occurs. The model is used purely as a feature extractor; a simple
linear (logistic regression) probe is trained on the train split and evaluated
on held-out genomes from the test split.

---

## Lifecycle in detail

Both task variants share the same `prepare()` step and the same `score()`
probe-and-evaluate logic. Only `run()` differs (which layers the adapter is asked for).

### 1. `prepare()`

- Loads train and test CSV files via `bacbench_essentiality_data.load_genomes()`.
- Verifies sha256 integrity for each file (raises loudly on drift).
- Enforces column contract: `{genome_name, genus, essential, sequence}`.
- Checks `essential` is `"0"` or `"1"` for every row.
- Normalises sequences (strip / uppercase / U→T).

### 2. `run(adapter)` — base task (`bacbench-essentiality`)

Each gene may be longer than the model's context window. The windowing recipe
(from BacBench) chunks genes that exceed `max_context` (the task argument if set, else the
adapter's declared `max_context`, in nucleotides):

```
windows = chunk(gene_seq, size=max_context, overlap=window_overlap)
# overlap default = 32 nt; min_seq_len = 32 nt
```

Windows shorter than `min_seq_len` are dropped; if that would drop every window of a gene, the
gene is kept whole.

Steps:

1. Build a **flat list of all windows** across every gene in both train and test
   genomes (batch-first: one list, one adapter call).
2. Make exactly one batched adapter call:

   ```
   result = adapter.embed(flat_windows, layers="last", pool="mean")
   # result.embeddings: shape [n_windows, 1, H]  (one layer, mean-pooled)
   ```

3. **Regroup per gene**: For genes that were windowed, mean the window embeddings
   back to one vector:

   $$\mathbf{h}_{\text{gene}} = \frac{1}{W}\sum_{w=1}^{W} \mathbf{h}_w \in \mathbb{R}^H$$

4. **Regroup per genome**: Collect gene vectors into `[n_genes, H]` matrices for
   each genome.

At an 8k-nucleotide context the 163,944 train + test genes become 164,630 windows; a model
with a shorter context sees long genes as more, shorter fragments (1,020-nt ProkBERT: 259,403
windows). Every paper model's count is in its record (`n_windows_embedded`).

### 2b. `run(adapter)` — layer-sweep task (`bacbench-essentiality-layer-sweep`)

Same windowing as above. The adapter is asked for **all layers** in one forward
pass:

```
result = adapter.embed(flat_windows, layers="all", pool=self.pool)
# result.embeddings: shape [n_windows, n_layers, H]
```

Windows are regrouped per gene (mean across windows) and then per genome, producing
`{layer_id → [genome → [n_genes, H] matrix]}`. The pooled vector of every window at every
layer is held in host memory, so RAM grows with windows × layers × hidden size.

Which layers are swept can be controlled:

- Default: all layers returned by the model (`layers="all"`).
- Environment variable `GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS` — accepts
  range syntax (`"6-12"`) or comma-separated values (`"0,4,8,12"`).
- Constructor arg `sweep_layers`.

The pool used can be overridden via `GLMBENCH_BACBENCH_LAYER_SWEEP_POOL`
(default `"mean"`).

When both rows run in one benchmark (as in `loam_paper`), the sweep's forward pass serves the
base row too: the last layer is sliced from it, so the single-layer row costs no GPU time.

### 3. `score()`

Both tasks use the same probe-and-evaluate procedure (only the feature matrices differ):

**Probe fitting** (on train split):

```python
scaler = StandardScaler().fit(X_train)       # normalise features
probe  = LogisticRegression(C=probe_C, max_iter=probe_max_iter,
                            random_state=seed)   # C=1.0, max_iter=1000, seed=1
probe.fit(scaler.transform(X_train), y_train)
```

`X_train` is the concatenation of all per-genome feature matrices (train split).
`y_train` is the concatenation of all binary labels.

**Per-genome evaluation** (test split):

For each test genome $g$ with features $X_g \in \mathbb{R}^{n_g \times H}$ and
labels $y_g \in \{0,1\}^{n_g}$:

$$\hat{p}_g = \text{probe}.\text{predict\_proba}(\text{scaler}.\text{transform}(X_g))[:,\,1]$$

$$\text{AUROC}_g = \text{roc\_auc\_score}(y_g,\; \hat{p}_g)$$
$$\text{AUPRC}_g = \text{average\_precision\_score}(y_g,\; \hat{p}_g)$$

Genomes where $y_g$ is single-class (all essential or all non-essential) have
undefined AUROC/AUPRC and are excluded from aggregation.

**Macro aggregation** across $G$ scored test genomes:

$$\text{macro\_mean\_auroc} = \frac{1}{G}\sum_{g=1}^{G} \text{AUROC}_g$$
$$\text{macro\_median\_auroc} = \text{median}_{g}\bigl(\text{AUROC}_g\bigr)$$

Mean and median AUPRC are computed identically. Mean AUROC is the primary metric
(matching the upstream paper).

**Confusion matrix** (pooled across all test genes, at `threshold = 0.5`):

Summed TP/FP/TN/FN → precision, recall, specificity, F1, accuracy.

**Layer-sweep only**: The above procedure is repeated independently for each layer.
The best layer is selected by `macro_mean_auroc` on the test split, after excluding
degenerate layers (pooled RMS ≤ 1e-12, whose score would be float noise). Its full metric set
becomes the task's primary output; all layers' AUROC/AUPRC are also reported.

---

## Output

### Metrics (for the leaderboard)

| Key | Description | Variants |
|---|---|---|
| `macro_mean_auroc` | **Primary.** Mean AUROC across test genomes. | both |
| `macro_median_auroc` | Median AUROC across test genomes. | both |
| `macro_mean_auprc` | Mean AUPRC across test genomes. | both |
| `macro_median_auprc` | Median AUPRC across test genomes. | both |
| `precision`, `recall`, `specificity`, `f1`, `accuracy` | Pooled test confusion. | both |
| `tp`, `fp`, `tn`, `fn` | Raw confusion counts at threshold 0.5. | both |
| `n_test_genes`, `n_test_genomes_scored` | Test-set sizes. | both |
| `best_layer` | Layer index with highest macro\_mean\_auroc. | layer-sweep only |
| `layer_{id}_macro_mean_auroc` | Per-layer AUROC (one key per swept layer). | layer-sweep only |
| `layer_{id}_macro_mean_auprc` | Per-layer AUPRC. | layer-sweep only |

### Metadata (stored, not ranked)

- `per_genome`: per-test-genome breakdown — name, genus, n_genes, n_essential, scored flag, auroc, auprc
- `n_windows_embedded`, `embedding_dim`, `embed_seconds`, `embed_seconds_per_window`
- `threshold`, `probe_C`, `probe_seed`, `max_context`, `feature_source`, `task_version`
- Layer-sweep only: `swept_layers`, `best_layer`, `per_layer` table, `best_layer_selection`
  (considered and excluded layers), `depth_health` (per-layer RMS and inter-layer cosine,
  computed from the sweep's own array)

---

## Data location

| What | Where |
|---|---|
| Manifest | `src/glmbench/tasks/data/bacbench_essentiality/manifest.yaml` |
| Split CSVs | `src/glmbench/tasks/data/bacbench_essentiality/essential_genes_{train,validation,test}.csv` (gitignored; fetched) |
| Tiny CI fixture | `tests/data/bacbench_essentiality_tiny/` (committed, synthetic) |
| Vendor reference | `src/glmbench/tasks/_vendor/bacbench_essentiality/run_train_cls.py` |
| Fetch script | `scripts/fetch_bacbench_essentiality.py` |

The manifest path can be overridden at runtime via the environment variable
`GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST` — useful for restricting a run to a
subset of genomes without touching the code (`scripts/make_essentiality_subset.py` writes such
a subset).

---

## How to run

```bash
# One-time data fetch (needs HuggingFace + `datasets`)
python scripts/fetch_bacbench_essentiality.py

# Base embedding task
glmbench run \
  --model specs/loam/LOAM-100M.yaml \
  --benchmark bacbench \
  --results results/

# Layer sweep (all layers in one pass)
glmbench run \
  --model specs/external/genomeocean-100M.yaml \
  --benchmark bacbench_layer_sweep \
  --results results/

# Restrict the layer sweep to a subset (e.g., layers 6–12)
GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS="6-12" \
glmbench run \
  --model specs/external/genomeocean-100M.yaml \
  --benchmark bacbench_layer_sweep \
  --results results/

# Dry-run: inspect the runner command without executing
glmbench run \
  --model specs/loam/LOAM-100M.yaml \
  --benchmark bacbench \
  --dry-run
```

**Capability availability by adapter:**

| Task | `loam-hf` | `evo2-arc` | `evo` | `genomeocean` | `glm2` | `prokbert` | `ntv3` | `echo` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bacbench-essentiality` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `bacbench-essentiality-layer-sweep` | ✅ | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Every public adapter returns all requested layers from a single forward pass.
¹ `evo2-arc` sweeps the taps named in its spec (17 for `specs/external/evo2-arc-7b-resid.yaml`).

---

## Design notes

- **One GPU pass per task**: Both tasks make exactly one batched adapter call for all windows
  across all 46 train + test genomes, and share it when they run together.
- **Windowing is never silent**: Genes that are windowed contribute multiple
  windows to the batch; after the embed call, windows are averaged back to one
  vector per gene. The number of windows embedded is reported in metadata.
- **Probe is fixed across both variants**: Always `sklearn.LogisticRegression`
  with the same hyperparameters, trained on the same train split. The only thing
  that changes is the feature vector.
- **Metric re-derivation**: The upstream uses torchmetrics, which is not torch-free.
  AUROC and AUPRC from `sklearn.metrics` are rank-equivalent (same correctness, different
  implementation). The vendor copy is kept as the reference.
- **Single-class genome handling**: If a test genome happens to have only essential
  or only non-essential genes (AUROC/AUPRC undefined), it is excluded from macro
  aggregation with a flag in `per_genome` metadata.
- **The probe does not always converge**: lbfgs can stop at 1,000 iterations on some layers,
  so the last digits of a score can depend on the scikit-learn and BLAS build
  ([REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md)).
- **Read a score against the k-mer floor**: a model-free k-mer counter scores
  **0.6531** macro-mean AUROC (k = 5) through this task's own code, so only the headroom above
  it is evidence about a model. A sweep's best layer is selected on the test split, so report
  it next to the last-layer value ([CAVEATS.md](CAVEATS.md)).
