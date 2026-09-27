# Task: DGEB EC Classification (DNA)

> _Predict an enzyme's EC (Enzyme Commission) number from its gene's DNA: mean-pool a fixed-length embedding, fit a logistic-regression probe, and report macro-F1 on a held-out split (DNA/NA modality only)._

Two related tasks that share the same data and lifecycle but differ in how the embedding
layer is chosen:

| Registry key | Capability | What it probes |
|---|---|---|
| `dgeb-ec-classification-dna` | `EMBEDDING` | Frozen embeddings at a single configurable layer (default `"last"`) |
| `dgeb-ec-classification-dna-layer-sweep` | `EMBEDDING` | All layers in one forward pass; reports the best-F1 layer |

Benchmarks: `dgeb_ec` (both rows) and `loam_paper`.

---

## Source

Adapted from **DGEB** — the Diverse Genomic Embedding Benchmark (West-Roberts, Kravitz,
Jha, Cornman, Hwang; bioRxiv 2024.07.10.602933; Apache-2.0).

DGEB's *EC Classification* frames Enzyme Commission number prediction as an embedding
probe: encode each protein-coding gene, mean-pool a fixed-length representation from the
model, fit a logistic-regression classifier on a train split, and evaluate macro-F1 on a
held-out test split. DGEB ships **two** modalities — amino-acid (`tattabio/ec_classification`)
and nucleic-acid (`tattabio/ec_classification_dna`). gLMBench implements the **NA/DNA
variant only** (genomic LMs ingest nucleotides; no adapter ingests protein).

DGEB's classification evaluator is already torch-free (sklearn `LogisticRegression` +
`f1_score`/`accuracy_score`), so unlike BacBench it is **vendored verbatim and called** —
this is a true metric vendor, not a re-derivation. See
`_vendor/dgeb/NOTICE` for the one inert deviation (torch seeding removed from the
`Evaluator` base, which cannot change a score fixed by `random_state`).

---

## Code and files

| File | Role |
|---|---|
| `src/glmbench/tasks/dgeb_ec_classification.py` | Both task classes + shared base (`DGEBECClassificationBase`), the window/truncate chunkers, and `parse_layer_list`. |
| `src/glmbench/tasks/dgeb_ec_data.py` | Data loader: reads per-split CSVs, enforces manifest sha256, normalises sequences. |
| `src/glmbench/tasks/dgeb_ec_metrics.py` | Thin wrapper that *calls* the vendored DGEB evaluator (no math of its own). |
| `src/glmbench/tasks/_vendor/dgeb/evaluators.py` | Char-for-char copy of upstream `logRegClassificationEvaluator` (the metric authority). |
| `src/glmbench/tasks/_vendor/dgeb/{NOTICE,SHA256SUMS,LICENSE}` | Provenance (commit `e284167`, Apache-2.0) + integrity pin. |
| `src/glmbench/tasks/data/dgeb_ec_dna/manifest.yaml` | Per-file sha256, row counts, length stats, n_label_classes. |
| `scripts/fetch_dgeb_ec_dna.py` | One-time fetch of the two splits. |
| `src/glmbench/benchmarks/dgeb_ec.yaml` | Benchmark: both EC tasks. |

---

## Data

**Two splits, 640 genes total, 128 EC classes:**

| Split | Genes | Per class |
|---|---|---|
| Train | 512 | 4 |
| Test | 128 | 1 |

- Source: HuggingFace `tattabio/ec_classification_dna`, revision `cd61c74b…` (pinned in the
  fetch script; DGEB, Apache-2.0). Underlying coding sequences are from UniProtKB "reviewed"
  entries (CC BY 4.0).
- Columns (upstream): `Entry` (UniProt accession), `Sequence` (CDS nucleotides), `Label`
  (EC class string). Vendored CSVs use lowercase headers `entry, sequence, label`.
- Sequence length: **102 nt to 14,739 nt** (mean ~1.6 kb in train, ~1.9 kb in test) — the max
  **exceeds the context of every model in the paper**, so over-context handling matters (see
  Lifecycle).
- **Normalisation**: strip whitespace, uppercase, `U → T` (once, in the loader).

---

## What is being measured

Whether a frozen genomic language model encodes enzyme function: do genes that catalyse
the same reaction (same EC class) cluster in embedding space well enough for a linear probe
to separate 128 classes? No fine-tuning — the model is a pure feature extractor; a logistic
regression probe is fit on the train split and evaluated on the held-out test split.

---

## Lifecycle in detail

Both variants share `prepare()` and the embed-and-group machinery (`_embed`); only the
`layers` argument and the per-layer scoring differ.

### 1. `prepare()`

- Loads train + test CSVs via `dgeb_ec_data.load_ec_splits()`.
- Verifies sha256 integrity for each file (raises `ManifestHashError` on drift).
- Enforces the column contract `{entry, sequence, label}` (raises
  `ECClassificationContractError`), drops empty rows, normalises sequences.

### 2. `run(adapter)`

Each coding sequence may exceed the model's context window (`max_context`: the task argument
if set, else the adapter's declared value, in nucleotides). The over-context strategy is
**toggleable** (`overflow_strategy`, env `GLMBENCH_DGEB_EC_OVERFLOW`):

- **`window`** (default): overlapping chunks of length `max_context` (overlap 32, min 32),
  mean-pooled back to one vector — never drops an instance.
- **`truncate`**: keep the **central** `max_context`-nt slice of the sequence (DGEB-style,
  but centred rather than head-truncated). One window per instance.

Either way every instance reduces to exactly one feature vector. The number of overflowing
sequences is counted and logged (never silent). At an 8k-nucleotide context 4 train and 1 test
sequence overflow; at ProkBERT's 1,020 nt, 301 and 77. Each record carries its counts as
`n_train_overflow` / `n_test_overflow`.

Steps:

1. Build a **flat list of all windows** across train + test (batch-first: one list).
2. Make exactly one batched adapter call:
   ```
   result = adapter.embed(flat_windows, layers=<layer|all>, pool="mean")
   ```
   - Single-layer task: `layers=[layer]` or `"last"` (default).
   - Layer-sweep task: `layers="all"` (or a restricted list).
3. **Regroup per instance**: mean the window vectors back to one `[n_layers, H]` vector.
4. Split into `[n_train, n_layers, H]` and `[n_test, n_layers, H]` stacks + the label lists.

NaN embeddings and a 1:1 length-mismatch both raise loudly. When both rows run in one
benchmark (`dgeb_ec`, `loam_paper`), they share one forward pass: the sweep embeds every layer
and the single-layer row is read from it.

### 3. `score()`

**Optional standardisation** (`standard_scale`, env `GLMBENCH_DGEB_EC_SCALE`; **default
off** — DGEB passes raw embeddings): `StandardScaler` is fit on train, applied to both.

**Probe + metric** — through the vendored DGEB evaluator (one layer for the base task; once
per layer for the sweep):

```python
clf = LogisticRegression(random_state=seed, max_iter=1000)   # DGEB defaults; seed=42
clf.fit(X_train, y_train)
y_pred   = clf.predict(X_test)
f1       = f1_score(y_test, y_pred, average="macro")          # primary
accuracy = accuracy_score(y_test, y_pred)
```

`primary_metric` is **macro-F1**, matching the DGEB paper's reported EC metric.
(The evaluator emits `ap` only for binary tasks; the 128-class EC set never triggers it.)

**Layer-sweep only**: the probe runs once per layer; the best layer is chosen by macro-F1 on
the test split, after excluding degenerate layers (pooled RMS ≤ 1e-12, whose score would be
float noise). Its `f1`/`accuracy` become the headline, and every layer's `f1`/`accuracy` is
also reported.

A train split with <2 EC classes, or a NaN F1, returns `ERROR` (not a poisoned `OK` row).

---

## Output

### Metrics (for the leaderboard)

| Key | Description | Variants |
|---|---|---|
| `f1` | **Primary.** Macro-F1 on the test split. | both |
| `accuracy` | Accuracy on the test split. | both |
| `n_train`, `n_test`, `n_classes` | Dataset sizes. | both |
| `best_layer` | Layer index with the highest macro-F1. | layer-sweep only |
| `layer_{id}_f1`, `layer_{id}_accuracy` | Per-layer scores (one key per swept layer). | layer-sweep only |

### Metadata (stored, not ranked)

- `overflow_strategy`, `n_train_overflow`, `n_test_overflow`, `n_windows_embedded`
- `pool`, `standard_scale`, `max_context`, `max_iter`, `probe_seed`, `embedding_dim`, `embed_seconds`
- `layer` / `feature_source` (base) or `swept_layers`, `best_layer`, `per_layer` table,
  `best_layer_selection`, `depth_health` (sweep)
- `task_version`, `probe` (provenance string)

---

## Data location

| What | Where |
|---|---|
| Manifest | `src/glmbench/tasks/data/dgeb_ec_dna/manifest.yaml` |
| Split CSVs | `src/glmbench/tasks/data/dgeb_ec_dna/ec_classification_dna_{train,test}.csv` (gitignored; fetched) |
| Tiny CI fixture | `tests/data/dgeb_ec_dna_tiny/` (committed; 3 composition-separable classes) |
| Vendor authority | `src/glmbench/tasks/_vendor/dgeb/evaluators.py` |
| Fetch script | `scripts/fetch_dgeb_ec_dna.py` |

Runtime env seams (build tasks without a custom YAML): `GLMBENCH_DGEB_EC_MANIFEST`,
`GLMBENCH_DGEB_EC_LAYER`, `GLMBENCH_DGEB_EC_OVERFLOW`, `GLMBENCH_DGEB_EC_SCALE`,
`GLMBENCH_DGEB_EC_SWEEP_LAYERS`, `GLMBENCH_DGEB_EC_SWEEP_POOL`.

---

## How to run

```bash
# One-time data fetch (box with HuggingFace + `datasets`)
python scripts/fetch_dgeb_ec_dna.py

# Both EC rows: last layer + layer sweep, one forward pass
glmbench run \
  --model specs/loam/LOAM-100M.yaml \
  --benchmark dgeb_ec \
  --results results/

# Pick the single-layer row's layer without a custom YAML
GLMBENCH_DGEB_EC_LAYER="6" \
glmbench run --model specs/external/genomeocean-100M.yaml --benchmark dgeb_ec --results results/

# Truncate-central instead of window-and-mean for over-context genes
GLMBENCH_DGEB_EC_OVERFLOW="truncate" \
glmbench run --model specs/loam/LOAM-100M.yaml --benchmark dgeb_ec --results results/
```

**Capability availability by adapter:**

| Task | `loam-hf` | `evo2-arc` | `evo` | `genomeocean` | `glm2` | `prokbert` | `ntv3` | `echo` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dgeb-ec-classification-dna` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `dgeb-ec-classification-dna-layer-sweep` | ✅ | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Every public adapter returns all requested layers from a single forward pass.
¹ `evo2-arc` sweeps the taps named in its spec (17 for `specs/external/evo2-arc-7b-resid.yaml`).

**Reading the numbers.** With 128 classes and one test gene per class, random guessing gives
a macro-F1 of about 1/128 ≈ 0.008, and a model-free k-mer counter run through this task's own
code scores **0.0191** (k = 4). That near-zero floor is what makes EC the cleanest test of
learned representation in the suite. The same one-gene-per-class test set also makes the score
coarse: one flipped prediction moves macro-F1 by roughly 0.01, so small gaps, and best-layer
values picked over many layers, deserve caution ([CAVEATS.md](CAVEATS.md)). The paper's values
for every model and layer are in [`paper/reference_scores.csv`](../paper/reference_scores.csv).

---

## Design notes

- **One GPU pass**: all train + test windows are embedded in a single batched adapter call.
  The layer-sweep returns every layer from that one pass.
- **Over-context is never silent**: both `window` and `truncate` count and log how many
  sequences overflowed; neither drops an instance (critical — test is 1 gene/class).
- **Metric is a true vendor**: DGEB's classification evaluator is torch-free, so
  it is copied char-for-char and *called*, with a `SHA256SUMS` pin and an equivalence test
  (wrapper == direct vendored call). Stronger than the BacBench re-derivation.
- **DGEB fidelity toggles**: no `StandardScaler` and a single embedding layer by default
  (DGEB's protocol); the layer sweep scores every layer the model exposes.
