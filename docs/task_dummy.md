# Tasks: Dummy (smoke tests)

> _Infrastructure smoke tests on five hard-coded toy DNA sequences: exercise the full run lifecycle, leaderboard, and capability negotiation end-to-end without real data or a GPU._

Two lightweight tasks used to verify the core benchmark infrastructure without
requiring real data or GPU resources.

| Registry key | Required capability | Purpose |
|---|---|---|
| `dummy` | `SEQUENCE_LOGLIKELIHOOD` | Validates the run lifecycle and leaderboard end-to-end. |
| `dummy-embed` | `EMBEDDING` | Validates capability negotiation (score-only adapters yield a clean N/A). |

Benchmark: `dummy_bench`

---

## Source

No external dataset. Five hard-coded toy DNA sequences are embedded directly in the
task module.

---

## Code and files

| File | Role |
|---|---|
| `src/glmbench/tasks/dummy.py` | Both task classes: `DummyScoreTask` and `DummyEmbedTask`. |
| `src/glmbench/benchmarks/dummy_bench.yaml` | Benchmark definition that includes both tasks. |

---

## `dummy` — score task

### Data

Five toy sequences hard-coded in the module:

```
ACGTACGT
TTTTAAAA
GGGGCCCC
ACACACAC
TGCATGCA
```

### Lifecycle

**`prepare()`**: Stores the five sequences in `self._seqs`.

**`run(adapter)`**: One batched call:

```python
self._scores = adapter.score_sequences(self._seqs, reduction="mean")
```

Returns a list of five per-sequence mean log-likelihoods.

**`score()`**: Computes mean, min, and max over the five scores. Validates that
the result is non-empty and contains no NaN values.

### Output metrics

| Key | Description |
|---|---|
| `mean_score` | **Primary.** Mean log-likelihood across the five sequences. |
| `min_score` | Minimum across the five. |
| `max_score` | Maximum across the five. |

Metadata: `n_scored=5`, `n_missing=0`, `task_version="0.1.0"`.

---

## `dummy-embed` — embedding task

### Data

First two sequences from the dummy set (`ACGTACGT`, `TTTTAAAA`).

### Lifecycle

**`prepare()`**: Stores the two sequences.

**`run(adapter)`**:

```python
result = adapter.embed(self._seqs, layers="last", pool="mean")
```

Returns an `EmbeddingResult` with `embeddings` shape `[2, 1, H]` (2 sequences,
1 layer, H dimensions). Computes the L2 norm of each sequence's embedding:

$$n_i = \lVert \mathbf{h}_i \rVert_2$$

**`score()`**: Mean of the two norms. Validates no NaN.

### Output metrics

| Key | Description |
|---|---|
| `mean_embed_norm` | **Primary.** Mean L2 norm across the two embedded sequences. |

Metadata: `n_scored=2`, `n_missing=0`.

---

## Why these tasks exist

- **`dummy`** provides a deterministic smoke test that exercises the entire stack
  (adapter wire protocol, subprocess runner, response parsing, TaskResult,
  leaderboard JSON) with no external data and no GPU requirement.
- **`dummy-embed`** proves that capability negotiation works correctly: any adapter
  that does not declare `EMBEDDING` (e.g., a score-only adapter) returns a
  first-class `N/A` result with the missing capability named, rather than a crash
  or a silent zero.

Running `dummy_bench` with the `EchoAdapter` (the model-free test double) is the
fastest possible CI check of the full pipeline.

---

## How to run

```bash
# Run both dummy tasks with the fully capable EchoAdapter (no model, no GPU)
glmbench run \
  --model specs/echo-full.yaml \
  --benchmark dummy_bench \
  --results results/

# Add a score-only Echo model: its `dummy-embed` cell is N/A
glmbench run \
  --model specs/echo-score-only.yaml \
  --benchmark dummy_bench \
  --results results/

# Build the leaderboard (two rows, one explicit N/A cell)
glmbench leaderboard \
  --benchmark dummy_bench \
  --results results/ \
  --out results/dummy_bench/LEADERBOARD.md
```

The `EchoAdapter` returns constant values derived from the input sequences (not
random), so results are deterministic and reproducible.
