# Reproducing the LOAM paper's benchmark results

This page says what "reproduce" means here, how to do it for each of the paper's 13 models, what
it costs, and how closely your numbers should agree with the paper's.

## What the paper reports, and what this repository reproduces

| In the paper | What it shows | Reproducible from this repository |
|---|---|---|
| Benchmark figure (score vs. parameters) | 13 models × 3 tasks; best layer for the two probe tasks | Yes: every value is in `paper/records/` |
| Layer-sweep figure | per-layer probe scores, the last-layer vs. best-layer difference, the k-mer floor | Yes: every tap is in the sweep rows; the floor is `paper/kmer_floor.json` |
| Benchmark task table | task sizes, licences, source revisions, floors | Yes: from the task code, data manifests and `paper/kmer_floor.json` |
| Evaluation-protocol details in the Methods | probe settings, context limits, comparator roster, over-context counts | Yes: from the task code, the specs and the records' `metadata` |
| RNAGym per-assay breakdown | per-assay Spearman, LOAM vs. Evo 2 | Per-assay Spearman: yes. Wild-type surprisal and the counts of homologs in the training corpus: no (the latter needs the training corpus) |
| Model cards' benchmark results | e.g. "LOAM-624M reaches 0.3939 macro-F1" | Yes: the same numbers as the benchmark figure |

The five task rows are the benchmark `loam_paper`: `bacbench-essentiality`,
`bacbench-essentiality-layer-sweep`, `dgeb-ec-classification-dna`,
`dgeb-ec-classification-dna-layer-sweep` and `rnagym-dms`.

## Steps

1. **Install** the main environment, and the Evo 2 environment if you want the Evo2-7B row
   ([INSTALL.md](INSTALL.md)). For bitwise agreement use the pinned versions,
   `envs/constraints-paper.txt`.
2. **Fetch the task data** (about 230 MB; `scripts/reproduce_paper.sh` does this for you):

   ```bash
   python scripts/fetch_bacbench_essentiality.py
   python scripts/fetch_dgeb_ec_dna.py
   python scripts/fetch_rnagym_prok.py
   ```

   Each script downloads from the upstream source at a pinned revision, rebuilds the inputs the
   task uses, and checks every file against the sha256 in the task's manifest. A mismatch is
   an error.
3. **Run a model and compare it with the paper:**

   ```bash
   scripts/reproduce_paper.sh --model LOAM-624M            # one model
   scripts/reproduce_paper.sh --model "Evo2-7B (residual stream)"   # needs GLMBENCH_EVO2_PYTHON
   scripts/reproduce_paper.sh --all                         # all 13
   scripts/reproduce_paper.sh --list                        # model ids and specs
   ```

   This is `glmbench run --model <spec> --benchmark loam_paper --results results/` followed by
   `python scripts/compare_to_paper.py results/loam_paper/glmb_<hash>.json`. For the LOAM models,
   [`notebooks/loam_hf_benchmark.ipynb`](../notebooks/loam_hf_benchmark.ipynb) does the same in a
   few cells.

A run writes its record after every task and resumes where it stopped: a completed task of the
same model hash is skipped.

## How closely your numbers should agree

**LOAM: same weights, a different adapter, the same numbers.** The paper scored the LOAM
checkpoints with the training codebase's own model implementation. This repository scores the
released weights, the same checkpoints exported to the Hugging Face format (their sha256 is
checked against the export record), through the `loam-hf` adapter. The adapter reproduces the
native arithmetic, not just its semantics ([USING_LOAM.md](USING_LOAM.md)). In the reference
environment the result is **bitwise identical** to the paper's record for all four models, on every
metric of all five rows: headline values, every tap of both sweeps, every RNAGym assay and the
count metrics (83, 95, 111 and 127 metrics for LOAM-25M, -100M, -340M and -624M; 416 of 416).
The same holds in a clean environment built only from `envs/constraints-paper.txt`: LOAM-25M run
through the notebook there reproduces its record bitwise too. Those runs are in [`paper/public_route/`](../paper/public_route), and
`tests/unit/test_paper_reference.py` checks them with `compare_to_paper.py --exact`. The model
hashes differ from the paper's by construction (a different adapter, and a digest of the weight
file rather than of a training checkpoint); `paper/models.csv` maps each to its model id.

**The 9 other models: the same hash, the same weights, the same adapter.** Each spec in
`specs/external/` computes the model hash of its paper record (`tests/unit/test_spec_hashes.py`):
the same weights revision, remote-code revision, serving dtype and adapter version.

**The reference environment** is the one pinned in `envs/constraints-paper.txt`: Python 3.11,
the CUDA 12.6 build of torch 2.12.0, transformers 5.12.1 and the exact core packages.

**On other GPUs or library versions**, bf16 kernels and BLAS builds can move the last digits.
`scripts/compare_to_paper.py` then applies these tolerances to the headline metric, every sweep
tap and every RNAGym assay:

| Task row | Metric | Tolerance | Why |
|---|---|---|---|
| `rnagym-dms` | Spearman (macro and per assay) | \|Δ\| ≤ 0.001 | Rank statistics barely move under bf16 noise. |
| essentiality, and every tap of its sweep | macro-mean AUROC | \|Δ\| ≤ 0.002 | ~27.7k test genes; the probe's solver sometimes stops at its iteration cap, which amplifies small feature changes. |
| EC, and every tap of its sweep | macro-F1 | \|Δ\| ≤ 0.016 | 128 test rows, one per class: one flipped prediction moves macro-F1 by ~0.008, so this allows two flips. |
| both sweeps | best-layer index | the same, or the paper's best tap within the tolerance above of your best | |

Known sources of non-bitwise results:

- bf16 inference: on LOAM, bf16 and fp32 differ by a mean of 0.0105 and a maximum of 0.2345
  nats per token;
- the essentiality probe (scikit-learn LogisticRegression, lbfgs) does not always converge within
  its 1,000 iterations, so its last digits depend on the numpy/scipy/scikit-learn builds and
  their BLAS;
- Evo 1.5's outputs are not batch-invariant (no attention mask, and its long convolutions
  see the padding), so its numbers depend on the spec's `batch_size` and `token_budget`;
- the GPU kernels chosen at run time, and BLAS threading.

Two rules for reading the numbers, which the paper follows:

- **The best layer is selected on the test split**, so a best-layer score is a retrospective
  upper bound. Report it next to the last-layer score.
- **Ranks are within a task and within a readout.** Never average AUROC, macro-F1 and Spearman.

More on reading the numbers: [CAVEATS.md](CAVEATS.md).

## Compute and memory

Run each model on one CUDA GPU. We recommend 48 GB of GPU memory for the two 7B models with the
specs' batch settings; everything else needs much less. The essentiality embedding pass
(164,630 windows) dominates the GPU time and grows with model size, so the two 7B models take
by far the longest; EC and RNAGym are small, and the sweeps reuse the same pass. Add CPU time
for the probe fits: one fit per tap, so deep models take longest.

**Host RAM.** The essentiality layer sweep keeps the pooled embeddings of every window at every
tap in memory, in float64, and then builds a per-gene matrix of the same size. Plan for
about three times *windows × taps × hidden size × 8 bytes* (164,630 windows): roughly 35 GB for LOAM-100M, 80 GB for LOAM-340M, 120 GB for LOAM-624M and over 500 GB for a 7B model with 33 taps (table in [INSTALL.md](INSTALL.md)). The 7B rows therefore need a
large-memory machine. Do not run
another memory-heavy job next to a 7B sweep: under memory pressure the sweep stalls without
logging anything.

## What is not in this repository

- The native LOAM implementation the paper's LOAM rows were produced with (it belongs to the
  training codebase). The released weights and the `loam-hf` adapter reproduce its numbers
  bitwise, as above.
- The wild-type surprisal and the training-corpus homology counts of the RNAGym breakdown.
- The paper's scaling, depth-health, long-range attention and interpretability analyses, which
  are not benchmark results.
