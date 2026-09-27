# BacBench gene-essentiality (DNA) — vendored data

Backing data for the `bacbench-essentiality` task. One gene = one `(DNA sequence, essential
0/1)` instance, tagged with `genome_name` + `genus`; the probe is fit on the **train**
genomes and evaluated **per test genome** (macro AUROC/AUPRC), exactly as in BacBench.

## Layout

- `manifest.yaml` — per-split file + **sha256** + counts (the integrity gate). Committed.
- `essential_genes_{train,validation,test}.csv` — `genome_name,genus,essential,sequence`,
  one row per gene. **Gitignored** (~200 MB); fetched on demand.
- `NOTICE` — provenance + license (Apache-2.0).

## Populate (one-time, needs HuggingFace + `pip install datasets`)

```bash
python scripts/fetch_bacbench_essentiality.py          # all splits → CSVs + real sha256
```

The fetch reads HuggingFace `macwiatrak/bacbench-essential-genes-dna` at the pinned revision
`0210c2445e89c765c0fe5d69618b8a6c262e5d30`, extracts each gene's model input — **CDS + 128 bp
upstream promoter**, strand-aware, no reverse complement — via the byte-identical
`bacbench_essentiality_data.get_dna_seq` recipe, then re-emits this manifest with hashes
(byte for byte the committed one).

CI never needs the download: it runs against the committed tiny fixture under
`tests/data/bacbench_essentiality_tiny/`.
