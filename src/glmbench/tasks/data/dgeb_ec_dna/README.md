# DGEB EC classification (DNA) — vendored data

Task: `dgeb-ec-classification-dna` (and `…-layer-sweep`). A 128-way single-label
classification of protein-coding genes into Enzyme Commission (EC) classes, probed from
frozen genomic-LM **embeddings** (mean-pooled), scored by macro-F1 via the vendored DGEB
`logRegClassificationEvaluator` (`_vendor/dgeb/`).

- Source: HuggingFace `tattabio/ec_classification_dna`, revision
  `cd61c74b4930cf9f1963e6d73ff7f14e2c8e74dd` (DGEB, Apache-2.0).
- Splits: `train` (512), `test` (128); 128 EC classes.
- Files here: `manifest.yaml` (sha256 + counts), this README, `NOTICE`, `.gitignore`
  (heavy CSVs are fetched, not committed).

## Fetch

```bash
python scripts/fetch_dgeb_ec_dna.py     # downloads + writes CSVs + (re)emits manifest hashes
```

CI does not need this — it runs against the committed tiny fixture
(`tests/data/dgeb_ec_dna_tiny/`).
