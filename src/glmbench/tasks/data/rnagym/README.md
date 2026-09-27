# RNAGym prokaryotic protein-coding DMS

The RNAGym DMS task scores the **11 `mRNA-coding ∩ PROK`** assays of RNAGym: every
deep mutational scan of a protein-coding gene from a prokaryote in the RNAGym reference
sheet.

## What's committed

- `reference_sheet_final.csv` — the RNAGym assay index. The 11-assay subset is
  selected **programmatically** from it (`RNA_TYPE == "mRNA-coding"` and
  `ASSAY_DESCRIPTION == "PROK"`), never a hard-coded list.
- `prok_coding_manifest.yaml` — 11 `DMS_ID`s + `rna_type` + sha256 + row count +
  CDS length. This is the integrity contract: `verify_manifest` diffs each file's
  sha256 against it and fails loudly on drift.
- `NOTICE` — MIT attribution.

## What's NOT committed (gitignored cache)

- `processed_DMS_files/*.csv` — the 11 assay CSVs (~18 MB), fetched on demand and pinned
  by the committed manifest.

## Repopulate

```bash
python3 scripts/fetch_rnagym_prok.py            # HF mirror (default)
python3 scripts/fetch_rnagym_prok.py --source official   # marks.hms zip
python3 scripts/fetch_rnagym_prok.py --source local      # re-emit manifest from on-disk CSVs
```

The fetch is pinned: the reference sheet comes from GitHub `MarksLab-DasLab/RNAGym` at
`e7c064617c8bf6a3c2ef2ddb6238c1abde340602` and the assays from the HuggingFace mirror
`Marks-lab/RNAgym` at `bc6ccf4c5bf445ca25991f2a74448ae7e153b9f6`. A fetch at those revisions
re-emits `prok_coding_manifest.yaml` byte for byte. The official zip carries no revision; it
yields the same bytes today.

The fetch script runs where `marks.hms.harvard.edu` / `huggingface.co` are
reachable. CI exercises the **tiny synthetic fixture** under
`tests/data/rnagym_tiny/` instead, which IS committed.

## Loader contract

Columns `{mutant, sequence, DMS_score}` (the HF parquet's `dms_score` is also
accepted). WT rows (NaN/empty `mutant`) are dropped; sequences are normalized once
with `strip().upper().replace("U", "T")` so every model sees identical DNA inputs.
