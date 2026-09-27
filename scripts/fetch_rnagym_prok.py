#!/usr/bin/env python3
"""One-time fetcher for the RNAGym prokaryotic protein-coding DMS subset.

Data lands under ``src/glmbench/tasks/data/rnagym/`` (or ``--data-dir``). Each assay's
``rna_type`` is recorded in the manifest so the **type-mean** macro can aggregate across RNA
types without re-reading the reference sheet at score time.

Run this **once** on a box where the RNAGym hosts are reachable. It:

1. Downloads ``reference_sheet_final.csv`` from the RNAGym GitHub repo at
   ``GITHUB_COMMIT``, vendoring it.
2. Selects the subset **programmatically** — ``RNA_TYPE == "mRNA-coding"`` and
   ``ASSAY_DESCRIPTION == "PROK"`` — never a hard-coded list. These are the 11 assays of
   protein-coding genes from prokaryotes, every such assay the reference sheet lists.
3. Downloads each assay's processed file into ``processed_DMS_files/{DMS_ID}.csv`` with
   the ``{mutant, sequence, DMS_score}`` column contract.
4. Writes ``prok_coding_manifest.yaml`` pinning each file by sha256 (+ n_rows, cds_len,
   rna_type).

Download backends, chosen by ``--source``:
  - ``--source hf``       : per-assay parquet from the HuggingFace mirror
                            ``Marks-lab/RNAgym`` at ``HF_COMMIT`` (default). This is the
                            backend the committed manifest was produced with, and a fetch
                            re-emits that manifest byte for byte.
  - ``--source official`` : ``fitness_processed_assays.zip`` from marks.hms.harvard.edu.
                            It yields the same bytes today, but the URL carries no
                            revision, so it is not pinned.
  - ``--source local``    : re-emit the manifest from CSVs already present on disk
                            (no network).

The committed manifest is what makes the integrity gate real: a unit test diffs every
vendored file's sha256 against it and fails loudly on drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "glmbench" / "tasks" / "data" / "rnagym"
PROCESSED = "processed_DMS_files"

# The upstream commits the committed manifest was produced from.
GITHUB_COMMIT = "e7c064617c8bf6a3c2ef2ddb6238c1abde340602"  # MarksLab-DasLab/RNAGym
HF_COMMIT = "bc6ccf4c5bf445ca25991f2a74448ae7e153b9f6"  # Marks-lab/RNAgym (HuggingFace)

REFERENCE_SHEET_URL = (
    f"https://raw.githubusercontent.com/MarksLab-DasLab/RNAGym/{GITHUB_COMMIT}/"
    "fitness/reference_sheet_final.csv"
)
OFFICIAL_ZIP_URL = (
    "https://marks.hms.harvard.edu/rnagym/fitness_prediction/"
    "fitness_processed_assays.zip"
)
HF_ASSAY_URL = (
    f"https://huggingface.co/datasets/Marks-lab/RNAgym/resolve/{HF_COMMIT}/"
    "fitness_prediction/assays/{dms_id}.parquet"
)

SELECT_RNA_TYPE = "mRNA-coding"
SELECT_ASSAY = "PROK"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_reference_rows(data_dir: Path) -> list[dict[str, str]]:
    """Return the committed reference sheet's rows (utf-8-sig tolerant)."""
    raw = (data_dir / "reference_sheet_final.csv").read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def fetch_reference_sheet(data_dir: Path) -> list[dict[str, str]]:
    print(f"[fetch] reference sheet: {REFERENCE_SHEET_URL}")
    raw = urllib.request.urlopen(REFERENCE_SHEET_URL, timeout=60).read()  # noqa: S310
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "reference_sheet_final.csv").write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    print(f"[fetch] reference sheet: {len(rows)} assays")
    return rows


def select_subset(rows: list[dict[str, str]]) -> list[str]:
    sel = [
        r["DMS_ID"]
        for r in rows
        if r["RNA_TYPE"] == SELECT_RNA_TYPE and r["ASSAY_DESCRIPTION"] == SELECT_ASSAY
    ]
    print(f"[select] {SELECT_RNA_TYPE} ∩ {SELECT_ASSAY}: {len(sel)} assays")
    for d in sel:
        print(f"         {d}")
    return sel


def _rna_type_map(rows: list[dict[str, str]]) -> dict[str, str]:
    return {r["DMS_ID"]: r["RNA_TYPE"] for r in rows}


def _normalize_csv_text(text: str) -> tuple[str, int, int]:
    """Re-emit a CSV with the canonical header ``{mutant, sequence, DMS_score}``.

    Returns ``(csv_text, n_rows, cds_len)``. Accepts ``DMS_score`` or ``dms_score``.
    """
    reader = csv.DictReader(io.StringIO(text))
    fns = reader.fieldnames or []
    score_key = "DMS_score" if "DMS_score" in fns else "dms_score"
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["mutant", "sequence", "DMS_score"])
    n = 0
    cds_len = 0
    for row in reader:
        seq = row.get("sequence", "")
        cds_len = max(cds_len, len(seq))
        writer.writerow([row.get("mutant", ""), seq, row.get(score_key, "")])
        n += 1
    return out.getvalue(), n, cds_len


def fetch_from_hf(dms_id: str) -> str:
    import pyarrow.parquet as pq  # lazy — only needed for the HF backend

    url = HF_ASSAY_URL.format(dms_id=dms_id)
    raw = urllib.request.urlopen(url, timeout=120).read()  # noqa: S310
    table = pq.read_table(io.BytesIO(raw))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=table.column_names)
    writer.writeheader()
    for row in table.to_pylist():
        writer.writerow(row)
    return out.getvalue()


def fetch_from_official_zip(dms_ids: list[str]) -> dict[str, str]:
    print(f"[fetch] official zip: {OFFICIAL_ZIP_URL}")
    raw = urllib.request.urlopen(OFFICIAL_ZIP_URL, timeout=600).read()  # noqa: S310
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".csv")}
    out: dict[str, str] = {}
    for dms_id in dms_ids:
        if dms_id not in names:
            raise KeyError(f"{dms_id} not found in official zip")
        out[dms_id] = zf.read(names[dms_id]).decode("utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["hf", "official", "local"], default="hf")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="where to write the data + manifest (default: the task's data directory).",
    )
    args = ap.parse_args()
    data_dir: Path = args.data_dir
    processed_dir = data_dir / PROCESSED

    if args.source == "local":
        # Re-emit the manifest from CSVs already on disk; reference sheet stays as-is.
        rows = _read_reference_rows(data_dir)
        dms_ids = select_subset(rows)
        raw_csv = {
            d: (processed_dir / f"{d}.csv").read_text() for d in dms_ids
        }
    else:
        rows = fetch_reference_sheet(data_dir)
        dms_ids = select_subset(rows)
        processed_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = {}
        if args.source == "official":
            raw_csv = fetch_from_official_zip(dms_ids)
        else:
            for dms_id in dms_ids:
                print(f"[fetch] HF parquet: {dms_id}")
                raw_csv[dms_id] = fetch_from_hf(dms_id)

    rna_types = _rna_type_map(rows)

    manifest_lines = [
        "# RNAGym prokaryotic protein-coding DMS — vendored manifest (gLMBench data layer).",
        "# Generated by scripts/fetch_rnagym_prok.py. DO NOT EDIT BY HAND.",
        "# The metrics/integrity gate diffs each file's sha256 against this.",
        'source: "RNAGym (MarksLab-DasLab), MIT License"',
        'citation: "Notin et al., RNAGym, bioRxiv 2025.06.16.660049"',
        "reference_sheet: reference_sheet_final.csv",
        "selection:",
        f'  rna_type: "{SELECT_RNA_TYPE}"',
        f'  assay_description: "{SELECT_ASSAY}"',
        "processed_dir: processed_DMS_files",
        "assays:",
    ]
    for dms_id in dms_ids:
        text, n_rows, cds_len = _normalize_csv_text(raw_csv[dms_id])
        fpath = processed_dir / f"{dms_id}.csv"
        if args.source != "local":
            fpath.write_text(text)
        digest = _sha256_bytes(fpath.read_bytes())
        manifest_lines += [
            f"  - dms_id: {dms_id}",
            f"    file: processed_DMS_files/{dms_id}.csv",
            f'    rna_type: "{rna_types[dms_id]}"',
            f"    sha256: {digest}",
            f"    n_rows: {n_rows}",
            f"    cds_len: {cds_len}",
        ]
        print(f"[write] {fpath.name}: {n_rows} rows, cds_len={cds_len}, sha256={digest[:12]}…")

    (data_dir / "prok_coding_manifest.yaml").write_text("\n".join(manifest_lines) + "\n")
    print(f"[done] manifest → {data_dir / 'prok_coding_manifest.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
