#!/usr/bin/env python3
"""One-time fetcher for the BacBench gene-essentiality DNA subset (gLMBench data layer).

Mirrors ``scripts/fetch_rnagym_prok.py``: downloads the upstream HuggingFace dataset,
extracts the per-gene model input **once** (CDS + 128 bp upstream promoter, strand-aware,
via the byte-identical BacBench recipe :func:`...bacbench_essentiality_data.get_dna_seq`),
writes one compact per-split CSV (gitignored heavy cache), and (re)emits the sha256-pinned
``manifest.yaml`` that the task's integrity gate diffs against.

Source dataset: ``macwiatrak/bacbench-essential-genes-dna`` (Apache-2.0), pinned to the
commit the committed manifest was produced from (``HF_COMMIT``). Each row is one genome with
**per-contig** nested lists: ``dna_sequence`` (List[str]), ``start`` / ``end`` / ``strand`` /
``essential`` (List[List[...]]), plus ``genome_name`` / ``genus``. Run this ONCE on a box
where HuggingFace is reachable (``pip install datasets``):

    python scripts/fetch_bacbench_essentiality.py                 # all splits
    python scripts/fetch_bacbench_essentiality.py --splits test   # one split
    python scripts/fetch_bacbench_essentiality.py --data-dir /tmp/bb   # write elsewhere

The committed manifest is what makes the integrity gate real: a unit test loads via the
manifest and the real run verifies each split file's sha256 against it. A fetch at the pinned
commit re-emits that manifest byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "glmbench" / "tasks" / "data" / "bacbench_essentiality"
SRC_DIR = REPO_ROOT / "src"

HF_DATASET = "macwiatrak/bacbench-essential-genes-dna"
HF_COMMIT = "0210c2445e89c765c0fe5d69618b8a6c262e5d30"  # 2026-05-03, the genus-disjoint splits
UPSTREAM_LICENSE = "Apache-2.0"
PROMOTER_LEN = 128
SPLITS = ("train", "validation", "test")

# Make the in-repo data layer importable for the verbatim get_dna_seq recipe + sha256.
sys.path.insert(0, str(SRC_DIR))
from glmbench.tasks.bacbench_essentiality_data import (  # noqa: E402
    get_dna_seq,
    sha256_file,
)


def _iter_genes(row: dict) -> list[tuple[str, int]]:
    """Yield ``(gene_sequence, label)`` for every gene in one genome row.

    Handles the dataset's per-contig nesting: ``dna_sequence`` is a list of contig
    strings; ``start`` / ``end`` / ``strand`` / ``essential`` are lists-of-lists (one inner
    list per contig). A flat (single-contig, non-nested) layout is also tolerated.
    """
    dna = row["dna_sequence"]
    starts, ends, strands, essential = row["start"], row["end"], row["strand"], row["essential"]
    contigs: list[tuple[str, list, list, list, list]]
    if starts and isinstance(starts[0], list):
        dna_list = dna if isinstance(dna, list) else dna.split()
        contigs = list(zip(dna_list, starts, ends, strands, essential, strict=True))
    else:  # flat single-contig layout
        dna_str = dna[0] if isinstance(dna, list) else dna
        contigs = [(dna_str, starts, ends, strands, essential)]

    out: list[tuple[str, int]] = []
    for contig_dna, st, en, sd, ess in contigs:
        for s, e, strand_val, label in zip(st, en, sd, ess, strict=True):
            seq = get_dna_seq(contig_dna, int(s), int(e), strand_val, promoter_len=PROMOTER_LEN)
            if seq:  # drop degenerate empty extractions (out-of-range coords)
                out.append((seq, int(label)))
    return out


def _write_split_csv(path: Path, rows: list[dict]) -> dict[str, object]:
    """Write ``genome_name,genus,essential,sequence`` for one split; return its stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_genes = 0
    n_ess = 0
    total_nt = 0
    min_len: int | None = None
    max_len = 0
    genera: set[str] = set()
    genomes: set[str] = set()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genome_name", "genus", "essential", "sequence"])
        for row in rows:
            gname = row["genome_name"]
            genus = row.get("genus") or ""
            genomes.add(gname)
            genera.add(genus)
            for seq, label in _iter_genes(row):
                w.writerow([gname, genus, label, seq])
                n_genes += 1
                n_ess += label
                seq_len = len(seq)
                total_nt += seq_len
                min_len = seq_len if min_len is None else min(min_len, seq_len)
                max_len = max(max_len, seq_len)
    return {
        "n_genomes": len(genomes),
        "n_genes": n_genes,
        "n_essential": n_ess,
        # Keep the exact numerator beside the count. ``avg_len`` is convenient
        # for inspection, while ``total_nt / n_genes`` remains the reproducible
        # authority for a weighted average across splits.
        "total_nt": total_nt,
        "min_len": min_len or 0,
        "max_len": max_len,
        "avg_len": total_nt / n_genes if n_genes else 0.0,
        "genera": sorted(g for g in genera if g),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--streaming", action="store_true", help="stream rows (low memory).")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="where to write the CSVs + manifest (default: the task's data directory).",
    )
    args = ap.parse_args(argv)
    data_dir: Path = args.data_dir

    from datasets import load_dataset  # lazy — only the fetch needs HF datasets

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "# BacBench gene-essentiality (DNA) — vendored manifest (gLMBench data layer).",
        "# Generated by scripts/fetch_bacbench_essentiality.py. DO NOT EDIT BY HAND.",
        "# The integrity gate diffs each split file's sha256 against this.",
        f'source: "{HF_DATASET} (HuggingFace), {UPSTREAM_LICENSE}"',
        f'revision: "{HF_COMMIT}"',
        'citation: "BacBench: Evaluating Genomic Language Models for Bacteria (ICLR 2026)."',
        f"promoter_len: {PROMOTER_LEN}",
        "splits:",
    ]
    for split in args.splits:
        ds = load_dataset(HF_DATASET, split=split, streaming=args.streaming, revision=HF_COMMIT)
        rows = list(ds)
        fpath = data_dir / f"essential_genes_{split}.csv"
        stats = _write_split_csv(fpath, rows)
        digest = sha256_file(fpath)
        manifest_lines += [
            f"  - name: {split}",
            f"    file: essential_genes_{split}.csv",
            f"    sha256: {digest}",
            f"    n_genomes: {stats['n_genomes']}",
            f"    n_genes: {stats['n_genes']}",
            f"    n_essential: {stats['n_essential']}",
            f"    total_nt: {stats['total_nt']}",
            f"    min_len: {stats['min_len']}",
            f"    max_len: {stats['max_len']}",
            f"    avg_len: {stats['avg_len']}",
            "    genera: [" + ", ".join(stats["genera"]) + "]",
        ]
        print(
            f"[write] {fpath.name}: {stats['n_genomes']} genomes, {stats['n_genes']} genes, "
            f"{stats['n_essential']} essential, {stats['total_nt']} nt "
            f"(avg {stats['avg_len']:.3f}), sha256={digest[:12]}…"
        )

    (data_dir / "manifest.yaml").write_text("\n".join(manifest_lines) + "\n")
    print(f"[done] manifest → {data_dir / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
