#!/usr/bin/env python3
"""Build a small genome subset of the BacBench essentiality data — for a TIMING/smoke run.

The full task embeds ~152k genes (train+test) in ONE batched call, which on a 7B model is
many GPU-minutes-to-hours. To estimate that cost cheaply, carve out a few genomes, run the
task on the subset, read ``embed_seconds`` / ``n_windows_embedded`` from the result, and
extrapolate by the gene-count ratio.

Usage (after ``scripts/fetch_bacbench_essentiality.py`` has populated the full CSVs):

    python scripts/make_essentiality_subset.py --n-train 2 --n-test 1 --out-dir /tmp/bb_sub
    export GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST=/tmp/bb_sub/manifest.yaml
    glmbench run --model specs/external/evo2-arc-7b-resid.yaml --benchmark bacbench \
        --results /tmp/bb_sub_results --keep-scratch

Then divide: ``full_seconds ≈ embed_seconds * (full_train+test_genes / subset_genes)``.
The script prints the full-corpus gene counts and the exact extrapolation formula.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from glmbench.tasks.bacbench_essentiality_data import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_manifest,
    sha256_file,
)


def _take_genomes(src_csv: Path, dst_csv: Path, n_genomes: int) -> tuple[int, int]:
    """Copy the first ``n_genomes`` genomes' rows from ``src_csv`` to ``dst_csv``.

    Returns ``(n_genomes_written, n_genes_written)``. Genome order follows first-appearance.
    """
    chosen: list[str] = []
    chosen_set: set[str] = set()
    rows_out: list[list[str]] = []
    header: list[str] = []
    with open(src_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        gcol = header.index("genome_name")
        for row in reader:
            g = row[gcol]
            if g not in chosen_set:
                if len(chosen) >= n_genomes:
                    continue
                chosen.append(g)
                chosen_set.add(g)
            if g in chosen_set:
                rows_out.append(row)
    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    return len(chosen), len(rows_out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="full-corpus manifest")
    ap.add_argument("--n-train", type=int, default=2, help="train genomes to keep (≥2 classes)")
    ap.add_argument("--n-test", type=int, default=1, help="test genomes to keep")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    src_dir = Path(args.manifest).parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    keep = {"train": args.n_train, "test": args.n_test}
    full_counts: dict[str, int] = {}
    lines = ["# Timing subset of BacBench essentiality — generated, do not commit.",
             "promoter_len: 128", "splits:"]
    sub_genes = 0
    for entry in manifest.get("splits", []):
        name = entry["name"]
        src_csv = src_dir / entry["file"]
        full_counts[name] = int(entry.get("n_genes", 0))
        if name not in keep:
            continue
        if not src_csv.exists():
            print(f"[error] {src_csv} missing — run scripts/fetch_bacbench_essentiality.py first.")
            return 1
        dst_csv = out_dir / entry["file"]
        ng, ngenes = _take_genomes(src_csv, dst_csv, keep[name])
        sub_genes += ngenes
        lines += [
            f"  - name: {name}",
            f"    file: {entry['file']}",
            f"    sha256: {sha256_file(dst_csv)}",
            f"    n_genomes: {ng}",
            f"    n_genes: {ngenes}",
        ]
        print(f"[subset] {name}: {ng} genomes, {ngenes} genes → {dst_csv.name}")

    (out_dir / "manifest.yaml").write_text("\n".join(lines) + "\n")

    full_embed = full_counts.get("train", 0) + full_counts.get("test", 0)
    print(f"[done] subset manifest → {out_dir / 'manifest.yaml'}")
    print(f"[scale] subset embeds {sub_genes} genes; full train+test ≈ {full_embed} genes.")
    if sub_genes:
        print(
            "[extrapolate] full_embed_seconds ≈ embed_seconds × "
            f"({full_embed} / {sub_genes}) = embed_seconds × {full_embed / sub_genes:.1f}"
        )
    print(
        "\nNext:\n"
        f"  export {('GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST')}={out_dir / 'manifest.yaml'}\n"
        "  glmbench run --model specs/external/evo2-arc-7b-resid.yaml --benchmark bacbench "
        "--results /tmp/bb_sub_results\n"
        "  # then read metadata.embed_seconds + n_windows_embedded from the result JSON."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
