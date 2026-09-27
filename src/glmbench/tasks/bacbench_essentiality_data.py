"""BacBench gene-essentiality data layer — vendored genomes with sha256 integrity.

The data half of the BacBench gene-essentiality task: load per-split, per-gene records
(``genome_name``, ``genus``, ``essential``, ``sequence``) from disk, enforce the record
contract, normalize each DNA sequence once (``strip / upper / U→T``, so every model sees
identical DNA), and verify every vendored file against the committed
sha256 manifest so silent data drift fails loudly (mirrors :mod:`rnagym_data`).

The per-gene **sequence** is pre-extracted at fetch time by :func:`get_dna_seq` — the
BacBench DNA-LM recipe (CDS + 128 bp upstream promoter, strand-aware, *no* reverse
complement), recycled verbatim from upstream ``bacbench/modeling/embed_dna.py::get_dna_seq``
(commit 64b7d458). It lives here (not just in the fetch script) so the extraction is unit
testable against the upstream semantics.

This module is **core (torch-free)**: pure stdlib + the dependency-light core deps.
Scoring (which needs a model) lives in each adapter's runner; the probe is CPU sklearn in
the task module, never here.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _raise_csv_field_limit() -> None:
    """Lift Python's default 128 KB csv field limit (process-global).

    A few source annotations span a whole replicon (multi-Mbp "genes"), and even real
    bacterial genes reach tens of kb — both exceed the 131072-byte default and raise
    ``_csv.Error: field larger than field limit``. BacBench reads the same rows via pandas
    (no such limit) and windows them, so we lift the limit rather than drop them. Set once
    at import (covers the loader **and** any sibling script that imports this module, e.g.
    ``scripts/make_essentiality_subset.py``). ``sys.maxsize`` can overflow C ``long`` on
    some platforms, so step down until accepted.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()

# Vendored-data root: manifest + per-split processed CSVs (gitignored heavy cache).
DEFAULT_BACBENCH_DIR = Path(__file__).resolve().parent / "data" / "bacbench_essentiality"
DEFAULT_MANIFEST = DEFAULT_BACBENCH_DIR / "manifest.yaml"

# The promoter length BacBench prepends to each gene for DNA LMs (Supp. Table 5).
DEFAULT_PROMOTER_LEN = 128

# Required columns of every processed per-split CSV.
_REQUIRED_COLUMNS = ("genome_name", "genus", "essential", "sequence")


class EssentialityContractError(ValueError):
    """Raised when a vendored CSV violates the {genome_name, genus, essential, sequence} contract."""


class ManifestHashError(RuntimeError):
    """Raised when a vendored file's sha256 diverges from the committed manifest."""


@dataclass
class Genome:
    """One genome's genes — sequences (CDS+promoter, normalized) + binary labels."""

    genome_name: str
    genus: str
    split: str
    sequences: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_essential(self) -> int:
        return int(sum(self.labels))

    @property
    def has_both_classes(self) -> bool:
        """True iff this genome contains ≥1 essential and ≥1 non-essential gene.

        Mirrors upstream ``_has_both_binary_classes`` — a genome lacking both classes has
        an undefined AUROC/AUPRC and is excluded from the macro (counted in metadata).
        """
        s = set(self.labels)
        return 0 in s and 1 in s


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_sequence(seq: str) -> str:
    """RNA→DNA normalization applied once, in the loader.

    ``strip / upper / U→T`` so every adapter receives identical DNA. Adapters may re-map
    to their own alphabet but must not change which nucleotides are present.
    """
    return seq.strip().upper().replace("U", "T")


def get_dna_seq(
    dna_seq: str,
    start: int,
    end: int,
    strand: float | str | None,
    promoter_len: int = DEFAULT_PROMOTER_LEN,
) -> str:
    """Extract one gene's DNA (CDS + upstream promoter) — the BacBench DNA-LM recipe.

    Recycled VERBATIM in behavior from upstream
    ``bacbench/modeling/embed_dna.py::get_dna_seq`` (commit 64b7d458). ``start``/``end`` are
    **1-based, end-inclusive** GenBank coordinates into ``dna_seq`` (the full contig). A
    ``promoter_len`` window is added on the gene's upstream side (strand-aware); when
    ``strand`` is unknown it is added on *both* sides. **No reverse complement is applied**
    — the model sees the forward-strand substring, exactly as upstream does.
    """
    seq_len = len(dna_seq)
    if strand in {"+", "-"}:
        strand = 1.0 if strand == "+" else -1.0

    if strand is None:  # promoter on both sides
        seq_start = max(1, start - promoter_len)
        seq_end = min(seq_len, end + promoter_len)
    elif float(strand) > 0:  # + strand: promoter upstream of start
        seq_start = max(1, start - promoter_len)
        seq_end = min(seq_len, end)
    else:  # - strand: promoter upstream is at the higher coordinate
        seq_start = max(1, start)
        seq_end = min(seq_len, end + promoter_len)

    # Python slice: 0-based, end-exclusive (upstream uses [seq_start-1 : seq_end]).
    return dna_seq[seq_start - 1 : seq_end].upper()


def load_split_csv(path: Path | str, *, split: str) -> list[Genome]:
    """Load one per-split CSV into ``Genome`` objects, enforcing the record contract.

    Rows are grouped by ``genome_name`` (input order preserved). Raises
    :class:`EssentialityContractError` on a missing column, a bad label, an empty
    sequence, or a genome whose rows disagree on ``genus``.
    """
    path = Path(path)
    genomes: dict[str, Genome] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise EssentialityContractError(
                f"{path.name}: missing required column(s) {missing}; has {fieldnames}, "
                f"needs {list(_REQUIRED_COLUMNS)}"
            )
        for lineno, row in enumerate(reader, start=2):
            gname = (row.get("genome_name") or "").strip()
            genus = (row.get("genus") or "").strip()
            raw_label = (row.get("essential") or "").strip()
            seq = preprocess_sequence(row.get("sequence") or "")
            if gname == "":
                raise EssentialityContractError(f"{path.name}:{lineno}: empty genome_name")
            if raw_label not in ("0", "1"):
                raise EssentialityContractError(
                    f"{path.name}:{lineno}: essential must be 0 or 1, got {raw_label!r}"
                )
            if seq == "":
                raise EssentialityContractError(
                    f"{path.name}:{lineno}: empty sequence for genome {gname!r}"
                )
            g = genomes.get(gname)
            if g is None:
                g = Genome(genome_name=gname, genus=genus, split=split)
                genomes[gname] = g
            elif g.genus != genus:
                raise EssentialityContractError(
                    f"{path.name}:{lineno}: genome {gname!r} has conflicting genus "
                    f"{genus!r} vs {g.genus!r}"
                )
            g.sequences.append(seq)
            g.labels.append(int(raw_label))
    if not genomes:
        raise EssentialityContractError(f"{path.name}: no gene rows after parsing")
    return list(genomes.values())


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Parse the vendored-data manifest (per-split file + sha256 + counts)."""
    with open(path) as f:
        manifest: dict[str, Any] = yaml.safe_load(f)
    return manifest


def verify_manifest(
    manifest: dict[str, Any], data_dir: Path | str, *, splits: tuple[str, ...] | None = None
) -> None:
    """Assert every vendored split file matches its recorded sha256. Loud on drift.

    Raises :class:`ManifestHashError` if a file is missing or its digest diverges — the
    integrity gate that keeps a real run from silently scoring tampered/re-fetched data.
    Only the requested ``splits`` are checked (``None`` → all listed).
    """
    data_dir = Path(data_dir)
    for entry in manifest.get("splits", []):
        if splits is not None and entry["name"] not in splits:
            continue
        rel = entry["file"]
        expected = entry.get("sha256")
        fpath = data_dir / rel
        if not fpath.exists():
            raise ManifestHashError(
                f"vendored split file missing: {fpath} "
                "(run scripts/fetch_bacbench_essentiality.py to populate)"
            )
        if expected is None:
            continue  # synthetic fixtures omit sha256 (loaded with verify_hashes=False)
        actual = sha256_file(fpath)
        if actual != expected:
            raise ManifestHashError(
                f"sha256 drift for {rel}: manifest={expected} actual={actual}. "
                "The vendored data was modified or re-fetched from a different revision."
            )


def load_genomes(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    data_dir: Path | str | None = None,
    verify_hashes: bool = True,
    splits: tuple[str, ...] | None = None,
) -> list[Genome]:
    """Load every genome declared in *manifest_path* (optionally verifying integrity first).

    The one-call entry point used by the task's ``prepare()``. ``data_dir`` defaults to the
    manifest's own directory. ``splits`` restricts which splits to load (e.g.
    ``("train", "test")`` — the probe never needs ``validation``).
    """
    manifest_path = Path(manifest_path)
    base = Path(data_dir) if data_dir is not None else manifest_path.parent
    manifest = load_manifest(manifest_path)
    if verify_hashes:
        verify_manifest(manifest, base, splits=splits)
    genomes: list[Genome] = []
    seen_split = False
    for entry in manifest.get("splits", []):
        name = entry["name"]
        if splits is not None and name not in splits:
            continue
        seen_split = True
        genomes.extend(load_split_csv(base / entry["file"], split=name))
    if not seen_split:
        raise EssentialityContractError(
            f"manifest {manifest_path} declares no matching splits (asked for {splits})"
        )
    return genomes


def main(argv: list[str] | None = None) -> int:
    """Tiny CLI: print per-split genome/gene/label counts for a manifest (debug aid)."""
    import argparse

    ap = argparse.ArgumentParser(description="Summarize a BacBench essentiality manifest.")
    ap.add_argument("manifest", nargs="?", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    genomes = load_genomes(args.manifest, verify_hashes=not args.no_verify)
    by_split: dict[str, list[Genome]] = {}
    for g in genomes:
        by_split.setdefault(g.split, []).append(g)
    for split, gs in by_split.items():
        n_genes = sum(len(g) for g in gs)
        n_ess = sum(g.n_essential for g in gs)
        print(
            f"{split:>11}: {len(gs):>3} genomes, {n_genes:>7} genes, "
            f"{n_ess:>6} essential ({100 * n_ess / max(1, n_genes):.1f}%)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
