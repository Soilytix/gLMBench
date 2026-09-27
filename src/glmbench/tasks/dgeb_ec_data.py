"""DGEB EC-classification (DNA) data layer — vendored splits with sha256 integrity.

The data half of the EC classification task: load per-split records
(``entry``, ``sequence``, ``label``) from disk, enforce the record contract, normalize
each DNA sequence once (``strip / upper / U→T``, so every model sees identical DNA), and
verify every vendored file against the committed sha256 manifest so
silent data drift fails loudly (mirrors :mod:`bacbench_essentiality_data` / :mod:`rnagym_data`).

Source: HuggingFace ``tattabio/ec_classification_dna`` (DGEB), columns ``Entry`` /
``Sequence`` / ``Label`` — written to compact lowercase-header CSVs at fetch time. An
``entry`` is a UniProt accession; ``label`` is the EC class string (128 classes); the task
is single-label multiclass.

This module is **core (torch-free)**: pure stdlib + the dependency-light core deps. The
model never appears here — embedding lives in each adapter's runner, the probe is CPU
sklearn in the task module.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _raise_csv_field_limit() -> None:
    """Lift Python's default 128 KB csv field limit (process-global).

    EC DNA sequences reach ~14.7 kb (well under the 131072-byte default), but a few
    pathological rows or future re-fetches could exceed it, raising
    ``_csv.Error: field larger than field limit``. Lift it once at import to match the
    pandas-based upstream loader. ``sys.maxsize`` can overflow C ``long`` on some
    platforms, so step down until accepted.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()

# Vendored-data root: manifest + per-split CSVs (gitignored heavy cache).
DEFAULT_EC_DIR = Path(__file__).resolve().parent / "data" / "dgeb_ec_dna"
DEFAULT_MANIFEST = DEFAULT_EC_DIR / "manifest.yaml"

# Required columns of every processed per-split CSV.
_REQUIRED_COLUMNS = ("entry", "sequence", "label")


class ECClassificationContractError(ValueError):
    """Raised when a vendored CSV violates the {entry, sequence, label} contract."""


class ManifestHashError(RuntimeError):
    """Raised when a vendored file's sha256 diverges from the committed manifest."""


@dataclass
class ECSplit:
    """One split's records: parallel ``entries`` / ``sequences`` (normalized) / ``labels``."""

    name: str
    entries: list[str]
    sequences: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_classes(self) -> int:
        return len(set(self.labels))


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


def load_split_csv(path: Path | str, *, split: str) -> ECSplit:
    """Load one per-split CSV into an :class:`ECSplit`, enforcing the record contract.

    Input order is preserved. Raises :class:`ECClassificationContractError` on a missing
    column, an empty entry/label, or an empty sequence.
    """
    path = Path(path)
    entries: list[str] = []
    sequences: list[str] = []
    labels: list[str] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in _REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ECClassificationContractError(
                f"{path.name}: missing required column(s) {missing}; has {fieldnames}, "
                f"needs {list(_REQUIRED_COLUMNS)}"
            )
        for lineno, row in enumerate(reader, start=2):
            entry = (row.get("entry") or "").strip()
            label = (row.get("label") or "").strip()
            seq = preprocess_sequence(row.get("sequence") or "")
            if entry == "":
                raise ECClassificationContractError(f"{path.name}:{lineno}: empty entry")
            if label == "":
                raise ECClassificationContractError(
                    f"{path.name}:{lineno}: empty label for entry {entry!r}"
                )
            if seq == "":
                raise ECClassificationContractError(
                    f"{path.name}:{lineno}: empty sequence for entry {entry!r}"
                )
            entries.append(entry)
            sequences.append(seq)
            labels.append(label)
    if not labels:
        raise ECClassificationContractError(f"{path.name}: no rows after parsing")
    return ECSplit(name=split, entries=entries, sequences=sequences, labels=labels)


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
    Only the requested ``splits`` are checked (``None`` → all listed). Entries whose
    manifest ``sha256`` is null (synthetic fixtures) are skipped.
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
                "(run scripts/fetch_dgeb_ec_dna.py to populate)"
            )
        if expected is None:
            continue
        actual = sha256_file(fpath)
        if actual != expected:
            raise ManifestHashError(
                f"sha256 drift for {rel}: manifest={expected} actual={actual}. "
                "The vendored data was modified or re-fetched from a different revision."
            )


def load_ec_splits(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    data_dir: Path | str | None = None,
    verify_hashes: bool = True,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, ECSplit]:
    """Load the requested splits declared in *manifest_path* (optionally verifying integrity).

    The one-call entry point used by the task's ``prepare()``. ``data_dir`` defaults to the
    manifest's own directory. Returns a ``{split_name: ECSplit}`` dict; raises if a
    requested split is absent from the manifest.
    """
    manifest_path = Path(manifest_path)
    base = Path(data_dir) if data_dir is not None else manifest_path.parent
    manifest = load_manifest(manifest_path)
    if verify_hashes:
        verify_manifest(manifest, base, splits=splits)
    by_name = {e["name"]: e for e in manifest.get("splits", [])}
    out: dict[str, ECSplit] = {}
    for name in splits:
        if name not in by_name:
            raise ECClassificationContractError(
                f"manifest {manifest_path} declares no split {name!r} (has {sorted(by_name)})"
            )
        out[name] = load_split_csv(base / by_name[name]["file"], split=name)
    return out
