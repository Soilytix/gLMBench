"""RNAGym DMS data layer — vendored assays with sha256 integrity.

The data half of the RNAGym prokaryotic protein-coding task: load the 11
``mRNA-coding ∩ PROK`` assays from disk, enforce the ``{mutant, sequence, DMS_score}``
contract, drop wild-type rows, normalize each sequence once (``strip / upper / U→T``,
so every model sees identical DNA inputs), and verify every vendored file against the
committed sha256 manifest so silent data drift fails loudly.

Each assay carries its ``rna_type`` so the type-mean macro
(``glmbench.tasks.metrics.macro``) can aggregate across RNA types.

This module is **core (torch-free)**: pure stdlib + the dependency-light core deps.
Scoring (which needs a model) lives in each adapter's runner, never here.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Vendored-data root: reference sheet + manifest + processed_DMS_files/.
DEFAULT_RNAGYM_DIR = Path(__file__).resolve().parent / "data" / "rnagym"
DEFAULT_MANIFEST = DEFAULT_RNAGYM_DIR / "prok_coding_manifest.yaml"

# Column-name tolerance: RNAGym's processed CSVs use ``DMS_score``; the HF parquet
# mirror uses ``dms_score``. Accept either; everything else is the contract.
_SCORE_KEYS = ("DMS_score", "dms_score")

# Fallback RNA type when a manifest entry omits ``rna_type`` (older manifest format).
_DEFAULT_RNA_TYPE = "mRNA-coding"


class AssayContractError(ValueError):
    """Raised when a vendored CSV violates the {mutant, sequence, DMS_score} contract."""


class ManifestHashError(RuntimeError):
    """Raised when a vendored file's sha256 diverges from the committed manifest."""


@dataclass
class Assay:
    """One loaded DMS assay — WT rows dropped, sequences normalized to DNA."""

    dms_id: str
    rna_type: str
    mutants: list[str]
    sequences: list[str]
    dms_scores: list[float]

    def __len__(self) -> int:
        return len(self.mutants)


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_sequence(seq: str) -> str:
    """RNA→DNA normalization applied once, in the loader.

    ``strip / upper / U→T`` so every adapter receives identical DNA. Adapters may
    re-map to their own alphabet but must not change which nucleotides are present.
    """
    return seq.strip().upper().replace("U", "T")


def load_assay_csv(
    path: Path | str,
    *,
    dms_id: str | None = None,
    rna_type: str = _DEFAULT_RNA_TYPE,
) -> Assay:
    """Load one assay CSV, enforcing the ``{mutant, sequence, DMS_score}`` contract.

    Drops wild-type rows (NaN / empty ``mutant``) and normalizes each sequence. Raises
    :class:`AssayContractError` on a missing column or an assay with no variant rows.
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "mutant" not in fieldnames or "sequence" not in fieldnames:
            raise AssayContractError(
                f"{path.name}: missing required column(s); has {fieldnames}, "
                "needs {'mutant', 'sequence', 'DMS_score'}"
            )
        score_key = next((k for k in _SCORE_KEYS if k in fieldnames), None)
        if score_key is None:
            raise AssayContractError(
                f"{path.name}: missing a DMS score column (one of {_SCORE_KEYS}); "
                f"has {fieldnames}"
            )
        mutants: list[str] = []
        sequences: list[str] = []
        scores: list[float] = []
        for row in reader:
            mut = (row.get("mutant") or "").strip()
            # Wild-type rows carry NaN/empty mutant — dropped before scoring.
            if mut == "" or mut.lower() == "nan":
                continue
            raw_score = (row.get(score_key) or "").strip()
            try:
                score = float(raw_score)
            except ValueError:
                continue
            if math.isnan(score):
                continue
            mutants.append(mut)
            sequences.append(preprocess_sequence(row["sequence"]))
            scores.append(score)
    if not mutants:
        raise AssayContractError(f"{path.name}: no non-WT variant rows after filtering")
    return Assay(
        dms_id=dms_id or path.stem,
        rna_type=rna_type,
        mutants=mutants,
        sequences=sequences,
        dms_scores=scores,
    )


# --- mutation parsing + WT reconstruction (masked-marginal LLR support) ----
#
# A mutant string is one or more single-nucleotide substitutions separated by ``,`` or
# ``:`` (e.g. ``"C70T"``, ``"G3A,G21A,G45T"``). Each token is ``<WT><1-based-pos><MUT>``.
# RNAGym/ProteinGym positions are **1-based** → converted to 0-based here (cross-checked
# empirically against the sequence in :func:`assay_reference`, never assumed). Bases are
# normalized U→T (matching :func:`preprocess_sequence`) so they compare against the
# normalized DNA sequence. Indels / multi-char / malformed tokens are *unscoreable* (the
# substitution-only masked-marginal method has no single masked-position analog for them)
# and return ``None`` — the caller skips-and-counts, never silently scores.

_MUTATION_TOKEN_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def _normalize_base(base: str) -> str:
    """RNA→DNA base normalization (matches :func:`preprocess_sequence`)."""
    return base.upper().replace("U", "T")


def parse_mutant(mutant: str) -> list[tuple[int, str, str]] | None:
    """Parse a mutant string → ``[(pos0, wt_base, mut_base), ...]`` (0-based), or ``None``.

    Returns ``None`` (unscoreable) if **any** token is not a clean single-nucleotide
    substitution — an indel, a multi-character token, a non-ACGT base, or a malformed
    token. Multi-site strings (``,``/``:``-separated) parse to one triple per site; the
    additive single-site masked-marginal approximation scores each independently.
    """
    tokens = [t.strip() for t in re.split(r"[,:]", mutant.strip()) if t.strip()]
    if not tokens:
        return None
    out: list[tuple[int, str, str]] = []
    for tok in tokens:
        m = _MUTATION_TOKEN_RE.match(tok)
        if m is None:
            return None  # indel / multi-char / malformed → unscoreable
        wt = _normalize_base(m.group(1))
        pos1 = int(m.group(2))
        mut = _normalize_base(m.group(3))
        if wt not in "ACGT" or mut not in "ACGT" or pos1 < 1:
            return None
        out.append((pos1 - 1, wt, mut))
    return out


def assay_reference(
    assay: Assay, parsed: list[list[tuple[int, str, str]] | None]
) -> str:
    """Reconstruct + verify the assay's reference (WT) sequence (derive-and-verify).

    For every *scoreable* variant, revert its listed mutations (set each mutated position
    back to its WT base) to obtain a candidate reference, cross-checking that the variant's
    own sequence carries the **MUT** base at each position (``sequence[pos0] == mut``). All
    scoreable variants must reconstruct the **identical** reference — otherwise the assay
    violates the single-WT contract (loud :class:`AssayContractError`). The agreed
    reference additionally satisfies ``reference[pos0] == wt`` for every mutation (by
    construction).

    Raises :class:`AssayContractError` if no variant is scoreable or the references diverge.
    """
    reference: str | None = None
    for wi, muts in enumerate(parsed):
        if muts is None:
            continue
        seq = assay.sequences[wi]
        chars = list(seq)
        for pos0, wt, mut in muts:
            if pos0 < 0 or pos0 >= len(seq):
                raise AssayContractError(
                    f"{assay.dms_id}: variant {assay.mutants[wi]!r} position {pos0} out of "
                    f"range for a {len(seq)}-nt sequence."
                )
            if chars[pos0] != mut:
                raise AssayContractError(
                    f"{assay.dms_id}: variant {assay.mutants[wi]!r} declares MUT {mut} at "
                    f"position {pos0} but sequence has {chars[pos0]!r} — mutant string and "
                    "sequence disagree (position base / 1-based offset mismatch)."
                )
            chars[pos0] = wt
        candidate = "".join(chars)
        if reference is None:
            reference = candidate
        elif candidate != reference:
            raise AssayContractError(
                f"{assay.dms_id}: variants reconstruct divergent reference (WT) sequences — "
                "the assay violates the single-WT contract."
            )
    if reference is None:
        raise AssayContractError(
            f"{assay.dms_id}: no scoreable (substitution-only) variant to reconstruct the "
            "reference (WT) from."
        )
    return reference


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Parse the vendored-data manifest (sha256 + n_rows + cds_len + rna_type per assay)."""
    with open(path) as f:
        manifest: dict[str, Any] = yaml.safe_load(f)
    return manifest


def verify_manifest(manifest: dict[str, Any], data_dir: Path | str) -> None:
    """Assert every vendored file matches its recorded sha256. Fails loudly on drift.

    Raises :class:`ManifestHashError` if a file is missing or its digest diverges — the
    integrity gate that keeps a real run from silently scoring tampered/re-fetched data.
    """
    data_dir = Path(data_dir)
    for entry in manifest.get("assays", []):
        rel = entry["file"]
        expected = entry["sha256"]
        fpath = data_dir / rel
        if not fpath.exists():
            raise ManifestHashError(
                f"vendored assay file missing: {fpath} "
                "(run scripts/fetch_rnagym_prok.py to populate)"
            )
        actual = sha256_file(fpath)
        if actual != expected:
            raise ManifestHashError(
                f"sha256 drift for {rel}: manifest={expected} actual={actual}. "
                "The vendored data was modified or re-fetched from a different revision."
            )


def load_assays(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    data_dir: Path | str | None = None,
    verify_hashes: bool = True,
) -> list[Assay]:
    """Load every assay declared in *manifest_path* (optionally verifying integrity first).

    The one-call entry point used by the task's ``prepare()``. ``data_dir`` defaults to
    the manifest's own directory (where ``processed_DMS_files/`` lives).
    """
    manifest_path = Path(manifest_path)
    base = Path(data_dir) if data_dir is not None else manifest_path.parent
    manifest = load_manifest(manifest_path)
    if verify_hashes:
        verify_manifest(manifest, base)
    assays: list[Assay] = []
    for entry in manifest.get("assays", []):
        fpath = base / entry["file"]
        assays.append(
            load_assay_csv(
                fpath,
                dms_id=entry["dms_id"],
                rna_type=entry.get("rna_type", _DEFAULT_RNA_TYPE),
            )
        )
    if not assays:
        raise AssayContractError(f"manifest {manifest_path} declares no assays")
    return assays
