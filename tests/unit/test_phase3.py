"""RNAGym data layer + metrics (vendored upstream).

Gate: gLMBench's metrics reproduce upstream RNAGym ``performance_fitness.py``'s own output
(within fp tolerance) on a frozen fixture — per-assay **and** type-mean macro — via the
vendored file, with no re-derived math; the data-integrity check trips on tampering.

All CPU-only and torch-free: the metric authority (`tasks/_vendor/rnagym/`) is the
unmodified upstream file; `tasks/metrics.py` is a thin wrapper that calls it.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glmbench.tasks import metrics
from glmbench.tasks._vendor.rnagym import performance_fitness as upstream
from glmbench.tasks.rnagym_data import (
    DEFAULT_MANIFEST,
    DEFAULT_RNAGYM_DIR,
    Assay,
    AssayContractError,
    ManifestHashError,
    assay_reference,
    load_assay_csv,
    load_assays,
    load_manifest,
    parse_mutant,
    preprocess_sequence,
    verify_manifest,
)

REPO = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO / "src" / "glmbench" / "tasks" / "_vendor" / "rnagym"
TINY_DIR = REPO / "tests" / "data" / "rnagym_tiny"
TINY_MANIFEST = TINY_DIR / "tiny_manifest.yaml"
FP_TOL = 1e-12


# --- fixtures / helpers --------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260622)


def _synthetic_assays(rng: np.random.Generator) -> list[dict]:
    """A frozen set of per-assay (dms, model) arrays across two RNA types.

    Two ``mRNA-coding`` + one ``Non-coding`` assay, unequal sizes — chosen so the
    type-mean macro differs from a plain mean over assays (catches a flat-mean bug).
    Includes ties in both arrays to exercise rank-averaging in Spearman/AUC/MCC.
    """
    specs = [
        ("A1", "mRNA-coding", 60),
        ("A2", "mRNA-coding", 45),
        ("B1", "Non-coding", 30),
    ]
    out = []
    for name, rna_type, n in specs:
        dms = rng.normal(size=n)
        dms[: n // 6] = dms[0]  # inject ties into the experimental scores
        model = 0.5 * dms + rng.normal(size=n)
        model[-3:] = model[-1]  # ties into the model scores
        out.append({"dms_id": name, "rna_type": rna_type, "dms": dms, "model": model})
    return out


# --- the load-bearing gate: wrapper ≡ upstream, per-assay + macro --------------


def test_assay_metrics_match_upstream_calculate_metrics(rng: np.random.Generator) -> None:
    """Per-assay wrapper output == upstream ``calculate_metrics`` called directly."""
    for a in _synthetic_assays(rng):
        wrapped = metrics.assay_metrics(a["dms"], a["model"])
        direct = upstream.calculate_metrics(
            np.asarray(a["dms"], dtype=float), np.asarray(a["model"], dtype=float)
        )
        assert wrapped["spearman"] == pytest.approx(direct["Spearman"], abs=FP_TOL)
        assert wrapped["auc"] == pytest.approx(direct["AUC"], abs=FP_TOL)
        assert wrapped["mcc"] == pytest.approx(direct["MCC"], abs=FP_TOL)


def test_macro_matches_upstream_type_mean(rng: np.random.Generator) -> None:
    """Type-mean macro == upstream ``calculate_RNA_types_averages_with_se`` directly.

    This pins gLMBench to the benchmark's own published macro definition, not to any
    re-derivation, exactly through the vendored path.
    """
    assays = _synthetic_assays(rng)
    per_assay = [
        {"rna_type": a["rna_type"], **metrics.assay_metrics(a["dms"], a["model"])}
        for a in assays
    ]
    wrapped = metrics.macro(per_assay)

    # Build upstream's expected dataframe shape and call its reducer DIRECTLY.
    rows = []
    for a in assays:
        m = upstream.calculate_metrics(
            np.asarray(a["dms"], dtype=float), np.asarray(a["model"], dtype=float)
        )
        rows.append(
            {
                "RNA_TYPE": a["rna_type"],
                "Spearman_model": m["Spearman"],
                "AUC_model": m["AUC"],
                "MCC_model": m["MCC"],
            }
        )
    df = pd.DataFrame(rows)
    types = sorted(df["RNA_TYPE"].unique())
    direct = upstream.calculate_RNA_types_averages_with_se(
        df, types, ["model"], calculate_se=False
    ).iloc[0]

    assert wrapped["macro_spearman"] == pytest.approx(direct["Spearman_All_Mean"], abs=FP_TOL)
    assert wrapped["macro_auc"] == pytest.approx(direct["AUC_All_Mean"], abs=FP_TOL)
    assert wrapped["macro_mcc"] == pytest.approx(direct["MCC_All_Mean"], abs=FP_TOL)


def test_macro_is_type_mean_not_plain_mean(rng: np.random.Generator) -> None:
    """Guard: the macro is the upstream *type-mean*, which here ≠ a flat mean.

    With 2 mRNA-coding + 1 Non-coding assay of unequal sizes, the mean-of-type-means
    differs from a plain mean over the three assays — so a wrapper that secretly did a
    flat mean would fail this.
    """
    assays = _synthetic_assays(rng)
    per_assay = [
        {"rna_type": a["rna_type"], **metrics.assay_metrics(a["dms"], a["model"])}
        for a in assays
    ]
    macro_spearman = metrics.macro(per_assay)["macro_spearman"]

    spearmans = [p["spearman"] for p in per_assay]
    plain_mean = float(np.mean(spearmans))
    # type-mean = mean( mean(A1,A2), B1 )
    type_mean = float(np.mean([np.mean(spearmans[:2]), spearmans[2]]))

    assert macro_spearman == pytest.approx(type_mean, abs=FP_TOL)
    assert abs(macro_spearman - plain_mean) > 1e-6  # genuinely not a flat mean


# --- optional cross-check oracle (extra, NOT the gate) -------------------------


def _independent_abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """A dep-light re-derivation (scipy ranks + Pearson) — independent of the wrapper."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def test_independent_oracle_agrees_with_upstream(rng: np.random.Generator) -> None:
    """Extra oracle: an independent abs-Spearman ≈ upstream's, on the fixture."""
    for a in _synthetic_assays(rng):
        direct = upstream.calculate_metrics(
            np.asarray(a["dms"], dtype=float), np.asarray(a["model"], dtype=float)
        )
        assert _independent_abs_spearman(a["dms"], a["model"]) == pytest.approx(
            direct["Spearman"], abs=1e-9
        )


# --- vendored-file integrity ---------------------------------------------------


def test_vendored_performance_fitness_sha256_pinned() -> None:
    """The 'verbatim' upstream copy matches its checked-in SHA256SUMS pin.

    A silent edit to the vendored metric authority is caught here (vendored code is
    copied, never edited).
    """
    sums = (VENDOR_DIR / "SHA256SUMS").read_text().strip().splitlines()
    pinned = {}
    for line in sums:
        digest, name = line.split()
        pinned[name.lstrip("*")] = digest
    assert "performance_fitness.py" in pinned, "SHA256SUMS missing performance_fitness.py"
    actual = hashlib.sha256((VENDOR_DIR / "performance_fitness.py").read_bytes()).hexdigest()
    assert actual == pinned["performance_fitness.py"], (
        "vendored performance_fitness.py was edited — re-vendoring is the only "
        "sanctioned change and must bump SHA256SUMS + the benchmark version."
    )


def test_vendor_has_notice() -> None:
    notice = (VENDOR_DIR / "NOTICE").read_text()
    assert "MIT" in notice and "RNAGym" in notice and "bioRxiv" in notice


# --- data integrity (manifest sha256 gate) -------------------------------------


def test_tiny_fixture_manifest_verifies() -> None:
    """The committed tiny fixture matches its own manifest (clean baseline)."""
    manifest = load_manifest(TINY_MANIFEST)
    verify_manifest(manifest, TINY_DIR)  # must not raise


def test_sha256_drift_on_tampered_csv_raises(tmp_path: Path) -> None:
    """Tampering one byte of a vendored CSV trips ManifestHashError."""
    # Copy the fixture, tamper one byte of a data row, point the manifest at the copy.
    work = tmp_path / "rnagym_tiny"
    shutil.copytree(TINY_DIR, work)
    target = work / "TOY1_ECOLX_synthetic.csv"
    text = target.read_text()
    # Flip a digit in a DMS_score without changing the row count / contract.
    tampered = text.replace("0.5", "0.6", 1)
    assert tampered != text
    target.write_text(tampered)

    manifest = load_manifest(work / "tiny_manifest.yaml")
    with pytest.raises(ManifestHashError, match="sha256 drift"):
        verify_manifest(manifest, work)


def test_missing_vendored_file_raises(tmp_path: Path) -> None:
    work = tmp_path / "rnagym_tiny"
    shutil.copytree(TINY_DIR, work)
    (work / "TOY1_ECOLX_synthetic.csv").unlink()
    manifest = load_manifest(work / "tiny_manifest.yaml")
    with pytest.raises(ManifestHashError, match="missing"):
        verify_manifest(manifest, work)


# --- loader contract -----------------------------------------------------------


def test_loader_drops_wt_and_converts_u_to_t(tmp_path: Path) -> None:
    """WT rows dropped; sequences normalized strip/upper/U→T."""
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text(
        "mutant,sequence,DMS_score\n"
        ",AUGGCC,1.0\n"          # WT (empty mutant) — dropped
        "nan,augGCC,0.9\n"       # WT (literal nan) — dropped
        "A1G,GUGGCC, 0.5 \n"     # variant; U→T, upper, score trimmed
        "A2C,gtgGcc,-0.3\n"      # variant; lowercase → upper
    )
    assay = load_assay_csv(csv_path, dms_id="toy", rna_type="mRNA-coding")
    assert len(assay) == 2
    assert assay.mutants == ["A1G", "A2C"]
    assert assay.sequences == ["GTGGCC", "GTGGCC"]  # no U, all upper
    assert assay.dms_scores == [0.5, -0.3]
    assert "U" not in "".join(assay.sequences)


def test_loader_accepts_lowercase_dms_score_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    csv_path.write_text("mutant,sequence,dms_score\nA1G,ACGT,0.2\n")
    assay = load_assay_csv(csv_path)
    assert assay.dms_scores == [0.2]


def test_loader_missing_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("mutant,DMS_score\nA1G,0.2\n")  # no 'sequence'
    with pytest.raises(AssayContractError, match="missing required column"):
        load_assay_csv(csv_path)


def test_loader_missing_score_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("mutant,sequence\nA1G,ACGT\n")
    with pytest.raises(AssayContractError, match="DMS score column"):
        load_assay_csv(csv_path)


def test_loader_empty_after_filter_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("mutant,sequence,DMS_score\n,ACGT,1.0\nnan,ACGT,0.9\n")
    with pytest.raises(AssayContractError, match="no non-WT variant"):
        load_assay_csv(csv_path)


def test_preprocess_sequence() -> None:
    assert preprocess_sequence("  augcUu \n") == "AUGCUU".replace("U", "T")
    assert preprocess_sequence("acgt") == "ACGT"


def test_load_assays_end_to_end_on_tiny_fixture() -> None:
    """The one-call entry point loads all fixture assays with rna_type + integrity."""
    assays = load_assays(TINY_MANIFEST)
    assert len(assays) == 3
    ids = {a.dms_id for a in assays}
    assert ids == {"TOY1_ECOLX_synthetic", "TOY2_ECOLX_synthetic", "TOY3_NONCODE_synthetic"}
    rna_types = {a.rna_type for a in assays}
    assert rna_types == {"mRNA-coding", "Non-coding"}
    for a in assays:
        assert len(a) == len(a.sequences) == len(a.dms_scores) == len(a.mutants) > 0
        assert all("U" not in s for s in a.sequences)


def test_load_assays_can_skip_hash_verification(tmp_path: Path) -> None:
    """verify_hashes=False loads even when a file's bytes drift (explicit opt-out)."""
    work = tmp_path / "rnagym_tiny"
    shutil.copytree(TINY_DIR, work)
    # Append a benign variant row → digest drifts but the contract still holds.
    target = work / "TOY1_ECOLX_synthetic.csv"
    target.write_text(target.read_text() + "Z9Z,ATGGCAAAGCTGGTTGAACGTTTCCAGGACATCGGTAAACCGTTCGCAGCTGGTCTGGAAAAATAA,0.42\n")
    with pytest.raises(ManifestHashError):
        load_assays(work / "tiny_manifest.yaml", verify_hashes=True)
    assays = load_assays(work / "tiny_manifest.yaml", verify_hashes=False)
    assert len(assays) == 3


# --- real-data path (skips on CI where the gitignored CSVs are absent) ---------


@pytest.mark.skipif(
    not (DEFAULT_RNAGYM_DIR / "processed_DMS_files" / "BLAT_ECOLX_Firnberg_2014.csv").exists(),
    reason="vendored RNAGym CSVs absent (gitignored cache); run scripts/fetch_rnagym_prok.py",
)
def test_real_manifest_loads_11_prok_assays() -> None:
    """When the vendored cache is present, the real manifest loads the 11 prok assays."""
    assays = load_assays(DEFAULT_MANIFEST)
    assert len(assays) == 11
    assert all(a.rna_type == "mRNA-coding" for a in assays)
    assert any(a.dms_id.startswith("BLAT_ECOLX") for a in assays)
    # every sequence is DNA (no U) and uppercase.
    for a in assays:
        assert all(s == s.upper() and "U" not in s for s in a.sequences)


# --- mutation parser + WT reconstruction (masked-marginal LLR support) ------


def test_parse_mutant_single_site_1based_to_0based():
    # "C70T" → wt C, 1-based pos 70 → 0-based 69, mut T.
    assert parse_mutant("C70T") == [(69, "C", "T")]


def test_parse_mutant_multi_site_comma_and_colon():
    assert parse_mutant("G3A,G21A,G45T") == [(2, "G", "A"), (20, "G", "A"), (44, "G", "T")]
    # ProteinGym-style colon separator is also accepted.
    assert parse_mutant("A1C:T4G") == [(0, "A", "C"), (3, "T", "G")]


def test_parse_mutant_normalizes_rna_bases():
    # RNA U normalizes to DNA T (matches preprocess_sequence) on both WT and MUT.
    assert parse_mutant("U5C") == [(4, "T", "C")]
    assert parse_mutant("a3u") == [(2, "A", "T")]


def test_parse_mutant_rejects_indels_and_malformed():
    for bad in ("G3", "3A", "GG3A", "G3AA", "ins5A", "G3-", "", "  "):
        assert parse_mutant(bad) is None, bad


def test_parse_mutant_rejects_non_acgt():
    assert parse_mutant("X3A") is None
    assert parse_mutant("G3Z") is None


def _assay(mutants, sequences):
    return Assay(
        dms_id="TEST",
        rna_type="mRNA-coding",
        mutants=mutants,
        sequences=sequences,
        dms_scores=[0.0] * len(mutants),
    )


def test_assay_reference_reconstructs_and_verifies():
    wt = "ACGTACGT"
    # two substitution variants of the same WT
    v1 = "ATGTACGT"  # pos1 C→T
    v2 = "ACGTACTT"  # pos6 G→T
    assay = _assay(["C2T", "G7T"], [v1, v2])
    parsed = [parse_mutant(m) for m in assay.mutants]
    assert assay_reference(assay, parsed) == wt


def test_assay_reference_position_cross_check_trips():
    """A 1-based/0-based slip (sequence base ≠ declared MUT) fails loudly."""
    wt = "ACGTACGT"
    # mutant says C2T but the sequence's pos1 (0-based) is 'C', not 'T' — mismatch.
    assay = _assay(["C2T"], [wt])
    parsed = [parse_mutant(m) for m in assay.mutants]
    with pytest.raises(AssayContractError, match="disagree|MUT"):
        assay_reference(assay, parsed)


def test_assay_reference_divergent_wt_trips():
    assay = _assay(
        ["C2T", "G7T"],
        ["ATGTACGT", "ACGTACTA"],  # second reverts to a different WT (last base A vs T)
    )
    parsed = [parse_mutant(m) for m in assay.mutants]
    with pytest.raises(AssayContractError, match="divergent"):
        assay_reference(assay, parsed)


def test_assay_reference_no_scoreable_trips():
    assay = _assay(["ins3A"], ["ACGTACGT"])  # only an indel → nothing to reconstruct from
    parsed = [parse_mutant(m) for m in assay.mutants]
    with pytest.raises(AssayContractError, match="no scoreable"):
        assay_reference(assay, parsed)


def test_assay_reference_skips_unscoreable_but_uses_substitutions():
    wt = "ACGTACGT"
    assay = _assay(["C2T", "ins3A"], ["ATGTACGT", "ACGTACGT"])
    parsed = [parse_mutant(m) for m in assay.mutants]
    # the indel is skipped; the substitution still reconstructs the WT.
    assert assay_reference(assay, parsed) == wt
