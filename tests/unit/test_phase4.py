"""RNAGym DMS task (the first real, model-agnostic task).

Gate: the RNAGym-DMS task yields a complete, well-formed ``TaskResult`` against the
``EchoAdapter`` and correctly returns ``N/A`` against an embedding-only adapter. Stress
tests: all per-assay keys + macro + BLAT present; empty/short sequences handled; truncation
logged + counted when a sequence exceeds the adapter's context; the batch-first invariant
(one ``score_sequences`` call for all variants) is honored.

CPU-only and torch-free — rides the ``EchoAdapter`` (deterministic fake scores), the
committed tiny fixture, and synthetic in-tmp assays.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from glmbench.adapters.base import Capability, ModelAdapter
from glmbench.adapters.echo import EchoAdapter
from glmbench.tasks import metrics
from glmbench.tasks.base import ResultStatus
from glmbench.tasks.rnagym_dms import RnagymDmsTask

REPO = Path(__file__).resolve().parents[2]
TINY_MANIFEST = REPO / "tests" / "data" / "rnagym_tiny" / "tiny_manifest.yaml"
DEFAULT_MANIFEST = (
    REPO / "src" / "glmbench" / "tasks" / "data" / "rnagym" / "prok_coding_manifest.yaml"
)
REAL_CSV = (
    REPO
    / "src"
    / "glmbench"
    / "tasks"
    / "data"
    / "rnagym"
    / "processed_DMS_files"
    / "BLAT_ECOLX_Firnberg_2014.csv"
)

_BASES = "ACGT"


def _wt(seq_len: int) -> str:
    """A deterministic WT sequence of length ``seq_len`` starting with a start codon."""
    body = ("ATG" + "ACGTACGTGC" * ((seq_len // 10) + 1))[:seq_len]
    return body


def _write_assay_csv(path: Path, *, n_variants: int, seq_len: int) -> None:
    """Write one synthetic substitution-only assay CSV (WT row + ``n_variants`` rows)."""
    wt = _wt(seq_len)
    lines = ["mutant,sequence,DMS_score", f",{wt},1.0"]  # WT row (empty mutant) → dropped
    for i in range(n_variants):
        pos = i % seq_len
        old = wt[pos]
        new = _BASES[(_BASES.index(old) + 1 + (i % 3)) % 4]
        variant = wt[:pos] + new + wt[pos + 1 :]
        score = round(0.1 + 0.07 * i - 0.003 * (i * i % 11), 4)
        lines.append(f"{old}{pos + 1}{new},{variant},{score}")
    path.write_text("\n".join(lines) + "\n")


def _write_benchmark(
    tmp_path: Path, specs: list[tuple[str, str, int, int]]
) -> Path:
    """Build a synthetic manifest + CSVs in ``tmp_path``; return the manifest path.

    ``specs`` = list of ``(dms_id, rna_type, n_variants, seq_len)``. The manifest omits
    sha256 (callers use ``verify_hashes=False``) so synthetic assays need no real digest.
    """
    proc = tmp_path / "processed_DMS_files"
    proc.mkdir(parents=True, exist_ok=True)
    entries = []
    for dms_id, rna_type, n_variants, seq_len in specs:
        rel = f"processed_DMS_files/{dms_id}.csv"
        _write_assay_csv(tmp_path / rel, n_variants=n_variants, seq_len=seq_len)
        entries.append(
            f"  - dms_id: {dms_id}\n"
            f"    file: {rel}\n"
            f'    rna_type: "{rna_type}"\n'
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("processed_dir: processed_DMS_files\nassays:\n" + "".join(entries))
    return manifest


class _CountingAdapter(ModelAdapter):
    """Wraps an EchoAdapter, recording each ``score_sequences`` call (batch-first probe)."""

    name = "counting-echo"
    adapter_version = "0.1.0"
    CAPABILITIES = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})

    def __init__(self, inner: EchoAdapter) -> None:
        self._inner = inner
        self.calls: list[int] = []

    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return self._inner.model_hash()

    def describe(self) -> dict:
        return self._inner.describe()

    def score_sequences(self, sequences: list[str], *, reduction: str = "mean") -> list[float]:
        self.calls.append(len(sequences))
        return self._inner.score_sequences(sequences, reduction=reduction)


# --- the gate: well-formed result vs EchoAdapter; N/A vs embedding-only -------


def test_gate_well_formed_result_against_echo() -> None:
    """RnagymDmsTask vs EchoAdapter on the tiny fixture → OK, well-formed result."""
    task = RnagymDmsTask(TINY_MANIFEST)
    result = task.evaluate(EchoAdapter())

    assert result.status is ResultStatus.OK
    assert result.task == "rnagym-dms"
    assert result.primary_metric == "macro_spearman"

    m = result.metrics
    # macro trio + accounting present and finite.
    for key in ("macro_spearman", "macro_auc", "macro_mcc"):
        assert key in m and m[key] == m[key]  # not NaN
    assert m["n_assays"] == 3.0
    assert m["n_missing_predictions"] == 0.0

    # One per-assay spearman key per fixture assay.
    per_assay_keys = [k for k in m if k.endswith("_spearman") and not k.startswith("macro_")]
    assert set(per_assay_keys) == {
        "TOY1_ECOLX_synthetic_spearman",
        "TOY2_ECOLX_synthetic_spearman",
        "TOY3_NONCODE_synthetic_spearman",
    }
    # The tiny fixture has no BLAT_ECOLX assay → no breakout line.
    assert "blat_ecolx_spearman" not in m

    # Metadata accounting.
    assert result.metadata["n_missing"] == 0
    assert result.metadata["reduction"] == "mean"
    assert len(result.metadata["per_assay"]) == 3
    n_scored = result.metadata["n_scored"]
    assert n_scored == sum(p["n_variants"] for p in result.metadata["per_assay"]) > 0


def test_gate_na_against_embedding_only_adapter() -> None:
    """An embedding-only adapter → first-class N/A, no crash (capability negotiation)."""
    adapter = EchoAdapter(capabilities={Capability.EMBEDDING})
    result = RnagymDmsTask(TINY_MANIFEST).evaluate(adapter)

    assert result.status is ResultStatus.NA
    assert result.primary_metric is None
    assert result.metrics == {}
    assert result.metadata["missing_capabilities"] == ["sequence_loglikelihood"]


# --- BLAT breakout + full per-assay shape (synthetic, names mirror the real set) ---


def test_blat_breakout_and_per_assay_keys(tmp_path: Path) -> None:
    """A BLAT_ECOLX assay surfaces a ``blat_ecolx_spearman`` line == its per-assay value."""
    manifest = _write_benchmark(
        tmp_path,
        [
            ("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 8, 60),
            ("CCDB_ECOLI_Adkar_2012", "mRNA-coding", 7, 45),
            ("IF1_ECOLI_Kelsic_2016", "mRNA-coding", 6, 30),
        ],
    )
    task = RnagymDmsTask(manifest, verify_hashes=False)
    result = task.evaluate(EchoAdapter())

    assert result.status is ResultStatus.OK
    m = result.metrics
    assert "blat_ecolx_spearman" in m
    assert m["blat_ecolx_spearman"] == m["BLAT_ECOLX_Firnberg_2014_spearman"]
    assert result.metadata["blat_dms_id"] == "BLAT_ECOLX_Firnberg_2014"
    # one per-assay key per assay (the BLAT breakout line is counted separately)
    per_assay_keys = [k for k in m if k in {f"{p['dms_id']}_spearman" for p in result.metadata["per_assay"]}]
    assert len(per_assay_keys) == 3


def test_macro_routed_through_upstream_wrapper(tmp_path: Path) -> None:
    """The reported macro equals ``metrics.macro`` over the per-assay metadata."""
    manifest = _write_benchmark(
        tmp_path,
        [
            ("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 8, 60),
            ("RNC_ECOLI_Weeks_2023", "Non-coding", 7, 45),
        ],
    )
    task = RnagymDmsTask(manifest, verify_hashes=False)
    result = task.evaluate(EchoAdapter())

    expected = metrics.macro(result.metadata["per_assay"])
    assert result.metrics["macro_spearman"] == pytest.approx(expected["macro_spearman"])
    assert result.metrics["macro_auc"] == pytest.approx(expected["macro_auc"])
    assert result.metrics["macro_mcc"] == pytest.approx(expected["macro_mcc"])


# --- batch-first invariant: one call for ALL variants -------------------------


def test_single_batched_score_call(tmp_path: Path) -> None:
    """All variants across all assays are scored in exactly ONE adapter call."""
    manifest = _write_benchmark(
        tmp_path,
        [
            ("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 5, 30),
            ("CCDB_ECOLI_Adkar_2012", "mRNA-coding", 6, 30),
            ("IF1_ECOLI_Kelsic_2016", "mRNA-coding", 4, 30),
        ],
    )
    adapter = _CountingAdapter(EchoAdapter())
    task = RnagymDmsTask(manifest, verify_hashes=False)
    result = task.evaluate(adapter)

    assert result.status is ResultStatus.OK
    assert len(adapter.calls) == 1  # exactly one batched call
    assert adapter.calls[0] == 5 + 6 + 4  # every variant in that one call


# --- truncation: logged + counted, never silent --------------------------------


def test_truncation_logged_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Over-context variants are skipped, counted, and logged loudly (not truncated)."""
    manifest = _write_benchmark(
        tmp_path,
        [
            ("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 6, 30),   # in-context
            ("CCDB_ECOLI_Adkar_2012", "mRNA-coding", 7, 100),     # all over-context
        ],
    )
    task = RnagymDmsTask(manifest, verify_hashes=False, max_context=50)
    with caplog.at_level(logging.WARNING):
        result = task.evaluate(EchoAdapter())

    assert result.status is ResultStatus.OK  # the in-context assay still scores
    assert result.metrics["n_missing_predictions"] == 7.0  # the whole 100-nt assay
    assert any("max_context" in r.message for r in caplog.records)

    # The over-context assay reports NaN metrics + n_missing; excluded from the macro,
    # which stays finite (computed from the in-context assay alone).
    over = next(
        p for p in result.metadata["per_assay"] if p["dms_id"] == "CCDB_ECOLI_Adkar_2012"
    )
    assert over["n_variants"] == 0 and over["n_missing"] == 7
    assert result.metrics["macro_spearman"] == result.metrics["macro_spearman"]  # not NaN


def test_adapter_declared_max_context_is_honored(tmp_path: Path) -> None:
    """When the task has no cap, the adapter's own ``max_context`` attribute is used."""
    manifest = _write_benchmark(
        tmp_path, [("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 6, 100)]
    )
    adapter = EchoAdapter()
    adapter.max_context = 50  # type: ignore[attr-defined]
    task = RnagymDmsTask(manifest, verify_hashes=False)  # no explicit cap
    result = task.evaluate(adapter)

    # The whole 100-nt assay is over the adapter's 50-nt cap → everything skipped → ERROR.
    assert result.status is ResultStatus.ERROR
    assert result.metadata["n_missing"] == 6


def test_all_over_context_is_error(tmp_path: Path) -> None:
    """If every variant is over-context, the task is a loud ERROR, not an all-NaN row."""
    manifest = _write_benchmark(
        tmp_path, [("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 6, 60)]
    )
    task = RnagymDmsTask(manifest, verify_hashes=False, max_context=1)
    result = task.evaluate(EchoAdapter())

    assert result.status is ResultStatus.ERROR
    assert result.primary_metric is None
    assert "0 in-context variants" in result.metadata["error"]


# --- short sequences are handled without crashing ----------------------------


def test_short_sequences_handled(tmp_path: Path) -> None:
    """A tiny assay (short CDS, few variants) scores cleanly — no edge-case crash."""
    manifest = _write_benchmark(
        tmp_path,
        [
            ("BLAT_ECOLX_Firnberg_2014", "mRNA-coding", 4, 6),
            ("CCDB_ECOLI_Adkar_2012", "mRNA-coding", 3, 9),
        ],
    )
    result = RnagymDmsTask(manifest, verify_hashes=False).evaluate(EchoAdapter())
    assert result.status is ResultStatus.OK
    assert result.metrics["n_missing_predictions"] == 0.0


# --- integrity gate still bites through the task -------------------------------


def test_verify_hashes_default_trips_on_tampered_data(tmp_path: Path) -> None:
    """With verify_hashes on, a tampered CSV trips the integrity gate (loud, in prepare).

    ``prepare`` runs *outside* ``evaluate``'s try block by design, so a data
    integrity failure propagates loudly rather than being swallowed as an ERROR row —
    a run must never quietly score tampered data.
    """
    import shutil

    from glmbench.tasks.rnagym_data import ManifestHashError

    work = tmp_path / "rnagym_tiny"
    shutil.copytree(TINY_MANIFEST.parent, work)
    target = work / "TOY1_ECOLX_synthetic.csv"
    target.write_text(target.read_text().replace("0.5", "0.6", 1))

    with pytest.raises(ManifestHashError, match="sha256 drift"):
        RnagymDmsTask(work / "tiny_manifest.yaml").evaluate(EchoAdapter())


# --- real-data path (skips on CI where the gitignored CSVs are absent) ---------


@pytest.mark.skipif(
    not REAL_CSV.exists(),
    reason="vendored RNAGym CSVs absent (gitignored cache); run scripts/fetch_rnagym_prok.py",
)
def test_real_11_assays_yield_11_per_assay_keys() -> None:
    """When the cache is present, the real benchmark yields all 11 per-assay keys + BLAT."""
    result = RnagymDmsTask(DEFAULT_MANIFEST).evaluate(EchoAdapter())
    assert result.status is ResultStatus.OK
    assert result.metrics["n_assays"] == 11.0
    per_assay_keys = [
        k
        for k in result.metrics
        if k in {f"{p['dms_id']}_spearman" for p in result.metadata["per_assay"]}
    ]
    assert len(per_assay_keys) == 11
    assert "blat_ecolx_spearman" in result.metrics
    assert result.metrics["n_missing_predictions"] == 0.0  # prok CDS ≪ echo "context"


# --- masked-marginal LLR fallback (MLM path) + AR regression -----------------


def test_llr_fallback_via_echo_mlm_yields_ok_and_method() -> None:
    """An MLM-only adapter (MASKED_MARGINAL_LLR) scores the tiny fixture via the fallback."""
    mlm = EchoAdapter(capabilities={Capability.MASKED_MARGINAL_LLR})
    result = RnagymDmsTask(TINY_MANIFEST).evaluate(mlm)

    assert result.status is ResultStatus.OK
    assert result.metadata["scoring_method"] == "masked_marginal_llr"
    for key in ("macro_spearman", "macro_auc", "macro_mcc"):
        assert result.metrics[key] == result.metrics[key]  # finite (not NaN)
    # tiny fixture is all clean substitutions → nothing unscoreable.
    assert result.metadata["n_unscoreable"] == 0
    assert result.metrics["n_missing_predictions"] == 0.0


def test_ar_path_records_method_and_is_unchanged_by_fallback_declaration() -> None:
    """AR adapters keep choosing the primary; declaring the fallback changes nothing.

    A full-capability echo (offers both SEQUENCE_LOGLIKELIHOOD and MASKED_MARGINAL_LLR)
    must select the primary and produce metrics identical to a score-only echo.
    """
    full = RnagymDmsTask(TINY_MANIFEST).evaluate(EchoAdapter())  # all caps
    score_only = RnagymDmsTask(TINY_MANIFEST).evaluate(
        EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD})
    )
    assert full.metadata["scoring_method"] == "sequence_loglikelihood"
    assert score_only.metadata["scoring_method"] == "sequence_loglikelihood"
    # byte-for-byte identical metric values (the AR path is untouched).
    assert full.metrics == score_only.metrics


def test_llr_and_loglik_differ_in_general() -> None:
    """The two scoring paths are genuinely different computations (sanity)."""
    ar = RnagymDmsTask(TINY_MANIFEST).evaluate(EchoAdapter())
    mlm = RnagymDmsTask(TINY_MANIFEST).evaluate(
        EchoAdapter(capabilities={Capability.MASKED_MARGINAL_LLR})
    )
    assert ar.metrics["macro_spearman"] != mlm.metrics["macro_spearman"]


def test_llr_indel_skipped_and_counted(tmp_path: Path) -> None:
    """An indel variant is unscoreable under LLR → skipped and counted, never scored."""
    proc = tmp_path / "processed_DMS_files"
    proc.mkdir(parents=True)
    wt = "ATGACGTACGTACGTACGTTAA"
    # 3 clean substitutions + 1 indel-style (unparseable) mutant.
    rows = ["mutant,sequence,DMS_score", f",{wt},1.0"]
    subs = [(4, "C"), (7, "A"), (10, "G")]
    for i, (pos, new) in enumerate(subs):
        old = wt[pos]
        if old == new:
            new = "ACGT"[("ACGT".index(old) + 1) % 4]
        var = wt[:pos] + new + wt[pos + 1 :]
        rows.append(f"{old}{pos + 1}{new},{var},{0.2 + 0.1 * i}")
    # An indel: deletion-style mutant string the parser rejects; sequence one shorter.
    rows.append(f"delA5,{wt[:4] + wt[5:]},0.9")
    (proc / "IDEL_ECOLX.csv").write_text("\n".join(rows) + "\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "processed_dir: processed_DMS_files\n"
        "assays:\n"
        "  - dms_id: IDEL_ECOLX\n"
        "    file: processed_DMS_files/IDEL_ECOLX.csv\n"
        '    rna_type: "mRNA-coding"\n'
    )
    result = RnagymDmsTask(manifest, verify_hashes=False).evaluate(
        EchoAdapter(capabilities={Capability.MASKED_MARGINAL_LLR})
    )
    assert result.status is ResultStatus.OK
    assert result.metadata["scoring_method"] == "masked_marginal_llr"
    assert result.metadata["n_unscoreable"] == 1
    assert result.metrics["n_missing_predictions"] == 1.0
    # the 3 clean substitutions were scored.
    assert result.metadata["n_scored"] == 3
