"""The paper's reference data is internally consistent, and the comparison script works.

- ``paper/reference_scores.csv`` and ``paper/LEADERBOARD.md`` regenerate byte-identically from
  ``paper/records`` (they are generated, never typed);
- every record carries exactly the five ``loam_paper`` rows, all ``ok``, and no private
  paths or host names;
- ``scripts/compare_to_paper.py`` passes the LOAM rows re-scored from the released weights
  (``paper/public_route``) bitwise, and fails a record with one perturbed value.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"
PAPER_TASKS = [
    "bacbench-essentiality",
    "bacbench-essentiality-layer-sweep",
    "dgeb-ec-classification-dna",
    "dgeb-ec-classification-dna-layer-sweep",
    "rnagym-dms",
]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _models() -> list[dict[str, str]]:
    with (PAPER / "models.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _records() -> list[Path]:
    return sorted((PAPER / "records").glob("glmb_*.json")) + sorted(
        (PAPER / "public_route").glob("glmb_*.json")
    )


def test_reference_table_regenerates_byte_identically() -> None:
    build = _script("build_reference_table")
    assert build.build_csv(PAPER) == (PAPER / "reference_scores.csv").read_text()
    assert build.build_leaderboard(PAPER) == (PAPER / "LEADERBOARD.md").read_text()


def test_models_csv_points_at_matching_records() -> None:
    models = _models()
    assert len(models) == 13
    for m in models:
        rec = json.loads((PAPER / m["paper_record"]).read_text())
        assert rec["model_hash"] == m["paper_model_hash"]
        if m["public_route_record"]:
            rec = json.loads((PAPER / m["public_route_record"]).read_text())
            assert rec["model_hash"] == m["public_route_model_hash"]
            assert rec["adapter_name"] == "loam-hf"


def test_sources_list_every_record() -> None:
    lines = [ln.split() for ln in (PAPER / "records" / "SOURCES.sha256").read_text().splitlines()
             if ln and not ln.startswith("#")]
    listed = {name for _, name in lines}
    assert listed == {p.name for p in (PAPER / "records").glob("glmb_*.json")}
    assert all(len(sha) == 64 for sha, _ in lines)


@pytest.mark.parametrize("path", _records(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_record_holds_the_five_paper_rows_and_nothing_private(path: Path) -> None:
    rec = json.loads(path.read_text())
    assert [t["task"] for t in rec["tasks"]] == PAPER_TASKS
    assert all(t["status"] == "ok" for t in rec["tasks"])
    assert rec["provenance"].get("host") in (None, "<redacted>")

    def strings(o):  # noqa: ANN001, ANN202
        if isinstance(o, dict):
            for v in o.values():
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)
        elif isinstance(o, str):
            yield o

    assert not [s for s in strings(rec) if s.startswith(("/", "~"))], "absolute path in a record"


def test_public_route_reproduces_the_paper_bitwise() -> None:
    compare = _script("compare_to_paper")
    routes = sorted((PAPER / "public_route").glob("glmb_*.json"))
    assert routes, "no public-route records"
    assert compare.main([str(p) for p in routes] + ["--exact", "--paper", str(PAPER)]) == 0


def _perturbed(tmp_path: Path, delta: float) -> Path:
    src = sorted((PAPER / "public_route").glob("glmb_*.json"))[0]
    rec = copy.deepcopy(json.loads(src.read_text()))
    sweep = next(t for t in rec["tasks"] if t["task"] == "bacbench-essentiality-layer-sweep")
    sweep["metrics"][f"layer_0_{sweep['primary_metric']}"] += delta
    out = tmp_path / src.name
    out.write_text(json.dumps(rec))
    return out


def test_compare_fails_a_perturbed_record(tmp_path: Path) -> None:
    compare = _script("compare_to_paper")
    assert compare.main([str(_perturbed(tmp_path, 0.01)), "--paper", str(PAPER)]) == 1


def test_tolerances_admit_noise_that_exact_mode_rejects(tmp_path: Path) -> None:
    compare = _script("compare_to_paper")
    path = _perturbed(tmp_path, 1e-6)
    assert compare.main([str(path), "--paper", str(PAPER)]) == 0
    assert compare.main([str(path), "--paper", str(PAPER), "--exact"]) == 1
