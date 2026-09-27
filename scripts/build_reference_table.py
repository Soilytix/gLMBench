#!/usr/bin/env python
"""Generate ``paper/reference_scores.csv`` and ``paper/LEADERBOARD.md`` from the paper records.

The LOAM manuscript's benchmark numbers ship as the records they were read from
(``paper/records/glmb_*.json``). This script flattens them into one long table, so every
number the paper reports (and every number it could have) is one row::

    python scripts/build_reference_table.py            # rewrites paper/reference_scores.csv + LEADERBOARD.md
    python scripts/build_reference_table.py --check    # exit 1 if the committed files are out of date

The table is generated, never typed: ``tests/unit/test_paper_reference.py`` regenerates it and
requires it to be byte-identical to the committed file.

Columns of ``reference_scores.csv``:

``model_id``      public model id (``paper/models.csv``)
``model_hash``    the paper record's model hash
``task``          one of the five ``loam_paper`` task rows
``metric``        the metric's key in the record, without its ``layer_<i>_`` prefix
``layer``         the tap index for per-layer sweep metrics, else empty
``is_primary``    1 for the task's headline metric (the value the paper plots), else 0
``value``         the recorded value, written with ``repr`` so it round-trips exactly
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The five task rows of the ``loam_paper`` benchmark, in its order.
PAPER_TASKS = (
    "bacbench-essentiality",
    "bacbench-essentiality-layer-sweep",
    "dgeb-ec-classification-dna",
    "dgeb-ec-classification-dna-layer-sweep",
    "rnagym-dms",
)
COLUMNS = ("model_id", "model_hash", "task", "metric", "layer", "is_primary", "value")
_LAYER = re.compile(r"^layer_(\d+)_(.+)$")


def read_models(paper: Path) -> list[dict[str, str]]:
    with (paper / "models.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return repr(float(value))
    raise TypeError(f"non-numeric metric value {value!r}")


def rows_for_record(model_id: str, record: dict) -> list[dict[str, str]]:
    by_task = {t["task"]: t for t in record["tasks"]}
    rows = []
    for task in PAPER_TASKS:
        t = by_task[task]
        primary = t["primary_metric"]
        for key in sorted(t["metrics"]):
            m = _LAYER.match(key)
            layer, metric = (m.group(1), m.group(2)) if m else ("", key)
            rows.append({
                "model_id": model_id,
                "model_hash": record["model_hash"],
                "task": task,
                "metric": metric,
                "layer": layer,
                "is_primary": "1" if (key == primary) else "0",
                "value": _fmt(t["metrics"][key]),
            })
    return rows


def build_csv(paper: Path) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for m in read_models(paper):
        record = json.loads((paper / m["paper_record"]).read_text())
        if record["model_hash"] != m["paper_model_hash"]:
            raise SystemExit(f"{m['paper_record']}: model_hash does not match paper/models.csv")
        # Sort numerically within each (task, metric) so a sweep reads 0, 1, 2, ... 10.
        rows = rows_for_record(m["model_id"], record)
        rows.sort(key=lambda r: (PAPER_TASKS.index(r["task"]), r["metric"],
                                 -1 if r["layer"] == "" else int(r["layer"])))
        w.writerows(rows)
    return buf.getvalue()


def build_leaderboard(paper: Path) -> str:
    """The paper's headline values, one row per model, in paper/models.csv order."""
    floor = json.loads((paper / "kmer_floor.json").read_text())["tasks"]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x:.4f}"

    def probe(rec: dict, task: str) -> str:
        by_task = {t["task"]: t for t in rec["tasks"]}
        last, sweep = by_task[task], by_task[f"{task}-layer-sweep"]
        best = sweep["metrics"][sweep["primary_metric"]]
        tap = sweep["metrics"].get("best_layer")
        tap_s = "" if tap is None else f" ({int(tap)})"
        return f"{fmt(last['metrics'][last['primary_metric']])} | {fmt(best)}{tap_s}"

    lines = [
        "# The LOAM paper's gLMBench results",
        "",
        "Generated from `paper/records/` by `scripts/build_reference_table.py`; do not edit.",
        "Last = the single-layer row (last layer); best = the layer-sweep row, with its tap index in",
        "brackets. The best tap is chosen on the test split, so it is a retrospective upper bound.",
        "Scores compare within a column only: never average AUROC, macro-F1 and Spearman.",
        "",
        "| Model | Params | Essentiality AUROC, last | best (tap) | EC macro-F1, last | best (tap) "
        "| RNAGym Spearman | RNAGym scoring | Record |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for m in read_models(paper):
        rec = json.loads((paper / m["paper_record"]).read_text())
        rna = next(t for t in rec["tasks"] if t["task"] == "rnagym-dms")
        method = (rna.get("metadata") or {}).get("scoring_method", "")
        params = int(m["params_measured"])
        p_s = f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.1f}M"
        lines.append(
            f"| {m['model_id']} | {p_s} | {probe(rec, 'bacbench-essentiality')} "
            f"| {probe(rec, 'dgeb-ec-classification-dna')} "
            f"| {fmt(rna['metrics'][rna['primary_metric']])} | {method.replace('_', ' ')} "
            f"| [`{m['paper_model_hash']}`]({m['paper_record']}) |"
        )
    floors = []
    for task in ("bacbench-essentiality", "dgeb-ec-classification-dna", "rnagym-dms"):
        f = (floor.get(task) or {}).get("floor")
        floors.append(f"{task} {fmt(f['value'])} (k = {f['k']})" if f else f"{task} not applicable")
    lines += ["", "Model-free k-mer floor (`kmer_floor.json`): " + "; ".join(floors) + ".", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", type=Path, default=REPO / "paper", help="the paper/ directory")
    ap.add_argument("--check", action="store_true", help="only check that the files are up to date")
    args = ap.parse_args(argv)

    outputs = {
        args.paper / "reference_scores.csv": build_csv(args.paper),
        args.paper / "LEADERBOARD.md": build_leaderboard(args.paper),
    }
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for p in stale:
            print(f"out of date: {p}", file=sys.stderr)
        return 1 if stale else 0
    for p, text in outputs.items():
        p.write_text(text)
    n_rows = outputs[args.paper / "reference_scores.csv"].count("\n") - 1
    print(f"wrote {args.paper / 'reference_scores.csv'} ({n_rows} rows) and LEADERBOARD.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
