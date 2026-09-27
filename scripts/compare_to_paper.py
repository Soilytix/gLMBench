#!/usr/bin/env python
"""Compare your benchmark results with the numbers of the LOAM paper.

::

    # after `glmbench run --model specs/loam/LOAM-25M.yaml --benchmark loam_paper --results results/`
    python scripts/compare_to_paper.py results/loam_paper/glmb_<hash>.json
    python scripts/compare_to_paper.py results/loam_paper/          # every record in a directory
    python scripts/compare_to_paper.py <record> --exact              # require bitwise agreement
    python scripts/compare_to_paper.py <candidate> --against <reference record>   # any two records

**Which paper row.** A record is matched to a paper model by its model hash (``paper/models.csv``
lists the paper hash of each model and, for LOAM, the hash of the same weights served from the
Hugging Face release), else by ``display_name``, else by ``--model-id``.

**What is compared.** For each of the five ``loam_paper`` rows: the headline metric, the
headline metric at every tap of the two layer sweeps, and the Spearman of every RNAGym assay.
All other shared metrics (counts, secondary metrics) are compared too and reported; they fail
the check only with ``--exact``.

**Tolerances.** ``--exact`` requires every value to be bitwise equal, which is what the
reference environment gives (docs/REPRODUCING_THE_PAPER.md). Without it, the per-task tolerances
below apply; they are meant for other GPUs and library versions, where bf16 kernels and BLAS
builds move the last digits. A sweep also passes its best-layer check if it picks the paper's
best tap, or if the paper's best tap scores within tolerance of the new best.

Exit status 1 if any record fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

PAPER_TASKS = (
    "bacbench-essentiality",
    "bacbench-essentiality-layer-sweep",
    "dgeb-ec-classification-dna",
    "dgeb-ec-classification-dna-layer-sweep",
    "rnagym-dms",
)

#: |Δ| allowed on the checked metrics away from the reference environment.
TOLERANCES = {
    # Rank statistics barely move under bf16 noise.
    "rnagym-dms": 0.001,
    # ~27.7k test genes; lbfgs sometimes stops at its iteration cap, which amplifies small
    # feature changes.
    "bacbench-essentiality": 0.002,
    "bacbench-essentiality-layer-sweep": 0.002,
    # 128 test rows, one per class: one flipped prediction moves macro-F1 by ~0.008, so this
    # allows two flips.
    "dgeb-ec-classification-dna": 0.016,
    "dgeb-ec-classification-dna-layer-sweep": 0.016,
}

_LAYER = re.compile(r"^layer_(\d+)_(.+)$")

Metrics = dict[tuple[str, str], float | None]  # (metric, layer) -> value; layer "" = not per-layer


# --- loading -------------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)


def record_tasks(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """task -> {"status", "primary", "metrics"} from a result record."""
    out = {}
    for t in record.get("tasks", []):
        metrics: Metrics = {}
        for key, v in (t.get("metrics") or {}).items():
            m = _LAYER.match(key)
            metrics[(m.group(2), m.group(1)) if m else (key, "")] = _num(v)
        out[t["task"]] = {"status": t.get("status"), "primary": t.get("primary_metric"),
                          "metrics": metrics}
    return out


def paper_tasks(paper: Path, model_id: str) -> dict[str, dict[str, Any]]:
    """task -> {"status", "primary", "metrics"} for one model, from reference_scores.csv."""
    out: dict[str, dict[str, Any]] = {}
    with (paper / "reference_scores.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["model_id"] != model_id:
                continue
            t = out.setdefault(r["task"], {"status": "ok", "primary": None, "metrics": {}})
            t["metrics"][(r["metric"], r["layer"])] = _num(r["value"])
            if r["is_primary"] == "1":
                t["primary"] = r["metric"]
    return out


def identify(record: dict[str, Any], models: list[dict[str, str]], model_id: str | None) -> str | None:
    if model_id:
        return model_id
    h = record.get("model_hash")
    for m in models:
        if h and h in (m["paper_model_hash"], m["public_route_model_hash"]):
            return m["model_id"]
    name = (record.get("model_spec") or {}).get("display_name")
    return next((m["model_id"] for m in models if m["model_id"] == name), None)


# --- comparison ----------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.6f}" if abs(x) < 1e4 else f"{x:.0f}"


def _checked(task: str, metric: str, layer: str, primary: str) -> bool:
    """Is this a score the paper reports (as opposed to a count or a secondary metric)?"""
    if metric == primary:
        return True
    return task == "rnagym-dms" and metric.endswith("_spearman") and layer == ""


def compare_task(task: str, ref: dict[str, Any], cand: dict[str, Any], tol: float,
                 exact: bool) -> tuple[list[str], list[str], bool]:
    """(summary row, detail lines, ok) for one task."""
    primary = ref["primary"] or cand["primary"]
    rm, cm = ref["metrics"], cand["metrics"]
    shared = sorted(set(rm) & set(cm))
    n_equal = sum(1 for k in shared if rm[k] == cm[k])
    ok = True
    worst = 0.0
    other_diff = []
    for k in shared:
        r, c = rm[k], cm[k]
        if r is None or c is None:
            if r != c:
                ok = False
            continue
        d = abs(c - r)
        if _checked(task, k[0], k[1], primary):
            worst = max(worst, d)
            if d > tol:
                ok = False
        elif d > 0:
            other_diff.append(k)
            if exact:
                ok = False
    missing = sorted(k for k in rm if k not in cm and _checked(task, k[0], k[1], primary))
    if missing:
        ok = False

    details: list[str] = []
    best_note = ""
    layers = sorted(int(l) for (m, l) in rm if m == primary and l != "")
    if layers:
        rb, cb = rm.get(("best_layer", "")), cm.get(("best_layer", ""))
        best_ok = rb is not None and cb is not None and int(rb) == int(cb)
        if not best_ok and rb is not None and cb is not None:
            # The paper's best tap, scored by the candidate, is within tolerance of the new best.
            at_paper_best = cm.get((primary, str(int(rb))))
            cand_best = cm.get((primary, str(int(cb))))
            best_ok = (at_paper_best is not None and cand_best is not None
                       and cand_best - at_paper_best <= tol and not exact)
        ok &= best_ok
        tap = lambda x: "—" if x is None else str(int(x))  # noqa: E731
        best_note = f"best tap {tap(rb)} → {tap(cb)}{'' if best_ok else ' ✗'}"
        details.append(f"\n**{task}**: `{primary}` per tap ({best_note})\n")
        details.append("| Tap | Paper | Yours | Δ |\n|---:|---:|---:|---:|")
        for i in layers:
            r, c = rm.get((primary, str(i))), cm.get((primary, str(i)))
            d = "—" if r is None or c is None else f"{c - r:+.2e}"
            details.append(f"| {i} | {_fmt(r)} | {_fmt(c)} | {d} |")
    assays = sorted(k for k in shared if task == "rnagym-dms" and k[0].endswith("_spearman")
                    and k[1] == "" and k[0] != primary)
    if assays:
        details.append(f"\n**{task}**: Spearman per assay\n")
        details.append("| Assay | Paper | Yours | Δ |\n|---|---:|---:|---:|")
        for k in assays:
            r, c = rm[k], cm[k]
            d = "—" if r is None or c is None else f"{c - r:+.2e}"
            details.append(f"| {k[0][: -len('_spearman')]} | {_fmt(r)} | {_fmt(c)} | {d} |")
    if other_diff:
        details.append(f"\n*{task}: {len(other_diff)} other metric(s) differ "
                       f"({'fails --exact' if exact else 'not checked'}):* "
                       + ", ".join(m + (f"@{l}" if l else "") for m, l in other_diff[:12])
                       + (" …" if len(other_diff) > 12 else ""))
    if missing:
        details.append(f"\n*{task}: missing from your record:* "
                       + ", ".join(m + (f"@{l}" if l else "") for m, l in missing))

    r_head, c_head = rm.get((primary, "")), cm.get((primary, ""))
    d_head = "—" if r_head is None or c_head is None else f"{c_head - r_head:+.2e}"
    row = (f"| {task} | `{primary}` | {_fmt(r_head)} | {_fmt(c_head)} | {d_head} | "
           f"{worst:.2e} | {n_equal}/{len(shared)} | {'PASS' if ok else '**FAIL**'} |")
    return [row], details, ok


def compare(ref: dict[str, dict[str, Any]], cand: dict[str, dict[str, Any]], *, title: str,
            atol: float | None, exact: bool) -> tuple[str, bool]:
    out = [title]
    tol_text = "exact (bitwise)" if exact else (f"|Δ| ≤ {atol:g}" if atol is not None
                                                else "per-task tolerances")
    out.append(f"**Tolerance on the checked metrics:** {tol_text}\n")
    out.append("| Task | Headline | Paper | Yours | Δ | Max \\|Δ\\| checked | Bitwise equal | |")
    out.append("|---|---|---:|---:|---:|---:|---:|---|")
    details: list[str] = []
    ok = True
    for task in PAPER_TASKS:
        if task not in ref:
            continue
        c = cand.get(task)
        if c is None or c["status"] != "ok":
            out.append(f"| {task} | — | — | {c and c['status']} | — | — | — | **FAIL** |")
            ok = False
            continue
        tol = 0.0 if exact else (atol if atol is not None else TOLERANCES[task])
        row, det, t_ok = compare_task(task, ref[task], c, tol, exact)
        out += row
        details += det
        ok &= t_ok
    out += details
    out.append(f"\n**Verdict: {'PASS' if ok else 'FAIL'}**\n")
    return "\n".join(out), ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path, help="result record(s), or directories of them")
    ap.add_argument("--paper", type=Path, default=REPO / "paper", help="the paper/ directory")
    ap.add_argument("--model-id", help="paper model id to compare against (default: identify)")
    ap.add_argument("--against", type=Path, help="compare with this record instead of the paper")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--exact", action="store_true", help="require bitwise agreement on every metric")
    g.add_argument("--atol", type=float, help="one |Δ| tolerance for every checked metric")
    ap.add_argument("--out", type=Path, help="also write the Markdown report here")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for p in args.records:
        paths += sorted(p.glob("glmb_*.json")) if p.is_dir() else [p]
    if not paths:
        print("no records found", file=sys.stderr)
        return 2
    with (args.paper / "models.csv").open(newline="") as fh:
        models = list(csv.DictReader(fh))

    reports, all_ok = [], True
    for path in paths:
        record = json.loads(path.read_text())
        cand = record_tasks(record)
        label = (record.get("model_spec") or {}).get("display_name") or record.get("model_hash")
        if args.against:
            ref_rec = json.loads(args.against.read_text())
            ref = record_tasks(ref_rec)
            title = f"## {label} vs. {args.against.name}\n"
        else:
            mid = identify(record, models, args.model_id)
            if mid is None:
                reports.append(f"## {path.name}\n\nNo paper model matches this record (hash "
                               f"{record.get('model_hash')}); pass --model-id.\n")
                all_ok = False
                continue
            ref = paper_tasks(args.paper, mid)
            title = f"## {label} ({path.name}) vs. the paper's {mid}\n"
        report, ok = compare(ref, cand, title=title, atol=args.atol, exact=args.exact)
        reports.append(report)
        all_ok &= ok

    text = "\n".join(reports)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
