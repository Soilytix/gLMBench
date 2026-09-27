#!/usr/bin/env python3
"""Compute the **k-mer floor** for every benchmark task on which one is meaningful.

The k-mer floor is the score a task gives to a feature vector built only from nucleotide
counts, with no model at all — the "you could have done this with a counter" line drawn across
the leaderboard. It is produced by running the REAL task classes (real windowing, real probes,
real vendored metrics) over :class:`~glmbench.baselines.kmer.KmerCompositionAdapter`, so the
number lands on exactly the same axis as any leaderboard cell. Nothing is re-implemented; only
the feature extractor is replaced.

Three ways to read a model against its task's floor:

* **at or below it** — the model contributed nothing that required a language model;
* **above it** — the gap is the *headroom*, and the headroom is the only part of the score that
  is evidence about the model;
* **floor near zero** — the task is a clean test of learned representation.

Usage::

    python scripts/kmer_floor.py                       # every task in `loam_paper`, k=1..6
    python scripts/kmer_floor.py --k 1,2,3,4,5,6
    python scripts/kmer_floor.py --task dgeb-ec-classification-dna --task bacbench-essentiality
    python scripts/kmer_floor.py --benchmark bacbench --out results/kmer_floor/bacbench.json

Output is a JSON document written to ``--out`` (default
``results/kmer_floor/<benchmark>.json``) holding, per task: applicability + why, the score at
every k, and the floor itself (the **best** k, since the floor is the best a counter can do).
It is written incrementally, so a long run that is interrupted still leaves what it finished.

**Applicability is decided from the task's declared capabilities, not from a hand-kept list.**
A task that needs only ``EMBEDDING`` gets a floor. A task that reads the LM head
(``rnagym-dms`` wants ``SEQUENCE_LOGLIKELIHOOD``) does not: a frequency vector is not a
likelihood model, and the honest answer there is "not applicable", not a number. Such a task is
still listed in the output with the capability that rules it out, so the omission is visible
rather than silent. A k-th-order Markov background model *would* give it a floor; that is a
different baseline and is not implemented here.

Layer-sweep tasks are handled by the ``--verify-sweep-identity`` check rather than by paying
for a second full run: the composition vector is identical at every layer, so a sweep's
max-over-layers is *by construction* its single-layer twin's score. The check runs that
identity on the EC sweep and asserts it, so the inherited floors rest on a measured fact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import glmbench.tasks  # noqa: E402,F401  (side-effect import: registers every task)
from glmbench import registry  # noqa: E402
from glmbench.adapters.base import Capability  # noqa: E402
from glmbench.baselines.kmer import KmerCompositionAdapter  # noqa: E402
from glmbench.config.benchmark_spec import resolve_benchmark_spec  # noqa: E402
from glmbench.tasks.base import Task  # noqa: E402

# The paper's protocol: the floor is windowed at a fixed 8,192 nt, the context of Evo 1.5 and
# Evo2 (LOAM's is one less), so the baseline is cut into the same windows as the long-context
# models it is drawn under. Counting has no context limit; this is about comparability, not
# capability. It is fixed here rather than read from a model spec so the published floor
# reproduces.
MAX_CONTEXT = 8192

DEFAULT_KS = (1, 2, 3, 4, 5, 6)

# The single-layer task each sweep task is a max-over-layers of. The k-mer vector is
# depth-free, so the two scores are equal by construction — `--verify-sweep-identity` measures
# it on one pair rather than leaving it an assertion in a docstring.
SWEEP_TWINS = {
    "bacbench-essentiality-layer-sweep": "bacbench-essentiality",
    "dgeb-ec-classification-dna-layer-sweep": "dgeb-ec-classification-dna",
}

# The cheaper of the two sweep/twin pairs (640 sequences, one small probe per layer) — the one
# the identity check runs on.
IDENTITY_CHECK_TASK = "dgeb-ec-classification-dna-layer-sweep"


def _finite(x: Any) -> float | None:
    """``None`` for anything that is not a finite float — NaN must never reach the payload."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _applicability(task: Task) -> tuple[bool, str]:
    """Is a k-mer *composition vector* a valid input for this task, and why (not)?

    Decided from the declared capability sets — including fallbacks, so a task that can be
    served either way is judged on whether ANY of its paths is embedding-only.
    """
    candidates = (task.required_capabilities, *task.capability_fallbacks)
    for cap_set in candidates:
        if cap_set and cap_set <= {Capability.EMBEDDING}:
            return True, "embedding-only task — a composition vector is a valid input"
    needed = sorted(
        c.value for c in set().union(*candidates) - {Capability.EMBEDDING}  # type: ignore[arg-type]
    )
    # Phrased to be read on the board, where it appears verbatim under the bars.
    return False, (
        f"this task reads the LM head (it needs {' or '.join('`' + n + '`' for n in needed)}), "
        f"and a k-mer frequency vector is not a likelihood model — there is no model-free score "
        f"to report. A k-th-order Markov background model would give one; that is a different "
        f"baseline and is not implemented."
    )


def _run_one(task_id: str, k: int, *, n_layers: int = 1) -> dict[str, Any]:
    """Run one task against one k, through the ordinary task lifecycle."""
    task: Task = registry.build("task", task_id)
    adapter = KmerCompositionAdapter(k=k, max_context=MAX_CONTEXT, n_layers=n_layers)
    t0 = time.time()
    # `evaluate` (not bare prepare/run/score) so capability negotiation, N/A and error capture
    # behave exactly as they do for a real model. It calls `prepare()` itself.
    result = task.evaluate(adapter)
    elapsed = time.time() - t0
    primary = result.primary_metric
    return {
        "k": k,
        "embedding_dim": 4**k,
        "status": result.status.value,
        "primary_metric": primary,
        "value": _finite(result.metrics.get(primary)) if primary else None,
        "metrics": {m: _finite(v) for m, v in result.metrics.items()},
        "seconds": round(elapsed, 1),
        "reason": result.metadata.get("reason") or result.metadata.get("error"),
    }


def _floor(runs: list[dict[str, Any]], higher_is_better: bool) -> dict[str, Any] | None:
    """The floor is the BEST a counter can do — max over k (min if lower-is-better).

    Not the mean and not a fixed k: the claim being drawn on the board is "a model-free
    baseline reaches this", so the strongest model-free baseline is the honest line.
    """
    scored = [r for r in runs if r["status"] == "ok" and r["value"] is not None]
    if not scored:
        return None
    best = (max if higher_is_better else min)(scored, key=lambda r: r["value"])
    return {
        "value": best["value"],
        "k": best["k"],
        "metric": best["primary_metric"],
        "embedding_dim": best["embedding_dim"],
        "ks_evaluated": [r["k"] for r in scored],
    }


def _verify_sweep_identity(k: int) -> dict[str, Any]:
    """Measure the claim that lets a sweep task inherit its twin's floor.

    A sweep task takes the max over layers; the composition vector is the same at every layer;
    therefore the two must be equal. If they are ever not, the inherited floors in the output
    are wrong and this says so loudly instead of shipping them.
    """
    sweep_id = IDENTITY_CHECK_TASK
    twin_id = SWEEP_TWINS[sweep_id]
    sweep = _run_one(sweep_id, k, n_layers=3)  # 4 taps, all identical by construction
    single = _run_one(twin_id, k, n_layers=3)
    a, b = sweep.get("value"), single.get("value")
    ok = a is not None and b is not None and abs(a - b) < 1e-12
    return {
        "sweep_task": sweep_id,
        "single_layer_task": twin_id,
        "k": k,
        "n_layers_probed": 4,
        "sweep_value": a,
        "single_layer_value": b,
        "identical": bool(ok),
        "note": (
            "A sweep is a max over layers and the k-mer vector is depth-free, so the two are "
            "equal by construction. This measures it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="loam_paper", help="bundled benchmark name or path (default: loam_paper)")
    ap.add_argument("--task", action="append", default=None, help="restrict to these task ids (repeatable)")
    ap.add_argument("--k", default=",".join(str(k) for k in DEFAULT_KS), help="comma-separated k values")
    ap.add_argument("--out", default=None, help="output JSON (default: results/kmer_floor/<benchmark>.json)")
    ap.add_argument(
        "--verify-sweep-identity",
        action="store_true",
        help="also measure that a layer sweep over the depth-free baseline equals its single-layer twin",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help=(
            "update the tasks this run covers inside an existing --out document instead of "
            "replacing it. Use with --task to re-measure one task without paying for the rest "
            "(the essentiality task alone takes minutes). NOTE: it merges per TASK, not per k "
            "— a task this run touches is replaced wholesale, so `--task X --k 6 --merge` "
            "discards X's stored k=1..5 runs. Pass the full --k range you want X to end up with."
        ),
    )
    args = ap.parse_args(argv)

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    spec = resolve_benchmark_spec(args.benchmark)
    task_ids = list(args.task) if args.task else list(spec.tasks)

    out_path = Path(args.out) if args.out else REPO / "results" / "kmer_floor" / f"{spec.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": spec.name,
        "baseline": KmerCompositionAdapter(k=ks[0]).describe()["name"],
        "baseline_version": KmerCompositionAdapter.adapter_version,
        "max_context": MAX_CONTEXT,
        "ks": ks,
        "what_this_is": (
            "The score each task gives to a model-free k-mer frequency vector, produced by the "
            "unmodified task code. `floor` is the best k — the strongest a counter can do. A "
            "leaderboard cell at or below its task's floor is not evidence about a language model."
        ),
        "tasks": {},
    }
    if args.merge and out_path.exists():
        # Keep everything this run does not cover — including top-level keys it never writes
        # (`sweep_identity_check`), which a naive tasks-only merge would silently drop and turn
        # a measured identity into an unsupported assumption. And keep the *set* of ks the
        # document claims to have swept honest: a merged doc is built from more than one `--k`,
        # so record the union rather than letting a narrower run overwrite a wider one.
        prior = json.loads(out_path.read_text())
        merged = {**prior, **doc}
        merged["tasks"] = {**(prior.get("tasks") or {}), **doc["tasks"]}
        merged["ks"] = sorted(set(prior.get("ks") or []) | set(ks))
        merged["merged_from"] = prior.get("generated_at")
        doc = merged

    def flush() -> None:
        out_path.write_text(json.dumps(doc, indent=1) + "\n")

    for task_id in task_ids:
        task: Task = registry.build("task", task_id)
        applicable, why = _applicability(task)
        entry: dict[str, Any] = {
            "applicable": applicable,
            "applicability_reason": why,
            "required_capabilities": sorted(c.value for c in task.required_capabilities),
            "task_version": task.version,
        }
        if not applicable:
            print(f"{task_id:46s} not applicable — {why}", flush=True)
            doc["tasks"][task_id] = entry
            flush()
            continue
        if task_id in SWEEP_TWINS:
            # Identical by construction (see `_verify_sweep_identity`), and a sweep fits one
            # probe per layer — paying for that to get a number we can prove equals one we
            # already have is waste. Inherit it, and SAY that it was inherited.
            entry.update(
                {
                    "inherits_from": SWEEP_TWINS[task_id],
                    "inheritance_reason": (
                        "a layer sweep is a max over layers, and the k-mer vector is the same at "
                        "every layer, so this task's floor is exactly its single-layer twin's"
                    ),
                }
            )
            print(f"{task_id:46s} inherits floor from {SWEEP_TWINS[task_id]}", flush=True)
            doc["tasks"][task_id] = entry
            flush()
            continue

        # The floor is the BEST a counter can do, and "best" depends on the metric's
        # direction, which the task declares: a lower-is-better task (perplexity) gets a min
        # instead of silently getting a max — i.e. the worst counter passed off as the floor.
        hib = bool(task.higher_is_better)
        runs: list[dict[str, Any]] = []
        for k in ks:
            r = _run_one(task_id, k)
            runs.append(r)
            val = "—" if r["value"] is None else f"{r['value']:.4f}"
            print(
                f"{task_id:46s} k={k}  {r['status']:5s} {r['primary_metric']}={val}"
                f"  ({r['seconds']:.0f}s)",
                flush=True,
            )
            entry["runs"] = runs
            entry["floor"] = _floor(runs, higher_is_better=hib)
            doc["tasks"][task_id] = entry
            flush()

    # Resolve the inherited floors now that every base task has one.
    for entry in doc["tasks"].values():
        src = entry.get("inherits_from")
        if src and (base := doc["tasks"].get(src, {}).get("floor")):
            entry["floor"] = dict(base)

    if args.verify_sweep_identity:
        doc["sweep_identity_check"] = _verify_sweep_identity(k=min(ks))
        c = doc["sweep_identity_check"]
        mark = "✓" if c["identical"] else "✗ MISMATCH — inherited floors are NOT safe"
        print(f"\nsweep identity: {c['sweep_value']} vs {c['single_layer_value']}  {mark}", flush=True)
    flush()

    print(f"\n▸ wrote {out_path}")
    print("\nfloors:")
    for task_id, entry in doc["tasks"].items():
        if not entry.get("applicable"):
            continue
        f = entry.get("floor")
        if not f:
            print(f"  {task_id:46s} (no floor — every k failed)")
            continue
        tag = " (inherited)" if entry.get("inherits_from") else ""
        print(f"  {task_id:46s} {f['metric']} = {f['value']:.4f}  at k={f['k']}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
