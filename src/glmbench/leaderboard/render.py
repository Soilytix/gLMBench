"""Leaderboard render — stored JSON → tidy CSV + Markdown (overall + per-task).

- :func:`aggregate_dataframe` flattens every stored result record into a tidy CSV: one
  row per model × task, columns = structural keys + ``metric.<name>`` + status.
- :func:`render_markdown` produces the Markdown board: an **overall** table (one row per
  model, headline = mean of each task's ``primary_metric``, ``N/A`` tasks excluded but
  counted in a coverage column) and a **per-task** table for each task. ``N/A``/``ERROR``
  cells are rendered explicitly.

This module is core (torch-free): pandas + stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

NA = "N/A"
ERR = "ERROR"

# Scoring methods that are explicit *fallbacks* from a task's primary capability.
# When any model scored a task via one of these, the per-task table emits a disclosure
# footnote — the cells mix scoring methods (both abs-Spearman-vs-DMS, but the method
# differs and must be disclosed, not silently mixed).
FALLBACK_SCORING_METHODS = frozenset({"masked_marginal_llr"})

# Human-readable labels for scoring-method provenance strings.
_METHOD_LABELS = {
    "sequence_loglikelihood": "sequence log-likelihood",
    "masked_marginal_llr": "masked-marginal LLR",
}


def _method_label(method: str | None) -> str:
    """Human-readable label for a ``scoring_method`` provenance string."""
    if not method:
        return "sequence log-likelihood"
    return _METHOD_LABELS.get(method, method)


def _short(model_hash: str) -> str:
    """A compact, human-glanceable model-hash suffix (after the ``glmb:`` prefix)."""
    return model_hash.split(":", 1)[-1][:10]


def _display_name(record: dict[str, Any]) -> str | None:
    """The optional human-readable model label from the spec (provenance only)."""
    name = (record.get("model_spec") or {}).get("display_name")
    return name or None


def _model_label(record: dict[str, Any]) -> str:
    """Human label keyed by the content hash so identity never collides.

    Form: ``<display_name> · <adapter>@<ver> (<short_hash>)`` when a display_name is set,
    else ``<adapter>@<ver> (<short_hash>)``. The short hash always trails the name, so two
    checkpoints sharing a display_name (or adapter@version) still render distinctly — their
    weights/config differ, so their hashes differ. The full hash is in the Model ID column
    (overall table) and the CSV ``model_hash`` field.
    """
    base = f"{record['adapter_name']}@{record['adapter_version']} ({_short(record['model_hash'])})"
    name = _display_name(record)
    return f"{name} · {base}" if name else base


def _fmt(value: float | None) -> str:
    if value is None:
        return NA
    return f"{value:.4f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored Markdown table from headers + string rows."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body]) if rows else "\n".join([head, sep])


def aggregate_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten result records into a tidy DataFrame — one row per model × task."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        for t in rec["tasks"]:
            row: dict[str, Any] = {
                "model_hash": rec["model_hash"],
                "display_name": _display_name(rec) or "",
                "adapter": rec["adapter_name"],
                "adapter_version": rec["adapter_version"],
                "benchmark": rec["benchmark"]["name"],
                "benchmark_version": rec["benchmark"]["version"],
                "task": t["task"],
                "status": t["status"],
                "primary_metric": t["primary_metric"],
                # Provenance: which scoring path produced this cell. AR causal
                # log-likelihood vs MLM masked-marginal LLR are both abs-Spearman-vs-DMS but
                # the method differs and must be disclosed, not hidden.
                "scoring_method": (t.get("metadata") or {}).get("scoring_method"),
            }
            for k, v in t["metrics"].items():
                row[f"metric.{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def _task_result(record: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    for t in record["tasks"]:
        if t["task"] == task_name:
            return t
    return None


def _headline(record: dict[str, Any], task_order: list[str]) -> tuple[float | None, int, int]:
    """Return (headline mean of primary metrics over OK tasks, n_ok, n_tasks)."""
    primaries: list[float] = []
    n_ok = 0
    for task_name in task_order:
        t = _task_result(record, task_name)
        if t is None or t["status"] != "ok":
            continue
        n_ok += 1
        pm = t["primary_metric"]
        if pm is not None and pm in t["metrics"]:
            primaries.append(float(t["metrics"][pm]))
    headline = sum(primaries) / len(primaries) if primaries else None
    return headline, n_ok, len(task_order)


def _overall_table(records: list[dict[str, Any]], task_order: list[str]) -> str:
    rows: list[tuple[float | None, list[str]]] = []
    for rec in records:
        headline, n_ok, n_tasks = _headline(rec, task_order)
        rows.append(
            (
                headline,
                # The full ``glmb:`` hash is the unique identity column — it disambiguates
                # any two models, including same-named checkpoints with different weights.
                [_model_label(rec), rec["model_hash"], _fmt(headline), f"{n_ok}/{n_tasks}"],
            )
        )
    # sort by headline desc; None (no scorable task) sinks to the bottom
    rows.sort(key=lambda r: (r[0] is not None, r[0] if r[0] is not None else 0.0), reverse=True)
    return _md_table(
        ["Model", "Model ID (hash)", "Headline (mean primary)", "Coverage"],
        [r[1] for r in rows],
    )


def _status_cell(t: dict[str, Any] | None) -> str | None:
    """Return an explicit non-OK cell string, or None if the task is OK and present."""
    if t is None:
        return NA
    if t["status"] == "na":
        return NA
    if t["status"] == "error":
        return ERR
    return None


def _scoring_method(t: dict[str, Any] | None) -> str | None:
    """The scoring-method provenance string of a task result (or ``None``)."""
    if t is None:
        return None
    return (t.get("metadata") or {}).get("scoring_method")


def _scoring_method_footnote(records: list[dict[str, Any]], task_name: str) -> str:
    """Disclosure footnote when ≥1 model scored this task via a non-primary method.

    Returns ``""`` when no model used a fallback method (no footnote). The disclosure is the
    footnote only (no per-cell column / superscript).
    """
    fallback_users: list[tuple[str, str]] = []
    for rec in records:
        t = _task_result(rec, task_name)
        if t is None or t["status"] != "ok":
            continue
        method = _scoring_method(t)
        if method in FALLBACK_SCORING_METHODS:
            fallback_users.append((_model_label(rec), _method_label(method)))
    if not fallback_users:
        return ""
    clauses = "; ".join(
        f"{label} scored via {method} (MLM fallback), not sequence log-likelihood"
        for label, method in fallback_users
    )
    return (
        f"> ⚠️ **Mixed scoring methods on `{task_name}`:** {clauses}. "
        "Masked-marginal LLR and sequence log-likelihood both reduce to abs-Spearman vs "
        "DMS — a rank metric, the AR-vs-MLM comparison ProteinGym/RNAGym make — so the "
        "column is comparable, but the underlying method differs."
    )


def _per_task_table(records: list[dict[str, Any]], task_name: str) -> str:
    # Collect the metric-key union for this task across models (primary metric first).
    metric_keys: list[str] = []
    primary: str | None = None
    for rec in records:
        t = _task_result(rec, task_name)
        if t is None:
            continue
        if primary is None and t["primary_metric"] is not None:
            primary = t["primary_metric"]
        for k in t["metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)
    ordered_metrics = (
        [primary, *[k for k in sorted(metric_keys) if k != primary]]
        if primary is not None
        else sorted(metric_keys)
    )

    headers = ["Model", "Status", *ordered_metrics]
    body: list[tuple[float | None, list[str]]] = []
    for rec in records:
        t = _task_result(rec, task_name)
        status = t["status"].upper() if t is not None else NA
        cell = _status_cell(t)
        if cell is not None:
            row = [_model_label(rec), status, *[cell for _ in ordered_metrics]]
            body.append((None, row))
            continue
        assert t is not None  # OK implies present
        metric_cells = [_fmt(t["metrics"].get(m)) for m in ordered_metrics]
        sort_val = (
            float(t["metrics"][primary])
            if primary is not None and primary in t["metrics"]
            else None
        )
        body.append((sort_val, [_model_label(rec), status, *metric_cells]))

    body.sort(key=lambda r: (r[0] is not None, r[0] if r[0] is not None else 0.0), reverse=True)
    return _md_table(headers, [r[1] for r in body])


def render_markdown(
    records: list[dict[str, Any]],
    *,
    benchmark_name: str,
    benchmark_version: str,
    task_order: list[str],
) -> str:
    """Render the full Markdown leaderboard for a benchmark."""
    lines: list[str] = []
    lines.append(f"# Leaderboard — {benchmark_name} (v{benchmark_version})")
    lines.append("")
    lines.append(f"{len(records)} model(s) · {len(task_order)} task(s).")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    if records:
        lines.append(_overall_table(records, task_order))
    else:
        lines.append("_No results yet._")
    lines.append("")
    for task_name in task_order:
        lines.append(f"## Task: {task_name}")
        lines.append("")
        lines.append(_per_task_table(records, task_name))
        footnote = _scoring_method_footnote(records, task_name)
        if footnote:
            lines.append("")
            lines.append(footnote)
        lines.append("")
    lines.append("---")
    lines.append("")
    footer = (
        "Cites RNAGym (Notin et al., bioRxiv 2025.06.16.660049; MIT). "
        "`N/A` = required capability missing; `ERROR` = task ran but failed."
    )
    # The LLR sentence is a claim about THIS board, so it is emitted only when some model
    # on it actually took the fallback: on an all-AR board it would be false.
    if any(
        _scoring_method(t) in FALLBACK_SCORING_METHODS
        for rec in records
        for task_name in task_order
        if (t := _task_result(rec, task_name)) is not None and t["status"] == "ok"
    ):
        footer += (
            " Variant-effect tasks score MLM encoders (e.g. gLM2) via **masked-marginal "
            "LLR** when causal sequence log-likelihood is unavailable; LLR and "
            "log-likelihood are both rank-correlated (abs-Spearman) against the DMS "
            "measurements, so the scores are comparable, but the scoring method differs "
            "(disclosed per-task above)."
        )
    lines.append(footer)
    lines.append("")
    return "\n".join(lines)


def write_leaderboard(
    records: list[dict[str, Any]],
    *,
    benchmark_name: str,
    benchmark_version: str,
    task_order: list[str],
    md_path: str | Path,
    csv_path: str | Path | None = None,
) -> Path:
    """Render + write the Markdown board (and optionally the tidy CSV). Returns md path."""
    md = render_markdown(
        records,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        task_order=task_order,
    )
    md_out = Path(md_path)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    if csv_path is not None:
        df = aggregate_dataframe(records)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return md_out
