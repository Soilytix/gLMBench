"""The leaderboard — accumulated results across models.

Append-only per-run result JSON (one file per ``model_hash`` × benchmark), aggregated
to a tidy CSV and rendered to Markdown (overall + per-task). Git-friendly: no DB, no
web server. This package is core (torch-free): numpy/pandas + stdlib only.
"""

from __future__ import annotations

from .render import aggregate_dataframe, render_markdown, write_leaderboard
from .store import (
    build_result_record,
    read_results,
    result_filename,
    write_result,
)

__all__ = [
    "aggregate_dataframe",
    "render_markdown",
    "write_leaderboard",
    "build_result_record",
    "read_results",
    "result_filename",
    "write_result",
]
