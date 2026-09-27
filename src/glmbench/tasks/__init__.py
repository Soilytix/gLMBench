"""Benchmark tasks.

Re-exports the task contract (:class:`Task`, :class:`TaskResult`,
:class:`ResultStatus`). Importing this package also imports each task module so the
tasks register themselves in the registry and become discoverable (``glmbench tasks
list``). Adding a task is a localized change: drop a module here, register it, done —
no edits to the runner.
"""

from __future__ import annotations

# Import for registration side effects (keep alphabetical as tasks are added).
from . import (
    bacbench_essentiality,  # noqa: E402,F401  (side-effect import: registers bacbench-essentiality)
    dgeb_ec_classification,  # noqa: E402,F401  (side-effect import: registers dgeb-ec-classification-dna)
    dummy,  # noqa: E402,F401  (side-effect import: registers dummy tasks)
    rnagym_dms,  # noqa: E402,F401  (side-effect import: registers rnagym-dms)
)
from .base import ResultStatus, Task, TaskResult

__all__ = [
    "ResultStatus",
    "Task",
    "TaskResult",
]
