"""Throttled per-batch progress + ETA emitter for runner mini-batch loops.

A runner processes its flat input list in mini-batches inside a subprocess (one batched
call per task). That subprocess is where ~all the wall-time goes, and without this it emits
nothing: the operator sees a silent terminal for minutes. :class:`BatchProgress`
gives each loop a one-line, wall-clock-**throttled** throughput/ETA readout on ``stderr``::

    [loam-hf/embed] 32768/131072 seqs (25%) | 00:00:48 elapsed | ETA 00:02:25 | 682/s

The line is streamed live to the console by the runner backend (see
``runner_backend.run_and_stream``); this module only formats + throttles it.

Design constraints:

- **Throttled by wall-time** (~5s) so fast ops stay quiet and slow ones surface.
- **A true no-op when disabled** (``GLMBENCH_PROGRESS=0``) — no formatting cost.
- **Never raises**: progress is diagnostic, never load-bearing, so a formatting/IO error
  must never crash the runner it is instrumenting.
- **Stdlib only** (``os`` / ``sys`` / ``time``) — no ``tqdm``, no third-party deps. This
  module is imported inside every runner env, which may be minimal.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO

# Env values (case-insensitive) that turn progress OFF. Anything else (incl. unset) → on.
_FALSY = frozenset({"0", "false", "no", "off", ""})


def progress_enabled() -> bool:
    """Whether progress output is enabled — ``GLMBENCH_PROGRESS`` unset/truthy ⇒ ``True``."""
    return os.environ.get("GLMBENCH_PROGRESS", "1").strip().lower() not in _FALSY


def _fmt_hms(s: float) -> str:
    """Duration as zero-padded ``HH:MM:SS`` (hours grow past 99 if ever needed)."""
    s = max(int(s), 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


class BatchProgress:
    """Throttled throughput/ETA reporter for a single runner mini-batch loop.

    Construct once with the loop's total item count and a ``"<runner>/<op>"`` label, call
    :meth:`update` with the running count of items completed after each mini-batch, then
    :meth:`close` when the loop ends. Output goes to ``stream`` (default ``stderr``),
    newline-terminated (never ``\\r``) so the tee'd ``runner.stderr`` file stays readable.
    """

    def __init__(
        self,
        total: int,
        label: str,
        *,
        unit: str = "items",
        every_seconds: float = 5.0,
        stream: TextIO | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.total = int(total)
        self.label = label
        self.unit = unit
        self.every_seconds = float(every_seconds)
        self._stream = stream if stream is not None else sys.stderr
        # None ⇒ consult the env now (frozen for this instance's lifetime).
        self.enabled = progress_enabled() if enabled is None else bool(enabled)
        self._start = time.perf_counter()
        self._last_emit = self._start
        self._n_done = 0
        self._emitted = False  # did any intermediate line print? (governs close() dedup)

    def update(self, n_done: int) -> None:
        """Record ``n_done`` items complete; print a line when due.

        The **first** update always prints (so the operator sees the workload size and an
        early ETA within the first batch, not after the first 5s throttle window); every
        later intermediate line is throttled to ``every_seconds``.
        """
        if not self.enabled or self.total <= 0:
            return
        try:
            self._n_done = int(n_done)
            now = time.perf_counter()
            due = (now - self._last_emit) >= self.every_seconds
            first = not self._emitted
            if (first or due) and self._n_done < self.total:
                self._emit(now)
                self._last_emit = now
                self._emitted = True
        except Exception:  # noqa: BLE001 - progress must never crash the runner
            pass

    def close(self) -> None:
        """Emit a final ``(100%)`` summary line (unless disabled or nothing to report)."""
        if not self.enabled or self.total <= 0:
            return
        try:
            self._n_done = self.total
            self._emit(time.perf_counter())
        except Exception:  # noqa: BLE001 - progress must never crash the runner
            pass

    def _emit(self, now: float) -> None:
        elapsed = max(now - self._start, 1e-9)
        done = self._n_done
        pct = (100.0 * done / self.total) if self.total else 100.0
        rate = done / elapsed  # items/s
        if done > 0:
            eta = elapsed * (self.total - done) / done
            eta_str = _fmt_hms(eta)
        else:
            eta_str = "--:--:--"
        line = (
            f"[{self.label}] {done}/{self.total} {self.unit} ({pct:.0f}%) | "
            f"{_fmt_hms(elapsed)} elapsed | ETA {eta_str} | {rate:.0f}/s"
        )
        self._stream.write(line + "\n")
        self._stream.flush()
