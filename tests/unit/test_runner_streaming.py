"""Live subprocess output streaming in the runner backend.

Proves the runner's output reaches the parent's stream **as it is produced**,
not buffered until the subprocess exits — and that the full transcript is still captured
for the ``runner.stdout``/``runner.stderr`` diagnostics files and the ``RunnerError``
message. CPU-only, no model.
"""

from __future__ import annotations

import sys
import time

import pytest

from glmbench.adapters.runner_backend import (
    LocalSubprocessRunner,
    RunnerError,
    run_and_stream,
)


class _Sink:
    """A forward target that records every write and can react to a marker."""

    def __init__(self, on_marker=None) -> None:
        self.text = ""
        self._on_marker = on_marker

    def write(self, s: str) -> None:
        self.text += s
        if self._on_marker is not None and "MARKER" in s:
            self._on_marker()

    def flush(self) -> None:  # noqa: D401 - stream protocol
        pass


def test_run_and_stream_forwards_live_before_exit(tmp_path) -> None:
    """The child blocks until it observes a sentinel the parent writes *on seeing MARKER*.

    If output were buffered until the child exits (the old ``capture_output`` behavior),
    the parent would never forward MARKER in time, the child would sit on its 30s guard,
    and the wall-time would blow past the bound. Live streaming unblocks it in milliseconds.
    """
    sentinel = tmp_path / "seen.flag"
    child = (
        "import sys, os, time\n"
        "print('MARKER', flush=True)\n"
        f"p = r'{sentinel}'\n"
        "deadline = time.time() + 30.0\n"
        "while not os.path.exists(p) and time.time() < deadline:\n"
        "    time.sleep(0.01)\n"
        "print('DONE', flush=True)\n"
    )
    sink = _Sink(on_marker=lambda: sentinel.write_text("x"))

    t0 = time.perf_counter()
    rc, captured = run_and_stream([sys.executable, "-c", child], forward=sink)
    elapsed = time.perf_counter() - t0

    assert rc == 0
    assert "MARKER" in captured and "DONE" in captured
    assert "MARKER" in sink.text and "DONE" in sink.text
    # Unblocked live, not by the 30s guard → must complete quickly.
    assert elapsed < 10.0, f"streaming was not live (took {elapsed:.1f}s)"


def test_run_and_stream_tag_prefixes_lines() -> None:
    sink = _Sink()
    rc, captured = run_and_stream(
        [sys.executable, "-c", "print('hello')"], forward=sink, tag="[op] "
    )
    assert rc == 0
    assert "[op] hello" in sink.text
    # The captured transcript (for diagnostics) is untagged.
    assert captured.strip() == "hello"


def test_run_and_stream_disabled_still_captures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLMBENCH_PROGRESS", "0")
    sink = _Sink()
    rc, captured = run_and_stream(
        [sys.executable, "-c", "print('quiet')"], forward=sink
    )
    assert rc == 0
    assert "quiet" in captured  # full transcript still returned
    assert sink.text == ""  # but nothing forwarded live (quiet CI path)


def test_run_and_stream_nonzero_returncode() -> None:
    sink = _Sink()
    rc, captured = run_and_stream(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"],
        forward=sink,
    )
    assert rc == 3
    assert "boom" in captured
    assert "boom" in sink.text


def test_real_batchprogress_streams_live_through_run_and_stream() -> None:
    """End-to-end composition proof: the REAL BatchProgress emitting inside a subprocess,
    forwarded live by the REAL run_and_stream — the exact path every instrumented runner
    uses (BatchProgress → stderr → _execute streams it to the console)."""
    child = (
        "from glmbench.runners._progress import BatchProgress\n"
        "p = BatchProgress(100, 'demo/embed', unit='windows', every_seconds=0.0)\n"
        "for done in (25, 50, 75, 100):\n"
        "    p.update(done)\n"
        "p.close()\n"
    )
    sink = _Sink()
    rc, captured = run_and_stream([sys.executable, "-c", child], forward=sink, tag="[demo] ")
    assert rc == 0
    # The formatted BatchProgress lines were both forwarded live and captured.
    assert "[demo/embed]" in sink.text
    assert "windows" in sink.text and "ETA" in sink.text
    assert "(100%)" in sink.text  # close() line made it through
    assert "[demo/embed]" in captured


def test_execute_error_path_keeps_scratch_and_captures(tmp_path) -> None:
    """A failing runner surfaces a RunnerError with the transcript + kept scratch path,
    and the runner.stdout / runner.stderr diagnostic files are still written."""
    backend = LocalSubprocessRunner(
        python_exe=sys.executable,
        runner_module="glmbench.runners._nonexistent_runner_xyz",  # import fails → nonzero
        scratch_dir=str(tmp_path),
        keep_scratch=True,
    )
    with pytest.raises(RunnerError) as ei:
        backend.execute("score_sequences", {"reduction": "mean"}, ["ACGT", "TTGG"])
    msg = str(ei.value)
    assert "score_sequences" in msg
    assert "Scratch kept at" in msg
    # The import error transcript is captured into the message…
    assert "_nonexistent_runner_xyz" in msg or "No module named" in msg
    # …and persisted to the scratch diagnostics files.
    scratch_dirs = list(tmp_path.glob("glmb-*"))
    assert scratch_dirs, "scratch dir should be kept on failure"
    out = (scratch_dirs[0] / "runner.stdout").read_text()
    err = (scratch_dirs[0] / "runner.stderr").read_text()
    assert out == err  # merged transcript written to both
    assert out.strip(), "captured transcript should be non-empty"
