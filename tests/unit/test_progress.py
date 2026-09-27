"""The BatchProgress throughput/ETA emitter.

Pure CPU, no model. Drives the emitter against an in-memory stream and a monkeypatched
clock to assert the line format, the disabled no-op, and the wall-clock throttle.
"""

from __future__ import annotations

import io

import pytest

from glmbench.runners import _progress
from glmbench.runners._progress import BatchProgress, progress_enabled


def test_update_and_close_emit_expected_fields() -> None:
    buf = io.StringIO()
    prog = BatchProgress(1000, "x/op", unit="seqs", every_seconds=0.0, stream=buf, enabled=True)
    for done in range(100, 1001, 100):
        prog.update(done)
    prog.close()
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln]
    assert lines, "expected at least one progress line"
    # Every line carries the label, unit, the total denominator, a %, an ETA and a rate.
    import re

    for ln in lines:
        assert ln.startswith("[x/op] ")
        assert "seqs" in ln
        assert "/1000 " in ln
        assert "%)" in ln
        assert "ETA" in ln
        assert "/s" in ln
        # elapsed is HH:MM:SS; ETA is HH:MM:SS (or --:--:-- before any progress).
        assert re.search(r"\d{2}:\d{2}:\d{2} elapsed", ln), ln
        assert re.search(r"ETA (\d{2}:\d{2}:\d{2}|--:--:--)", ln), ln
    # The final (close) line is 100%.
    assert "(100%)" in lines[-1]
    assert "1000/1000" in lines[-1]


def test_disabled_is_a_no_op() -> None:
    buf = io.StringIO()
    prog = BatchProgress(1000, "x/op", every_seconds=0.0, stream=buf, enabled=False)
    for done in range(100, 1001, 100):
        prog.update(done)
    prog.close()
    assert buf.getvalue() == ""


def test_env_toggle_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLMBENCH_PROGRESS", "0")
    assert progress_enabled() is False
    buf = io.StringIO()
    # enabled=None ⇒ consult the env (which we just set to 0).
    prog = BatchProgress(1000, "x/op", every_seconds=0.0, stream=buf, enabled=None)
    prog.update(500)
    prog.close()
    assert buf.getvalue() == ""
    monkeypatch.setenv("GLMBENCH_PROGRESS", "1")
    assert progress_enabled() is True


def test_zero_total_never_emits() -> None:
    buf = io.StringIO()
    prog = BatchProgress(0, "x/op", every_seconds=0.0, stream=buf, enabled=True)
    prog.update(0)
    prog.close()
    assert buf.getvalue() == ""


def test_throttle_limits_intermediate_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    # A controllable clock: perf_counter returns whatever we set.
    fake = {"t": 1000.0}
    monkeypatch.setattr(_progress.time, "perf_counter", lambda: fake["t"])

    buf = io.StringIO()
    prog = BatchProgress(1000, "x/op", every_seconds=5.0, stream=buf, enabled=True)
    # The FIRST update always prints (early workload/ETA); the rest are throttled by the
    # frozen clock so no further intermediate line appears.
    for done in range(100, 1001, 100):
        prog.update(done)
    assert buf.getvalue().count("\n") == 1, "only the first update should print under a frozen clock"
    assert "100/1000" in buf.getvalue()  # the first update's line
    # close() always emits its final 100% line.
    prog.close()
    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 2
    assert "(100%)" in lines[-1]


def test_throttle_emits_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {"t": 1000.0}
    monkeypatch.setattr(_progress.time, "perf_counter", lambda: fake["t"])
    buf = io.StringIO()
    prog = BatchProgress(1000, "x/op", every_seconds=5.0, stream=buf, enabled=True)

    prog.update(100)  # first update → always prints
    assert buf.getvalue().count("\n") == 1
    fake["t"] = 1002.0  # +2s < window ⇒ throttled
    prog.update(200)
    assert buf.getvalue().count("\n") == 1
    fake["t"] = 1006.0  # +6s ≥ window ⇒ another line
    prog.update(400)
    assert buf.getvalue().count("\n") == 2
    prog.close()
    assert buf.getvalue().count("\n") == 3
