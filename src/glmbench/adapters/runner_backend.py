"""Runner backends — execute a runner script in the adapter's foreign env.

Adapters do not import their model into the benchmark process; each shells out to
its own environment (a sibling venv/conda env, or a container) and exchanges data
over files. A backend owns one round trip:

1. create a scratch dir,
2. write the input JSONL + the :class:`~glmbench.adapters.wire.Request`,
3. build + run the command (``LocalSubprocessRunner`` invokes the runner module in
   another Python; ``DockerRunner`` runs it inside a container with the scratch dir
   bind-mounted),
4. read + validate the :class:`~glmbench.adapters.wire.Response`, load its payload
   into memory, and clean up the scratch dir (kept on failure or ``keep_scratch``).

Loud failures: a non-zero exit raises :class:`RunnerError` with the
runner's full captured stdout/stderr; a missing response/output file raises. A
``dry_run`` backend prints the exact command and executes nothing (used in tests).

This module is core (torch-free): stdlib + numpy only.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from glmbench.runners._progress import progress_enabled

from .wire import (
    Request,
    Response,
    WireProtocolError,
    read_arrays_npz,
    read_json_payload,
    write_sequences_jsonl,
    write_variant_jsonl,
)

# Default module names of the runner scripts each backend invokes inside the
# adapter env (``python -m <module> <request.json>``).
RUNNER_MODULE_DEFAULT = "glmbench.runners.echo_runner"


def run_and_stream(
    cmd: list[str],
    *,
    forward: TextIO | None = None,
    tag: str = "",
) -> tuple[int, str]:
    """Run *cmd*, forwarding its output live to *forward* while capturing it in full.

    The runner subprocess is where a task spends nearly all its wall-time; the mini-batch
    loop inside it emits throttled progress lines (see :mod:`glmbench.runners._progress`).
    This helper is what makes those lines *visible*: instead of
    ``subprocess.run(capture_output=True)`` (which buffers everything until the process
    exits), it reads the child's **merged** stdout+stderr
    line-by-line and echoes each line to *forward* (default ``sys.stderr``) as it arrives,
    while accumulating the same text so callers keep the full transcript for diagnostics
    and error messages.

    Used by :meth:`RunnerBackend._execute`; usable by any runner that launches a nested
    subprocess of its own (same need). When progress is disabled
    (``GLMBENCH_PROGRESS=0``) or *forward* is ``None`` it falls back to a plain buffered
    run (quiet CI) but still returns the complete captured text.

    Returns ``(returncode, captured_text)`` — the merged stdout+stderr as one string.
    """
    sink = forward if forward is not None else sys.stderr
    if not progress_enabled():
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert popen.stdout is not None
    for line in popen.stdout:
        chunks.append(line)
        try:
            sink.write(f"{tag}{line}" if tag else line)
            sink.flush()
        except Exception:  # noqa: BLE001 - forwarding is diagnostic, never fatal
            pass
    popen.stdout.close()
    returncode = popen.wait()
    return returncode, "".join(chunks)


class RunnerError(RuntimeError):
    """A runner exited non-zero or violated the protocol. Carries captured output."""


@dataclass
class RunResult:
    """What a backend hands back after a round trip.

    ``payload`` is loaded into memory before the scratch dir is cleaned: for an
    ``.npz`` output it is ``{name: ndarray}``; for a ``.json`` output it is the
    parsed object; empty for a dry run.
    """

    response: Response
    payload: dict[str, Any]
    scratch_dir: str


class RunnerBackend(ABC):
    """Base class for execution backends. One instance per adapter."""

    backend_name: str = "base"

    def __init__(
        self,
        *,
        scratch_dir: str | Path | None = None,
        keep_scratch: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.scratch_root = Path(scratch_dir) if scratch_dir is not None else None
        self.keep_scratch = keep_scratch
        self.dry_run = dry_run

    @abstractmethod
    def build_command(self, request_path: str, scratch_dir: str) -> list[str]:
        """Return the argv that runs the runner script over *request_path*."""

    def _make_scratch(self) -> Path:
        if self.scratch_root is not None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="glmb-", dir=str(self.scratch_root)))
        return Path(tempfile.mkdtemp(prefix="glmb-"))

    def execute(
        self,
        op: str,
        params: dict[str, Any],
        sequences: list[str],
        ids: list[str] | None = None,
        *,
        output_name: str = "output.npz",
    ) -> RunResult:
        """Run one batched op end to end and return its loaded payload."""
        return self._execute(
            op,
            params,
            lambda input_path: write_sequences_jsonl(input_path, sequences, ids),
            output_name=output_name,
        )

    def execute_variant(
        self,
        op: str,
        params: dict[str, Any],
        items: list[dict[str, Any]],
        ids: list[str] | None = None,
        *,
        output_name: str = "output.npz",
    ) -> RunResult:
        """Run one batched **variant-effect** op end to end (variant-record JSONL input).

        Mirrors :meth:`execute` but serializes ``{"id", "reference", "mutations"}`` records
        (via :func:`~glmbench.adapters.wire.write_variant_jsonl`) instead of ``{"id","seq"}``
        — the masked-marginal LLR path needs the reference + mutation list per item.
        """
        return self._execute(
            op,
            params,
            lambda input_path: write_variant_jsonl(input_path, items, ids),
            output_name=output_name,
        )

    def _execute(
        self,
        op: str,
        params: dict[str, Any],
        write_input: Any,
        *,
        output_name: str,
    ) -> RunResult:
        """Shared round-trip core: scratch + request + input file + run + read payload.

        *write_input* is a callback ``(input_path) -> None`` that serializes the op's input
        file; the two public entry points (:meth:`execute`, :meth:`execute_variant`) differ
        only in which JSONL writer they pass.
        """
        scratch = self._make_scratch()
        input_path = scratch / "input.jsonl"
        output_path = scratch / output_name
        request = Request(
            op=op,
            params=params,
            input_path=str(input_path),
            output_path=str(output_path),
        )
        request_path = scratch / "request.json"
        write_input(input_path)
        request.write(request_path)
        cmd = self.build_command(str(request_path), str(scratch))

        if self.dry_run:
            print(" ".join(cmd))
            return RunResult(
                response=Response(
                    op=op, status="dry_run", output_path=None, meta={"command": cmd}
                ),
                payload={},
                scratch_dir=str(scratch),
            )

        # Stream the runner's output live to the console (so the in-loop progress lines
        # are visible as they happen, not buffered until exit), while capturing the full
        # merged transcript for diagnostics. stdout
        # and stderr are merged; both scratch files receive the same transcript.
        returncode, captured = run_and_stream(cmd, forward=sys.stderr, tag=f"[{op}] ")
        (scratch / "runner.stdout").write_text(captured)
        (scratch / "runner.stderr").write_text(captured)
        if returncode != 0:
            raise RunnerError(
                f"Runner exited {returncode} for op '{op}' "
                f"({self.backend_name}). Scratch kept at {scratch}.\n"
                f"--- command ---\n{' '.join(cmd)}\n"
                f"--- output (stdout+stderr) ---\n{captured}"
            )

        response_path = scratch / "response.json"
        if not response_path.exists():
            raise RunnerError(
                f"Runner produced no response.json for op '{op}'. Scratch kept at "
                f"{scratch}.\n--- output (stdout+stderr) ---\n{captured}"
            )
        response = Response.read(response_path)
        if response.status == "error":
            raise RunnerError(
                f"Runner reported error for op '{op}': {response.error}. "
                f"Scratch kept at {scratch}."
            )
        payload = self._load_payload(response, scratch)

        if not self.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        return RunResult(response=response, payload=payload, scratch_dir=str(scratch))

    @staticmethod
    def _load_payload(response: Response, scratch: Path) -> dict[str, Any]:
        if response.output_path is None:
            raise WireProtocolError("Successful response carries no output_path.")
        out = Path(response.output_path)
        if not out.exists():
            raise RunnerError(f"Runner output file missing: {out} (scratch {scratch}).")
        if out.suffix == ".npz":
            return read_arrays_npz(out)
        if out.suffix == ".json":
            return {"json": read_json_payload(out)}
        raise WireProtocolError(f"Unknown runner output extension: {out.suffix}")


class LocalSubprocessRunner(RunnerBackend):
    """Run the runner module in a sibling Python env (``<python_exe> -m <module>``).

    ``python_exe`` points at the adapter's *own* environment (so the core's env
    stays clean); gLMBench must be importable there (installed, or the module on
    ``PYTHONPATH``).
    """

    backend_name = "local"

    def __init__(
        self,
        *,
        python_exe: str,
        runner_module: str = RUNNER_MODULE_DEFAULT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.python_exe = python_exe
        self.runner_module = runner_module

    def build_command(self, request_path: str, scratch_dir: str) -> list[str]:
        return [self.python_exe, "-m", self.runner_module, request_path]


class DockerRunner(RunnerBackend):
    """Run the runner module inside a container, bind-mounting the scratch dir.

    The scratch dir is mounted at the same absolute path inside the container so
    the request/output paths in the wire files resolve on both sides.
    """

    backend_name = "docker"

    def __init__(
        self,
        *,
        image: str,
        runner_module: str,
        python_exe: str = "python",
        gpus: str | None = None,
        docker_exe: str = "docker",
        extra_args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.image = image
        self.runner_module = runner_module
        self.python_exe = python_exe
        self.gpus = gpus
        self.docker_exe = docker_exe
        self.extra_args = list(extra_args) if extra_args else []

    def build_command(self, request_path: str, scratch_dir: str) -> list[str]:
        cmd = [self.docker_exe, "run", "--rm"]
        if self.gpus:
            cmd += ["--gpus", self.gpus]
        cmd += ["-v", f"{scratch_dir}:{scratch_dir}"]
        cmd += self.extra_args
        cmd += [self.image, self.python_exe, "-m", self.runner_module, request_path]
        return cmd
