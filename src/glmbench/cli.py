"""gLMBench command-line entrypoint.

::

    glmbench models list                       # registered adapters + declared capabilities
    glmbench tasks list                        # registered tasks + required capabilities
    glmbench run --model specs/x.yaml --benchmark rnagym_prok --results results/
                 [--keep-scratch] [--dry-run]
    glmbench leaderboard --benchmark rnagym_prok --results results/ --out <path.md>

This module must stay torch-free (the torch-free import gate): it imports only core.
Importing the adapters + tasks packages here registers the bundled adapters/tasks so the
registry is populated for ``models``/``tasks``/``run``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from glmbench import __version__, registry

# Side-effect imports populate the registries (adapters + tasks) so models/tasks/run
# can resolve registry names without hard-coded imports.
from glmbench import adapters as _adapters  # noqa: F401
from glmbench import tasks as _tasks  # noqa: F401
from glmbench.config.benchmark_spec import resolve_benchmark_spec
from glmbench.config.model_spec import load_model_spec
from glmbench.leaderboard.render import write_leaderboard
from glmbench.leaderboard.store import read_results
from glmbench.run import run_benchmark


def _cmd_models_list() -> int:
    names = registry.list("adapter")
    if not names:
        print("(no adapters registered)")
        return 0
    print("Registered adapters:")
    for name in names:
        cls = registry.resolve("adapter", name)
        caps = sorted(c.value for c in getattr(cls, "CAPABILITIES", frozenset()))
        version = getattr(cls, "adapter_version", "?")
        print(f"  {name}@{version}  capabilities: {caps or '(none)'}")
    return 0


def _cmd_tasks_list() -> int:
    names = registry.list("task")
    if not names:
        print("(no tasks registered)")
        return 0
    print("Registered tasks:")
    for name in names:
        cls = registry.resolve("task", name)
        reqs = sorted(c.value for c in getattr(cls, "required_capabilities", frozenset()))
        version = getattr(cls, "version", "?")
        print(f"  {name}@{version}  requires: {reqs or '(none)'}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    model_spec = load_model_spec(args.model)
    benchmark_spec = resolve_benchmark_spec(args.benchmark)
    record = run_benchmark(
        model_spec,
        benchmark_spec,
        results_dir=args.results,
        keep_scratch=args.keep_scratch,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )
    print(f"Ran '{benchmark_spec.name}' for model {record['model_hash']}:")
    for t in record["tasks"]:
        primary = t["primary_metric"]
        headline = t["metrics"].get(primary) if primary else None
        head_str = f"{primary}={headline:.4f}" if headline is not None else "—"
        print(f"  {t['task']:>16}: {t['status'].upper():>5}  {head_str}")
    if args.results:
        print(f"Wrote result to {Path(args.results) / benchmark_spec.name}/")
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    benchmark_spec = resolve_benchmark_spec(args.benchmark)
    records = read_results(args.results, benchmark_spec.name)
    out = args.out or str(Path(args.results) / benchmark_spec.name / "LEADERBOARD.md")
    csv_out = str(Path(out).with_suffix(".csv"))
    write_leaderboard(
        records,
        benchmark_name=benchmark_spec.name,
        benchmark_version=benchmark_spec.version,
        task_order=benchmark_spec.tasks,
        md_path=out,
        csv_path=csv_out,
    )
    print(f"Wrote leaderboard ({len(records)} model(s)) to {out}, {csv_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glmbench", description="gLMBench benchmark suite")
    parser.add_argument("-V", "--version", action="version", version=f"glmbench {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_models = sub.add_parser("models", help="model adapter registry")
    models_sub = p_models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list registered adapters + capabilities")

    p_tasks = sub.add_parser("tasks", help="task registry")
    tasks_sub = p_tasks.add_subparsers(dest="tasks_command")
    tasks_sub.add_parser("list", help="list registered tasks + required capabilities")

    p_run = sub.add_parser("run", help="run a model spec against a benchmark")
    p_run.add_argument("--model", required=True, help="path to a ModelSpec YAML")
    p_run.add_argument("--benchmark", required=True, help="benchmark name or path")
    p_run.add_argument("--results", default="results", help="results store dir")
    p_run.add_argument("--keep-scratch", action="store_true", help="keep runner scratch dirs")
    p_run.add_argument("--dry-run", action="store_true", help="print runner commands only")
    p_run.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute every task even if a prior record has OK results (default: resume)",
    )

    p_lb = sub.add_parser("leaderboard", help="(re)build CSV + Markdown from stored JSON")
    p_lb.add_argument("--benchmark", required=True, help="benchmark name or path")
    p_lb.add_argument("--results", default="results", help="results store dir")
    p_lb.add_argument("--out", default=None, help="Markdown output path")

    return parser


def main(argv: list[str] | None = None) -> int:
    # Surface INFO+ from the core on the CLI: live per-task progress in run.py, plus the
    # weak-weights-digest / declared-digest identity warnings. A vendored metric module
    # (rnagym/performance_fitness.py) calls logging.basicConfig(WARNING) at import, so a root
    # handler may already exist pinned at WARNING — set the `glmbench` logger to INFO directly
    # (its records propagate to that handler regardless of the root level), and add a handler
    # ourselves only if none exists yet.
    logging.getLogger("glmbench").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "models":
        if args.models_command == "list":
            return _cmd_models_list()
        parser.parse_args(["models", "--help"])
        return 2
    if args.command == "tasks":
        if args.tasks_command == "list":
            return _cmd_tasks_list()
        parser.parse_args(["tasks", "--help"])
        return 2
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "leaderboard":
        return _cmd_leaderboard(args)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
