"""The task contract — lifecycle + capability negotiation + result statuses.

A :class:`Task` declares the model *capabilities* it needs, requests data from the
adapter for every instance it must score (in **one** batched call), and reduces the
returned data to a score + metadata on CPU. Capability negotiation happens *before*
``prepare``/``run``: a task that needs a capability the adapter lacks returns a
first-class ``N/A`` result with metadata — never a crash, never a silent zero.

This module is core (torch-free): it only references the (torch-free) adapter contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from glmbench.adapters.base import Capability, ModelAdapter


class ResultStatus(str, Enum):  # noqa: UP042 - (str, Enum) form matches Capability
    """The three outcomes a task can record for a model."""

    OK = "ok"
    NA = "na"  # a required capability was missing
    ERROR = "error"  # the task ran but failed (the exception is recorded)


@dataclass
class TaskResult:
    """A task's verdict for one model.

    Attributes:
        task: The task's registry name.
        status: :class:`ResultStatus`.
        primary_metric: Which key in ``metrics`` is THE headline number (``None`` for
            ``NA``/``ERROR``).
        metrics: Flat ``name -> float`` map; per-assay and macro numbers both live here.
        metadata: Free-form provenance — n scored, n missing, truncations, timings, and
            the reason on ``NA``/``ERROR``.
    """

    task: str
    status: ResultStatus
    primary_metric: str | None
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status.value,
            "primary_metric": self.primary_metric,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskResult:
        """Reconstruct a ``TaskResult`` from its :meth:`to_dict` form (used by resume)."""
        return cls(
            task=d["task"],
            status=ResultStatus(d["status"]),
            primary_metric=d.get("primary_metric"),
            metrics=dict(d.get("metrics") or {}),
            metadata=dict(d.get("metadata") or {}),
        )


class Task(ABC):
    """Base class for every benchmark task.

    Subclasses set :attr:`name`, :attr:`version`, :attr:`required_capabilities`,
    :attr:`higher_is_better`, and implement :meth:`prepare` (load+validate data, no model),
    :meth:`run` (request data from the adapter, batched), and :meth:`score` (reduce to
    metrics on CPU). The :meth:`evaluate` driver wires negotiation + error capture around
    them.
    """

    name: str = ""
    version: str = ""
    required_capabilities: frozenset[Capability] = frozenset()
    # Direction of the primary metric. Readers that rank or take a best-of (e.g. the k-mer
    # floor in ``scripts/kmer_floor.py``) take a max when True and a min when False.
    higher_is_better: bool = True
    # Ordered alternative capability sets, tried in order when the primary
    # (``required_capabilities``) is unmet. The default ``()`` declares none. A *scoring*
    # task uses this to declare an ordered fallback — e.g. a variant-effect task falls
    # back from causal ``SEQUENCE_LOGLIKELIHOOD`` to MLM ``MASKED_MARGINAL_LLR``.
    capability_fallbacks: tuple[frozenset[Capability], ...] = ()
    # The capability set ``evaluate`` selected for this run (set before ``prepare``/``run``
    # so ``run`` can branch on it). ``None`` until negotiation picks one; defaults to the
    # primary for direct ``run`` calls that bypass ``evaluate``.
    _chosen_capabilities: frozenset[Capability] | None = None

    # --- forward-pass reuse (adapters/reuse.py) ----------------------------
    # Tasks sharing a ``fusion_group`` read the SAME corpus, so ONE forward pass can serve
    # all of them: `embed(layers="last")` is a slice of `embed(layers="all")`, and per-token
    # log-probs come from the same forward that produces the hidden states. Declaring the
    # group lets ``ReusingAdapter`` compute the widest request once and slice the rest.
    #
    # This is opt-in metadata, not behavior: the reuse plan is built from the tasks in the
    # benchmark ACTUALLY being run, so a task run on its own promotes nothing, caches
    # nothing, and costs exactly what it did before. Leave ``fusion_group = None`` for any
    # task whose corpus is its own (e.g. rnagym-dms).
    fusion_group: str | None = None
    fusion_wants_all_layers: bool = False  # this task calls embed(layers="all")
    fusion_wants_logprob: bool = False  # this task calls logprob_embedding()

    @abstractmethod
    def prepare(self) -> None:
        """Load + validate the task's data. Must not touch the model."""

    @abstractmethod
    def run(self, adapter: ModelAdapter) -> None:
        """Request everything this task must score from *adapter* in batched calls."""

    @abstractmethod
    def score(self) -> TaskResult:
        """Reduce the requested data to a :class:`TaskResult` on CPU."""

    def evaluate(self, adapter: ModelAdapter) -> TaskResult:
        """Negotiate capabilities (with ordered fallbacks), then run + score.

        Negotiation selects the **first satisfied** capability set among
        ``(required_capabilities, *capability_fallbacks)`` and stores it on the instance
        (``_chosen_capabilities``) **before** ``prepare``/``run`` so ``run`` can branch on
        which path was chosen. On success the chosen path is recorded in
        ``metadata["scoring_method"]`` (provenance — threads to the result JSON + CSV).

        If **no** alternative is satisfied the task is ``NA``: ``missing_capabilities``
        keeps the primary's missing set, and when fallbacks were declared
        ``missing_capabilities_by_alternative`` lists what *each* tried path was missing.
        A failure inside ``run``/``score`` is captured as ``ERROR`` so one bad task never
        kills the benchmark.
        """
        adapter_caps = adapter.capabilities()
        candidates = (self.required_capabilities, *self.capability_fallbacks)
        chosen: frozenset[Capability] | None = None
        missing_by_alt: list[list[str]] = []
        for cap_set in candidates:
            missing_set = sorted(c.value for c in (cap_set - adapter_caps))
            missing_by_alt.append(missing_set)
            if not missing_set:
                chosen = cap_set
                break

        if chosen is None:
            metadata: dict[str, Any] = {"missing_capabilities": missing_by_alt[0]}
            if self.capability_fallbacks:
                metadata["missing_capabilities_by_alternative"] = missing_by_alt
            return TaskResult(self.name, ResultStatus.NA, None, {}, metadata)

        self._chosen_capabilities = chosen
        scoring_method = _scoring_method_name(chosen)
        self.prepare()
        try:
            self.run(adapter)
            result = self.score()
            result.metadata.setdefault("scoring_method", scoring_method)
            return result
        except Exception as e:  # noqa: BLE001 - intentional: capture, don't crash the suite
            return TaskResult(
                self.name,
                ResultStatus.ERROR,
                None,
                {},
                {"error": repr(e), "scoring_method": scoring_method},
            )


def _scoring_method_name(capabilities: frozenset[Capability]) -> str:
    """A stable provenance string naming the chosen capability path.

    A single-capability set (the common case — tasks with no fallback set this to their
    one capability) renders as that capability's value (e.g. ``"sequence_loglikelihood"``,
    ``"masked_marginal_llr"``); a multi-capability set joins the sorted values with ``+``.
    """
    return "+".join(sorted(c.value for c in capabilities))
