"""Generic capability-fallback negotiation on the Task base.

The ``Task.evaluate`` driver selects the **first satisfied** capability set among
``(required_capabilities, *capability_fallbacks)``, stores it on the instance before
``run`` so ``run`` can branch, and records the chosen path in
``metadata["scoring_method"]``. A task with ``capability_fallbacks = ()`` is byte-for-byte
unchanged from before (same NA semantics, same metadata save the additive
``scoring_method`` on success).

Torch-free, model-free — uses the EchoAdapter with instance-configurable capabilities.
"""

from __future__ import annotations

from glmbench.adapters.base import Capability, ModelAdapter
from glmbench.adapters.echo import EchoAdapter
from glmbench.tasks.base import ResultStatus, Task, TaskResult


class _FallbackTask(Task):
    """A scoring task with a primary capability + one ordered fallback."""

    name = "fallback-probe"
    version = "1.0"
    required_capabilities = frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})
    capability_fallbacks = (frozenset({Capability.MASKED_MARGINAL_LLR}),)

    def __init__(self) -> None:
        self.chosen_during_run: frozenset[Capability] | None = None

    def prepare(self) -> None:
        pass

    def run(self, adapter: ModelAdapter) -> None:
        # _chosen_capabilities must already be set by evaluate.
        self.chosen_during_run = self._chosen_capabilities

    def score(self) -> TaskResult:
        return TaskResult(self.name, ResultStatus.OK, "x", {"x": 1.0}, {})


class _NoFallbackTask(Task):
    """An embedding-only task with NO fallback — back-compat reference."""

    name = "embed-only-probe"
    version = "1.0"
    required_capabilities = frozenset({Capability.EMBEDDING})

    def prepare(self) -> None:
        pass

    def run(self, adapter: ModelAdapter) -> None:
        pass

    def score(self) -> TaskResult:
        return TaskResult(self.name, ResultStatus.OK, "x", {"x": 1.0}, {})


def test_primary_present_selects_primary():
    task = _FallbackTask()
    result = task.evaluate(EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD}))
    assert result.status is ResultStatus.OK
    assert task.chosen_during_run == frozenset({Capability.SEQUENCE_LOGLIKELIHOOD})
    assert result.metadata["scoring_method"] == "sequence_loglikelihood"


def test_primary_absent_fallback_present_selects_fallback():
    task = _FallbackTask()
    result = task.evaluate(EchoAdapter(capabilities={Capability.MASKED_MARGINAL_LLR}))
    assert result.status is ResultStatus.OK
    assert task.chosen_during_run == frozenset({Capability.MASKED_MARGINAL_LLR})
    assert result.metadata["scoring_method"] == "masked_marginal_llr"


def test_neither_present_is_na_listing_all_alternatives():
    task = _FallbackTask()
    result = task.evaluate(EchoAdapter(capabilities={Capability.EMBEDDING}))
    assert result.status is ResultStatus.NA
    # primary's missing set kept as-is (back-compat), plus a per-alternative breakdown.
    assert result.metadata["missing_capabilities"] == ["sequence_loglikelihood"]
    assert result.metadata["missing_capabilities_by_alternative"] == [
        ["sequence_loglikelihood"],
        ["masked_marginal_llr"],
    ]
    assert "scoring_method" not in result.metadata


def test_no_fallback_task_na_is_byte_compatible():
    """Back-compat: a fallback-free task NA metadata is exactly today's shape."""
    task = _NoFallbackTask()
    result = task.evaluate(EchoAdapter(capabilities={Capability.SEQUENCE_LOGLIKELIHOOD}))
    assert result.status is ResultStatus.NA
    assert result.metadata == {"missing_capabilities": ["embedding"]}


def test_no_fallback_task_ok_adds_only_scoring_method():
    """Back-compat: on OK the only additive metadata is scoring_method."""
    task = _NoFallbackTask()
    result = task.evaluate(EchoAdapter(capabilities={Capability.EMBEDDING}))
    assert result.status is ResultStatus.OK
    assert result.metadata == {"scoring_method": "embedding"}


def test_error_path_records_scoring_method():
    class _Boom(_FallbackTask):
        def run(self, adapter: ModelAdapter) -> None:
            raise RuntimeError("boom")

    task = _Boom()
    result = task.evaluate(EchoAdapter(capabilities={Capability.MASKED_MARGINAL_LLR}))
    assert result.status is ResultStatus.ERROR
    assert result.metadata["scoring_method"] == "masked_marginal_llr"
    assert "boom" in result.metadata["error"]


# --- the wrapper must not swallow a negotiated capability -------------------


def test_gate_every_capability_method_is_forwarded_by_the_reuse_wrapper():
    """GATE: negotiation is worthless if the wrapper can't deliver the method it picked.

    ``ReusingAdapter`` subclasses ``ModelAdapter``, and every capability method on the base
    has a **fail-closed body** that raises ``CapabilityNotImplemented``. So an un-forwarded
    method is found by ordinary attribute lookup on the subclass, ``__getattr__`` never
    fires, and the wrapper raises — naming the INNER adapter as not implementing a
    capability the inner adapter implements *and declares*. Negotiation saw the capability
    (``capabilities()`` IS forwarded) and routed to it; the call then died.

    That shipped: gLM2-650M's ``rnagym-dms`` errored with ``CapabilityNotImplemented(
    "Adapter 'glm2' does not implement capability 'masked_marginal_llr'")`` in a full-suite
    run, while the same model+task passed standalone — the wrapper is installed whenever ANY
    pending task has a ``fusion_group``, so it also intercepts unfused tasks like rnagym-dms.
    ``per_token_logprobs`` had the same hole, unexercised only because no task calls it.

    Structural, so it holds for capabilities added later: nothing here is enumerated by hand.
    """
    from glmbench.adapters.base import CAPABILITY_METHODS
    from glmbench.adapters.reuse import ReusingAdapter

    missing = [
        method
        for method in CAPABILITY_METHODS.values()
        if getattr(ReusingAdapter, method, None) is getattr(ModelAdapter, method, None)
    ]
    assert not missing, (
        f"ReusingAdapter does not forward {missing} — these fall through to ModelAdapter's "
        f"fail-closed body and will raise CapabilityNotImplemented against the wrapped "
        f"adapter's name. Add an explicit forwarder in adapters/reuse.py."
    )
