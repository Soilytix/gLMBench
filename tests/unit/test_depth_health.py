"""Depth health: is the top of this stack alive?

* synthetic **positive** — a stack whose upper blocks are exact identity functions is
  flagged frozen at exactly those layers
* synthetic **negative** — a healthy stack is not flagged (a one-sided probe is how a
  diagnostic gets ignored)
* degenerate taps are named with their measured RMS
* a degenerate tap can never win a sweep
* "never measured" stays distinct from "healthy"; a normal sweep run produces the diagnostic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glmbench.diagnostics import DEGENERATE_RMS_ABS, depth_health, unit_roundoff
from glmbench.diagnostics.depth_health import usable_layers
from glmbench.tasks._depth import choose_best_layer, measure_depth_health, sweep_diagnostics

REPO = Path(__file__).resolve().parents[2]

RNG = np.random.default_rng(0)


def _stack(n_seq: int, n_layers: int, dim: int, *, frozen_from: int | None = None) -> np.ndarray:
    """A synthetic residual stack: each block adds a real update, until `frozen_from`.

    From ``frozen_from`` on, the block is an EXACT identity — a dead upper stack, and the
    reason the loss cannot see it: the stream simply stops changing.
    """
    x = RNG.normal(size=(n_seq, dim))
    layers = [x.copy()]
    for i in range(1, n_layers):
        if frozen_from is not None and i >= frozen_from:
            layers.append(layers[-1].copy())  # identity block
        else:
            layers.append(layers[-1] + 0.35 * RNG.normal(size=(n_seq, dim)))
    return np.stack(layers, axis=1)


# ---------------------------------------------------------------------------------------
# The probe fires on frozen, stays quiet on healthy.
# ---------------------------------------------------------------------------------------


def test_gate_synthetic_positive_identity_blocks_are_flagged_frozen() -> None:
    """Exactly the identity layers, and no others."""
    h = depth_health(_stack(64, 12, 32, frozen_from=8), list(range(12)), dtype="bfloat16")
    assert h.frozen_layers == [8, 9, 10, 11], (
        f"expected layers 8..11 frozen, got {h.frozen_layers}. Layer 8 is the first whose "
        f"update is identically zero (it is compared against layer 7)."
    )
    assert h.ratio_min == 0.0
    assert h.ratio_min_layer in h.frozen_layers
    assert not h.healthy
    # cos ≈ 1.0 across every frozen boundary — the other signature of a dead stack.
    for i, li in enumerate(h.layers):
        if li in h.frozen_layers:
            assert h.cos[i] is not None and h.cos[i] > 0.999999


def test_gate_synthetic_negative_a_healthy_stack_is_not_flagged() -> None:
    """One-sided probes get ignored. This is the other side."""
    h = depth_health(_stack(64, 12, 32), list(range(12)), dtype="bfloat16")
    assert h.frozen_layers == []
    assert h.degenerate_layers == []
    assert h.healthy
    assert h.ratio_min is not None and h.ratio_min > unit_roundoff("bfloat16")


def test_gate_a_small_but_real_update_is_not_frozen() -> None:
    """The boundary case: an update just ABOVE the dtype's unit roundoff is alive.

    A threshold that swallows small-but-real updates would condemn the deep end of every
    healthy model, and a diagnostic that cries wolf is one that gets switched off.
    """
    x = RNG.normal(size=(64, 32))
    ur = unit_roundoff("bfloat16")
    layers = [x]
    for _ in range(4):
        layers.append(layers[-1] + (ur * 10) * RNG.normal(size=(64, 32)))
    h = depth_health(np.stack(layers, axis=1), list(range(5)), dtype="bfloat16")
    assert h.frozen_layers == []


def test_frozen_threshold_follows_the_declared_dtype() -> None:
    """fp32's roundoff is 2^-24; an update bf16 calls frozen is alive in fp32."""
    x = RNG.normal(size=(32, 16))
    step = unit_roundoff("bfloat16") / 10  # below bf16's floor, far above fp32's
    stack = np.stack([x, x + step * RNG.normal(size=(32, 16))], axis=1)
    assert depth_health(stack, [0, 1], dtype="bfloat16").frozen_layers == [1]
    assert depth_health(stack, [0, 1], dtype="float32").frozen_layers == []


# ---------------------------------------------------------------------------------------
# Degenerate taps are named, with their measured RMS.
# ---------------------------------------------------------------------------------------


def test_gate_degenerate_taps_are_named_with_their_rms() -> None:
    """A numerically empty tap is reported as such, with the number."""
    stack = _stack(64, 6, 32)
    stack[:, 4, :] = 7.16e-17  # Evo 2's blocks.30, to the digit
    stack[:, 5, :] = 9.48e-16  # blocks.31
    h = depth_health(stack, list(range(6)), dtype="bfloat16")
    assert h.degenerate_layers == [4, 5]
    assert h.rms[4] < DEGENERATE_RMS_ABS and h.rms[5] < DEGENERATE_RMS_ABS
    assert h.to_dict()["degenerate_rms_abs"] == DEGENERATE_RMS_ABS  # the declared threshold
    assert not h.healthy


def test_a_live_but_small_tap_is_not_degenerate() -> None:
    """The measured live floor across the fleet is 3.4e-3 — nine orders above the cut."""
    stack = _stack(64, 4, 32) * 3.4e-3
    h = depth_health(stack, list(range(4)), dtype="bfloat16")
    assert h.degenerate_layers == []


def test_an_exploded_stack_does_not_condemn_its_own_healthy_layers() -> None:
    """The regression that killed the relative-threshold design.

    Evo 1.5 explodes to 3.3e7 in its upper half. A degeneracy test relative to the stack's
    max marks every healthy early layer degenerate — the pathology
    poisons its own reference. Pinned so nobody reintroduces it.
    """
    stack = _stack(64, 10, 32)
    stack[:, 6:, :] *= 1e8  # the upper stack blows up, as it really does
    h = depth_health(stack, list(range(10)), dtype="bfloat16")
    assert h.degenerate_layers == [], (
        "early layers of an exploded stack were called degenerate — that is the relative "
        "threshold coming back."
    )


# ---------------------------------------------------------------------------------------
# A degenerate tap can never win a sweep.
# ---------------------------------------------------------------------------------------


def test_gate_a_zero_rms_layer_with_the_top_score_is_excluded() -> None:
    """The runner-up wins, and the record says why.

    A plain `max` over scores would crown Evo 2's `blocks.31`, whose pooled RMS is 9e-16 and
    whose score does not even repeat across passes of the IDENTICAL computation.
    """
    stack = _stack(64, 5, 32)
    stack[:, 4, :] = 1e-18  # the degenerate tap…
    h = depth_health(stack, list(range(5)), dtype="bfloat16")
    scores = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.42, 4: 0.99}  # …carrying the max score

    best, why = choose_best_layer(list(range(5)), lambda li: scores[li], h)
    assert best == 3, "the degenerate tap won the sweep"
    assert why["excluded_layers"] == [4]
    assert why["would_have_picked"] == 4
    assert "degenerate" in why["overridden_because"]
    assert why["excluded_rms"]["4"] < DEGENERATE_RMS_ABS


def test_gate_a_frozen_layer_is_still_allowed_to_win() -> None:
    """Frozen ≠ degenerate. A frozen layer carries its predecessor's representation faithfully.

    Its score is real, just redundant. Excluding it would silently narrow what a sweep is
    permitted to find, which is a different (and unasked-for) change.
    """
    h = depth_health(_stack(64, 6, 32, frozen_from=4), list(range(6)), dtype="bfloat16")
    assert h.frozen_layers and not h.degenerate_layers
    best, why = choose_best_layer(list(range(6)), lambda li: float(li), h)
    assert best == 5
    assert why["excluded_layers"] == []


def test_gate_selection_is_unchanged_when_nothing_is_degenerate() -> None:
    """No health data, or a clean stack ⇒ byte-identical behaviour to plain `max`."""
    scores = {0: 0.1, 1: 0.9, 2: 0.4}
    assert choose_best_layer([0, 1, 2], lambda li: scores[li], None)[0] == 1
    h = depth_health(_stack(32, 3, 16), [0, 1, 2], dtype="bfloat16")
    assert choose_best_layer([0, 1, 2], lambda li: scores[li], h)[0] == 1


def test_gate_a_wholly_degenerate_stack_still_returns_a_layer() -> None:
    """No usable tap is a fact to report, not a reason to crash mid-suite."""
    stack = np.full((16, 4, 8), 1e-20)
    h = depth_health(stack, list(range(4)), dtype="bfloat16")
    assert h.degenerate_layers == [0, 1, 2, 3]
    assert usable_layers(h) == [0, 1, 2, 3]  # falls back to all, rather than to nothing
    best, _ = choose_best_layer([0, 1, 2, 3], lambda li: float(li), h)
    assert best == 3


def test_nan_scores_never_win() -> None:
    scores = {0: 0.5, 1: float("nan"), 2: 0.4}
    assert choose_best_layer([0, 1, 2], lambda li: scores[li], None)[0] == 0


# ---------------------------------------------------------------------------------------
# Shape / plumbing contracts.
# ---------------------------------------------------------------------------------------


def test_first_layer_has_no_predecessor_and_says_so() -> None:
    """`cos[0]`/`delta[0]` are None. Reporting 0.0 there would read as 'frozen'."""
    h = depth_health(_stack(16, 4, 8), list(range(4)), dtype="bfloat16")
    assert h.cos[0] is None and h.delta[0] is None
    assert 0 not in h.frozen_layers


def test_a_restricted_sweep_records_that_its_deltas_span_gaps() -> None:
    """`GLMBENCH_*_SWEEP_LAYERS="6-12"` gives non-adjacent taps.

    A delta across eight blocks is an UPPER bound on any single block's update, so it cannot
    be read as "this block is alive". Silence there would be a wrong claim, not a missing one.
    """
    h = depth_health(_stack(16, 3, 8), [6, 10, 14], dtype="bfloat16")
    assert h.notes and "gaps" in h.notes[0]


def test_layer_count_mismatch_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="refusing to guess"):
        depth_health(_stack(8, 3, 4), [0, 1], dtype="bfloat16")


def test_a_per_token_array_is_rejected() -> None:
    """A 4-D array would make `rms` a different quantity from the calibrated one."""
    with pytest.raises(ValueError, match="Pool over tokens"):
        depth_health(np.zeros((4, 5, 3, 8)), [0, 1, 2], dtype="bfloat16")


def test_measure_depth_health_degrades_to_none_rather_than_raising() -> None:
    """Observational: a shape it cannot read costs the run a diagnostic, never a result."""
    assert measure_depth_health(np.zeros((4, 3)), [0, 1, 2]) is None


def test_subsampling_is_deterministic_and_barely_moves_the_numbers() -> None:
    big = _stack(6000, 5, 16, frozen_from=3)
    a = depth_health(big, list(range(5)), dtype="bfloat16", max_sequences=1024)
    b = depth_health(big, list(range(5)), dtype="bfloat16", max_sequences=1024)
    full = depth_health(big, list(range(5)), dtype="bfloat16", max_sequences=None)
    assert a.to_dict() == b.to_dict(), "the sample must not move between runs"
    assert a.frozen_layers == full.frozen_layers
    assert a.notes and "evenly spaced" in a.notes[0]
    np.testing.assert_allclose(a.rms, full.rms, rtol=0.05)


def test_sweep_diagnostics_keeps_never_measured_distinct_from_healthy() -> None:
    """In the metadata block, absent is not the same claim as clean."""
    md = sweep_diagnostics(None, {})
    assert md["depth_health"] is None
    assert md["depth_health_measured"] is False
    h = depth_health(_stack(16, 3, 8), list(range(3)), dtype="bfloat16")
    md = sweep_diagnostics(h, {"considered_layers": [0, 1, 2], "excluded_layers": []})
    assert md["depth_health_measured"] is True
    assert md["depth_health"]["frozen_layers"] == []


# ---------------------------------------------------------------------------------------
# A normal sweep run produces the diagnostic FOR FREE, in every sweep task.
# ---------------------------------------------------------------------------------------


class _SweepAdapter:
    """A minimal embedding adapter whose upper blocks are exact identity functions.

    Not a mock of the diagnostic — a mock of a *sick model*. The sweep tasks must find the
    dead layers on their own, off the array they already built.
    """

    name = "depth-health-fixture"
    adapter_version = "0.1.0"

    def __init__(self, n_layers: int = 6, frozen_from: int = 4) -> None:
        from glmbench.adapters.base import Capability

        self.n_layers = n_layers
        self.frozen_from = frozen_from
        self.CAPABILITIES = frozenset({Capability.EMBEDDING})

    def capabilities(self):
        return self.CAPABILITIES

    def model_hash(self) -> str:
        return "glmb:depth-health-fixture"

    def describe(self) -> dict:
        return {"name": self.name, "embedding_dim": 8, "n_layers": self.n_layers}

    def embed(self, sequences, *, layers="last", pool="mean"):
        from glmbench.adapters.base import EmbeddingResult

        ids = list(range(self.n_layers + 1)) if layers == "all" else (
            [self.n_layers] if layers in ("last", None) else [int(i) for i in layers]
        )
        arrays = []
        for s in sequences:
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            base = rng.normal(size=8)
            rows = []
            for i in range(self.n_layers + 1):
                if i > 0:
                    base = base if i >= self.frozen_from else base + 0.4 * rng.normal(size=8)
                rows.append(base.copy())
            arrays.append(np.stack([rows[i] for i in ids], axis=0).astype(np.float32))
        return EmbeddingResult(
            arrays=arrays, layers=ids, pool=pool, token_spans=None, embedding_dim=8
        )


@pytest.mark.parametrize(
    ("module", "cls_name", "manifest"),
    [
        ("glmbench.tasks.dgeb_ec_classification", "DGEBECClassificationLayerSweepTask",
         "dgeb_ec_dna_tiny"),
        ("glmbench.tasks.bacbench_essentiality", "BacBenchEssentialityLayerSweepTask",
         "bacbench_essentiality_tiny"),
    ],
)
def test_a_normal_sweep_run_produces_the_diagnostic(module, cls_name, manifest) -> None:
    """No second pass, no second corpus — the curves come off the sweep's own array.

    This is also the gate that would have caught a missing import: a NameError inside a task
    is contained as an ERROR row, so a broken diagnostic does NOT crash the suite — it
    quietly turns one task red in a real run. Asserting the metadata is present is the only
    thing that notices.
    """
    import importlib

    task_cls = getattr(importlib.import_module(module), cls_name)
    path = REPO / "tests" / "data" / manifest / "tiny_manifest.yaml"
    result = task_cls(path).evaluate(_SweepAdapter())
    assert result.status.value == "ok", result.metadata.get("error")
    md = result.metadata
    assert md["depth_health_measured"] is True
    assert md["depth_health"]["frozen_layers"] == [4, 5, 6]
    assert "best_layer_selection" in md
    assert md["best_layer_selection"]["considered_layers"] == list(range(7))

