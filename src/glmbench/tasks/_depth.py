"""What a layer sweep owes the reader beyond a winner.

Every sweep task already holds a ``[n_items, n_layers, dim]`` array immediately after its one
batched ``embed(layers="all")`` call. Two things fall out of it for free, and this module is
where both live so the sweep tasks agree rather than each growing its own version:

* :func:`measure_depth_health` — the per-layer rms/cos/delta curves, stashed in the task's
  result metadata. No second forward pass, no second corpus: the diagnostic is ~free off work
  the sweep is doing anyway.
* :func:`choose_best_layer` — ``best_layer`` selection that **cannot** crown a numerically
  empty tap. A plain ``max(score)`` would happily pick Evo 2's ``blocks.31``, whose
  pooled RMS is 9e−16 and whose score on three passes of the identical computation was 0.142,
  0.3005 and −0.022. That is not a layer winning; that is noise winning.

The selection is **loud about what it excluded**, in the metadata, so "this model's best
layer is 30, not 31" is never a silent correction.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from glmbench.diagnostics.depth_health import DepthHealth, depth_health, usable_layers


def measure_depth_health(
    stacked: np.ndarray, layer_ids: list[int], *, dtype: str | None = None
) -> DepthHealth | None:
    """Depth health off a sweep's own array, or ``None`` if it cannot be computed.

    Returns ``None`` rather than raising: this is an **observational** diagnostic (it reports
    loudly and never halts), so a shape it does not recognise must cost the run a diagnostic,
    never a result. The absence is itself recorded by the caller as the "never measured" state
    (``depth_health_measured: false``), which is not the same claim as "healthy".
    """
    try:
        return depth_health(stacked, layer_ids, dtype=dtype)
    except (ValueError, TypeError, MemoryError):
        return None


def choose_best_layer(
    layer_ids: list[int],
    score_of: Callable[[int], float],
    health: DepthHealth | None,
) -> tuple[int, dict[str, Any]]:
    """Pick the winning layer, excluding degenerate taps. Returns ``(best, why)``.

    ``score_of`` returns the layer's primary metric; NaN is treated as ``-inf`` (a layer that
    could not be scored never wins), matching what the sweep tasks already did.

    Only **degenerate** taps are excluded, never **frozen** ones. A frozen layer carries its
    predecessor's representation faithfully — its score is real, just redundant — and dropping
    it would silently narrow what a sweep is allowed to find. A degenerate tap carries
    nothing, so its score is an artifact of float noise.
    """
    def _s(layer_id: int) -> float:
        v = score_of(layer_id)
        return v if isinstance(v, (int, float)) and not math.isnan(v) else float("-inf")

    allowed = usable_layers(health) if health is not None else list(layer_ids)
    excluded = [li for li in layer_ids if li not in set(allowed)]
    best = max(allowed, key=_s)

    why: dict[str, Any] = {
        "considered_layers": list(allowed),
        "excluded_layers": excluded,
        "exclusion_rule": (
            "a tap whose pooled residual RMS is at or below the declared degeneracy floor "
            "carries no signal, so any score it earns is float noise"
        )
        if excluded
        else None,
    }
    if excluded and health is not None:
        why["excluded_rms"] = {
            str(li): health.rms[health.layers.index(li)] for li in excluded
        }
        naive = max(layer_ids, key=_s)
        if naive != best:
            # The case the gate exists for. Say it out loud in the record: a reader comparing
            # this run to an older one must be able to see WHY the winner moved.
            why["would_have_picked"] = int(naive)
            why["overridden_because"] = (
                f"layer {naive} scored highest but is degenerate (pooled RMS "
                f"{health.rms[health.layers.index(naive)]:.3g}); the best NON-degenerate tap "
                f"is layer {best}"
            )
    return int(best), why


def sweep_diagnostics(
    health: DepthHealth | None, selection: dict[str, Any]
) -> dict[str, Any]:
    """The metadata block every sweep task attaches.

    ``depth_health_measured`` keeps "never measured" distinct from "healthy": a missing
    measurement is not a healthy stack.
    """
    return {
        "depth_health": health.to_dict() if health is not None else None,
        "depth_health_measured": health is not None,
        "best_layer_selection": selection,
    }
