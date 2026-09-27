"""Is the top of this stack alive?

**The failure this exists to catch.** A deep pre-LN transformer can train part of its stack
into exact identity functions (the "curse of depth", arXiv:2502.05795) with every training
guard green. The loss cannot notice: the model's final RMSNorm makes the output exactly
scale-invariant, so residual-stream growth is a free direction with zero loss gradient
opposing it, and a dead layer is invisible to the global gradient norm because the healthy
layers dominate it. It takes a direct activation probe to see it.

Downstream, on a benchmark, a dead tap shows up as an *inflated-looking* sweep: Evo 2's
MLP-branch taps ``blocks.30.mlp.l3``/``blocks.31.mlp.l3`` sit at RMS 7e−17 / 9e−16 —
numerically empty — and a best-layer ``max`` over a sweep will happily crown one of them on
the strength of noise. The identical computation scored 0.142, 0.3005 and −0.022 on three
passes.

**Three numbers per layer**, pooled over the task's corpus, computed on the
:class:`~glmbench.adapters.base.EmbeddingResult` a sweep has **already produced** — no second
forward, no second corpus:

* ``rms[i]``   = RMS(x_i)                      — pooled residual magnitude
* ``cos[i]``   = cos(x_i, x_{i-1})             — inter-layer cosine, adjacent
* ``delta[i]`` = RMS(x_i − x_{i-1}) / RMS(x_i) — relative update size

``delta`` is the relative update a training-time dead-layer probe watches (its minimum over
layers, ``min_L RMS(Δx_L)/RMS(x_L)``, is reported as ``ratio_min``), judged against the same
reference: the compute dtype's unit roundoff.

**Two verdicts, and they are different claims.** A layer is **frozen** when ``delta[i]`` is at
or below the dtype's unit roundoff: the block ran and changed nothing a number in that dtype
can represent. A tap is **degenerate** when ``rms[i]`` is ~0: there is no signal there at all,
and whatever a probe scores off it is noise. A frozen layer still carries the *previous*
layer's information; a degenerate one carries none.

Pure numpy, torch-free — it runs in the core process, off the arrays a sweep already holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Unit roundoff (machine epsilon / 2) per compute dtype — the reference ``delta`` is judged
#: against. A relative update at or below this is a change the dtype cannot represent: the
#: block ran and the stream did not move.
_UNIT_ROUNDOFF: dict[str, float] = {
    # bfloat16: 8 significant bits ⇒ eps = 2^-7, unit roundoff 2^-8.
    "bfloat16": 2.0**-8,
    "bf16": 2.0**-8,
    # float16: 11 significant bits.
    "float16": 2.0**-11,
    "fp16": 2.0**-11,
    "half": 2.0**-11,
    "float32": 2.0**-24,
    "fp32": 2.0**-24,
    "float64": 2.0**-53,
}

#: A tap whose pooled RMS is at or below this is numerically empty — whatever a probe scores
#: off it is float noise. **Absolute, and that choice is measured, not assumed.**
#:
#: The obvious design is a *relative* cut (RMS small compared to the stack's largest layer).
#: It was implemented, run against stored sweep embeddings, and it is WRONG here — wrong in
#: the worst direction, on exactly the models the diagnostic exists for. Evo 1.5's residual
#: stream explodes to 3.3e7 in its upper half, so its *healthy* early layers sit many orders
#: of magnitude below the max and a 1e-6 relative cut condemns 11 live layers. The pathology
#: poisons its own reference.
#:
#: A per-tap "does it vary with the input" test was also measured and does not separate: Evo
#: 2's two dead MLP-branch taps have across-sequence relative std 0.18 / 0.13, right among
#: the healthy layers — they are float noise, and noise varies.
#:
#: What the measurement does show is a clean, enormous gap. Across six models and 134 taps,
#: every live tap sits at ≥ 3.4e-3 and the two known-dead ones at 7.2e-17 and 9.5e-16: nine
#: orders of magnitude with nothing in between. 1e-12 sits in that void with ~9 orders of
#: margin on the live side and ~4 on the dead side.
DEGENERATE_RMS_ABS = 1e-12

#: Kept as a knob at ``0.0`` (disabled) rather than deleted, so that turning the relative
#: test back on is a deliberate act with the note above attached to it.
DEGENERATE_RMS_REL = 0.0


def unit_roundoff(dtype: str | None) -> float:
    """Unit roundoff for a compute dtype name, defaulting to bf16 (the usual inference dtype).

    Defaulting to the **coarsest** plausible dtype is the conservative direction for a
    *positive* diagnostic: bf16's 2^-8 is a larger threshold than fp32's 2^-24, so an
    unlabelled run is more likely to have a layer called frozen than to have a frozen layer
    missed. A diagnostic that under-reports is one nobody finds out about.
    """
    if dtype is None:
        return _UNIT_ROUNDOFF["bfloat16"]
    return _UNIT_ROUNDOFF.get(str(dtype).lower(), _UNIT_ROUNDOFF["bfloat16"])


@dataclass
class DepthHealth:
    """Per-layer depth-health record for one model × one corpus."""

    layers: list[int]
    rms: list[float]
    rel_std: list[float]  # across-sequence variation / rms — context only, no verdict
    cos: list[float | None]  # None at the first layer — no predecessor to compare against
    delta: list[float | None]
    frozen_layers: list[int]
    degenerate_layers: list[int]
    dtype: str
    unit_roundoff: float
    degenerate_rms_rel: float
    degenerate_rms_abs: float
    max_rms: float
    n_sequences: int
    ratio_min: float | None = None  # min over layers of delta
    ratio_min_layer: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "rms": self.rms,
            "rel_std": self.rel_std,
            "cos": self.cos,
            "delta": self.delta,
            "frozen_layers": self.frozen_layers,
            "degenerate_layers": self.degenerate_layers,
            "dtype": self.dtype,
            "unit_roundoff": self.unit_roundoff,
            "degenerate_rms_rel": self.degenerate_rms_rel,
            "degenerate_rms_abs": self.degenerate_rms_abs,
            "max_rms": self.max_rms,
            "n_sequences": self.n_sequences,
            "ratio_min": self.ratio_min,
            "ratio_min_layer": self.ratio_min_layer,
            "notes": self.notes,
        }

    @property
    def healthy(self) -> bool:
        """Every layer both moved the stream and carries signal."""
        return not self.frozen_layers and not self.degenerate_layers


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


#: Rows sampled for the pooled statistics. A sweep's stack runs to tens of GB in float32 at
#: ``layers="all"`` on a 33-layer model; casting all of it to float64 to compute three pooled
#: scalars per layer would double that for no gain. These are corpus-level averages, so a few
#: thousand rows fixes them to far more precision than any threshold here needs. The sample is
#: deterministic (evenly spaced, never random) so the diagnostic is reproducible run to run.
MAX_SEQUENCES = 4096


def depth_health(
    per_seq: np.ndarray,
    layers: list[int] | np.ndarray,
    *,
    dtype: str | None = None,
    max_sequences: int | None = MAX_SEQUENCES,
    degenerate_rms_rel: float = DEGENERATE_RMS_REL,
    degenerate_rms_abs: float = DEGENERATE_RMS_ABS,
) -> DepthHealth:
    """Compute the three depth-health curves from a sweep's pooled embeddings.

    Args:
        per_seq: ``[n_sequences, n_layers, dim]`` — exactly the array a layer-sweep task
            already holds (``EmbeddingResult.arrays`` stacked). Pooled over tokens; this
            function pools over sequences.
        layers: the layer ids that array's middle axis corresponds to, echoed back. They are
            **not assumed contiguous**: a restricted sweep (``GLMBENCH_*_SWEEP_LAYERS="6-12"``)
            gives non-adjacent taps, and ``delta`` between two taps eight blocks apart is a
            different quantity from ``delta`` between neighbours. Gaps are recorded in
            ``notes`` rather than silently averaged over.
        dtype: the model's compute dtype, which sets the frozen threshold. See
            :func:`unit_roundoff` for why the default is the coarse one.

    Returns:
        A :class:`DepthHealth`. ``cos[0]`` and ``delta[0]`` are ``None`` — the first tap has
        no predecessor, and reporting 0.0 there would read as "frozen".
    """
    arr = np.asarray(per_seq)
    if arr.ndim != 3:
        raise ValueError(
            f"depth_health expects [n_sequences, n_layers, dim]; got shape {arr.shape}. "
            "Pool over tokens first — a per-token array would make `rms` a different "
            "quantity from the one the threshold is calibrated against."
        )
    ids = [int(i) for i in np.asarray(layers).ravel()]
    if len(ids) != arr.shape[1]:
        raise ValueError(
            f"layers has {len(ids)} ids but the array has {arr.shape[1]} layers — refusing "
            "to guess the pairing."
        )

    n_total = int(arr.shape[0])
    notes: list[str] = []
    if max_sequences is not None and n_total > max_sequences:
        # Evenly spaced, not random: a diagnostic that moves between two runs on the same
        # stored embeddings is one nobody can act on.
        rows = np.linspace(0, n_total - 1, max_sequences).astype(np.int64)
        arr = arr[rows]
        notes.append(
            f"pooled over {max_sequences} of {n_total} sequences (evenly spaced, "
            "deterministic) — these are corpus-level averages and the full stack is too "
            "large to promote to float64."
        )
    n_seq = int(arr.shape[0])

    def _layer(i: int) -> np.ndarray:
        """One layer as float64 — the only thing promoted, so peak memory is [n_seq, dim]."""
        return np.asarray(arr[:, i, :], dtype=np.float64)

    gaps = [
        (ids[i - 1], ids[i]) for i in range(1, len(ids)) if ids[i] - ids[i - 1] != 1
    ]
    if gaps:
        notes.append(
            "adjacent-layer deltas span gaps in this sweep "
            f"({gaps}) — `delta` there measures the change across several blocks, not one, "
            "so it is an UPPER bound on any single block's update and cannot be read as "
            "'this block is alive'."
        )

    n_layers = int(arr.shape[1])
    rms: list[float] = []
    rel_std: list[float] = []
    cos: list[float | None] = [None]
    delta: list[float | None] = [None]

    prev: np.ndarray | None = None
    for i in range(n_layers):
        cur = _layer(i)
        r = float(np.sqrt(np.mean(cur**2)))
        rms.append(r)
        # How much of this tap's magnitude actually varies with the input. Reported as
        # context, with NO verdict attached: measured on stored sweeps it does not separate
        # live taps from dead ones (Evo 2's two dead MLP taps sit at 0.18/0.13, right among
        # the healthy ones). It is here because "this tap returns nearly the same vector for
        # every sequence" is a real and distinct way for a tap to be useless, and the curve
        # is free off the same array.
        rel_std.append(
            float(np.sqrt(np.mean((cur - cur.mean(axis=0, keepdims=True)) ** 2)) / r)
            if r > 0
            else 0.0
        )
        if prev is not None:
            # Per-sequence cosine, then mean: a corpus-wide flattened cosine would be
            # dominated by whichever sequences have the largest norm.
            num = np.sum(prev * cur, axis=1)
            den = np.linalg.norm(prev, axis=1) * np.linalg.norm(cur, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per = np.where(den > 0, num / den, np.nan)
            cos.append(_finite(np.nanmean(per)) if np.isfinite(per).any() else None)
            d_rms = float(np.sqrt(np.mean((cur - prev) ** 2)))
            delta.append(_finite(d_rms / r) if r > 0 else None)
        prev = cur
    max_rms = max(rms) if rms else 0.0

    ur = unit_roundoff(dtype)
    # Frozen: the block ran and moved the stream by less than the dtype can represent.
    frozen = [ids[i] for i, d in enumerate(delta) if d is not None and d <= ur]
    # Degenerate: the tap is numerically empty. Absolute — see DEGENERATE_RMS_ABS for the
    # measurement that rules the relative test out. `degenerate_rms_rel` defaults to 0.0
    # (disabled) and is honoured only if a caller deliberately turns it on.
    degenerate = [
        ids[i]
        for i, r in enumerate(rms)
        if r <= degenerate_rms_abs
        or (degenerate_rms_rel > 0 and max_rms > 0 and r / max_rms <= degenerate_rms_rel)
    ]

    scored = [(ids[i], d) for i, d in enumerate(delta) if d is not None]
    ratio_min_layer, ratio_min = min(scored, key=lambda kv: kv[1]) if scored else (None, None)

    return DepthHealth(
        layers=ids,
        rms=rms,
        rel_std=rel_std,
        cos=cos,
        delta=delta,
        frozen_layers=frozen,
        degenerate_layers=degenerate,
        dtype=str(dtype) if dtype else "bfloat16 (assumed)",
        unit_roundoff=ur,
        degenerate_rms_rel=degenerate_rms_rel,
        degenerate_rms_abs=degenerate_rms_abs,
        max_rms=max_rms,
        n_sequences=n_seq,
        ratio_min=ratio_min,
        ratio_min_layer=ratio_min_layer,
        notes=notes,
    )


def usable_layers(health: DepthHealth) -> list[int]:
    """The layer ids a sweep's ``best_layer`` may be chosen from.

    A **degenerate** tap is excluded: there is no signal in it, so any score it earns is
    noise, and a best-layer ``max`` would crown it on the strength of that noise — the
    identical computation on Evo 2's ``blocks.31.mlp.l3`` scored 0.142, 0.3005 and −0.022 on
    three passes. A **frozen** layer is NOT excluded: it carries the previous layer's
    representation faithfully, so its score is real, just redundant. Excluding it would
    silently rewrite what a sweep is allowed to find.

    Returns every layer when that would otherwise be empty — a stack with no usable tap is a
    fact for the caller to report loudly, not a reason to return nothing and crash.
    """
    bad = set(health.degenerate_layers)
    keep = [li for li in health.layers if li not in bad]
    return keep or list(health.layers)
