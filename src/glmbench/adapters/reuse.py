"""Cross-task forward-pass reuse — one forward serves every task that reads the same corpus.

**The waste this removes.** A single-layer task and its layer sweep read the *identical*
window list, and each would pay its own full forward pass over it:

    bacbench-essentiality              embed(layers="last")   164,630 windows
    bacbench-essentiality-layer-sweep  embed(layers="all")    the SAME 164,630 windows

and `"last"` is *literally a row of* `"all"` — the base task's output is a slice of what the
sweep already computed. Measured on a 17-layer model: embedding all 17 layers instead of 1
costs **+1.7%**, so running the base task separately would repeat the whole embedding pass
for a tensor slice the sweep already had in hand. The same relation holds for the EC pair.

**How this fixes it.** `ReusingAdapter` wraps a real adapter and, for each *fusion group* (a
set of tasks known to read one corpus), computes the **widest** thing any task in that group
will ask for — all layers, plus per-token log-probs if a logprob task is coming — in ONE call,
then serves the narrower requests by slicing. Later tasks in the group do no GPU work at all.

**Independent task runs are unaffected — by construction.** The promotion is driven by a
*plan* built from the tasks actually being run (`plan_fusion`). Run `bacbench-essentiality`
alone and its group contains no all-layer or logprob consumer, so nothing is promoted, nothing
is cached, and the adapter behaves byte-for-byte as it did before. Reuse only ever kicks in
when the redundant work was genuinely going to happen anyway.

**Safety.** The cache is keyed on a hash of the *actual sequences*, not on the group name. If
two tasks in one group ever disagree about their corpus, the hash differs, the lookup misses,
and the second task computes its own forward — a silent wrong answer is not reachable through
a stale-cache path. Entries are refcounted by the plan and dropped once their last consumer
has run, so the big all-layer array does not outlive its group.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from glmbench.adapters.base import Capability, EmbeddingResult, ModelAdapter

logger = logging.getLogger(__name__)


def ckey_not_cached(cache: dict[str, Any], key: str) -> bool:
    """True when we have not already banked log-probs for this corpus."""
    return key not in cache


def corpus_key(sequences: list[str]) -> str:
    """Stable content hash of a corpus — the cache identity, and the safety net.

    Hashing the sequences (rather than trusting the group label) is what makes a stale hit
    impossible: if two tasks in a group ever read different windows, they simply miss.
    """
    h = hashlib.sha256()
    h.update(str(len(sequences)).encode())
    for s in sequences:
        h.update(b"\x00")
        h.update(s.encode())
    return h.hexdigest()[:24]


@dataclass
class GroupPlan:
    """What a fusion group's tasks will collectively ask the model for."""

    name: str
    needs_all_layers: bool  # some task wants layers="all" → compute the superset once
    needs_logprob: bool  # some task wants per-token log-probs → get them in the same pass
    n_embed_consumers: int  # how many tasks will call embed() (for refcounted eviction)
    n_logprob_consumers: int


def plan_fusion(task_names: list[str], registry: Any) -> dict[str, GroupPlan]:
    """Build the per-group plan from the tasks ACTUALLY being run.

    This is what keeps independent runs honest: a group's plan reflects only the tasks in
    *this* benchmark. One task on its own ⇒ no all-layer promotion, no logprob fusion, no
    cache — identical behavior and identical cost to before this module existed.
    """
    groups: dict[str, GroupPlan] = {}
    for name in task_names:
        try:
            cls = registry.resolve("task", name)
        except Exception:  # noqa: BLE001 — an unknown task is run.py's problem, not ours
            continue
        group = getattr(cls, "fusion_group", None)
        if not group:
            continue
        plan = groups.setdefault(
            group,
            GroupPlan(group, False, False, 0, 0),
        )
        if getattr(cls, "fusion_wants_logprob", False):
            plan.needs_logprob = True
            plan.n_logprob_consumers += 1
        else:
            plan.n_embed_consumers += 1
            if getattr(cls, "fusion_wants_all_layers", False):
                plan.needs_all_layers = True
    return groups


class ReusingAdapter(ModelAdapter):
    """Adapter decorator that serves repeat forwards over one corpus from the first one.

    Transparent: every method not overridden here delegates to the wrapped adapter, so tasks
    cannot tell the difference apart from being faster.
    """

    def __init__(self, inner: ModelAdapter, plans: dict[str, GroupPlan]) -> None:
        self._inner = inner
        self._plans = plans
        self._group: str | None = None  # set by run.py before each task
        # (corpus_key, pool) -> the widest EmbeddingResult computed for that corpus
        self._embed_cache: dict[tuple[str, str], EmbeddingResult] = {}
        self._logprob_cache: dict[str, list[np.ndarray]] = {}
        self._embed_left: dict[str, int] = {}  # refcounts, for eviction
        self._logprob_left: dict[str, int] = {}
        self.reuse_stats = {"embed_hits": 0, "embed_misses": 0,
                            "logprob_hits": 0, "logprob_misses": 0}

    # --- run.py drives this ------------------------------------------------

    def begin_task(self, group: str | None) -> None:
        """Tell the wrapper which fusion group the next task belongs to (None ⇒ no reuse)."""
        self._group = group

    def _plan(self) -> GroupPlan | None:
        return self._plans.get(self._group) if self._group else None

    # --- transparent delegation -------------------------------------------

    def __getattr__(self, item: str) -> Any:
        # Only called for attributes NOT found on this class — i.e. everything we don't fuse.
        return getattr(self._inner, item)

    # `name` and `adapter_version` MUST be forwarded explicitly. They are declared as class
    # attributes on ModelAdapter (defaulting to ""), so normal attribute lookup finds those
    # empty strings on this subclass and __getattr__ NEVER fires. That shipped: the first
    # fused run wrote `adapter_name: ""` / `adapter_version: ""` into its result record,
    # which left the row unattributable — a silent provenance corruption, since the
    # model_hash (an explicit method, correctly delegated) was still right.
    @property
    def name(self) -> str:  # type: ignore[override]
        return self._inner.name

    @property
    def adapter_version(self) -> str:  # type: ignore[override]
        return self._inner.adapter_version

    # The readout declaration walks into the SAME trap, and it is the trap's worst case:
    # `readout` defaults to None on ModelAdapter, so ordinary lookup finds that None on this
    # subclass and `__getattr__` never fires. A fused run would then record `readout: null`
    # for a model that declares one — which reads as "never measured", a different claim
    # from "measured", on a record that is actually fine.
    @property
    def readout(self):  # type: ignore[override]
        return self._inner.readout

    @property
    def readout_note(self) -> str | None:  # type: ignore[override]
        return self._inner.readout_note

    @property
    def readout_evidence(self) -> str:  # type: ignore[override]
        return self._inner.readout_evidence

    def readout_declaration(self) -> dict[str, Any]:
        return self._inner.readout_declaration()

    def capabilities(self) -> frozenset[Capability]:
        return self._inner.capabilities()

    def model_hash(self) -> str:
        return self._inner.model_hash()

    def describe(self) -> dict[str, Any]:
        return self._inner.describe()

    def tokenize(self, sequences: list[str]) -> list[list[int]]:
        return self._inner.tokenize(sequences)

    def score_sequences(self, sequences: list[str], **kw: Any) -> np.ndarray:
        return self._inner.score_sequences(sequences, **kw)

    # Same trap as `name`/`adapter_version` above, one layer deeper: EVERY capability method
    # is declared on ModelAdapter with a fail-closed body that raises CapabilityNotImplemented,
    # so an un-forwarded one is found by ordinary lookup on this subclass and `__getattr__`
    # NEVER fires — the wrapper reports the INNER adapter's name as not implementing a
    # capability the inner adapter implements and declares. That shipped: gLM2-650M's
    # `rnagym-dms` errored with `CapabilityNotImplemented("Adapter 'glm2' does not implement
    # 'masked_marginal_llr'")` in a full-benchmark run, while the same task on the same model
    # passed standalone (one task ⇒ no fusion plan ⇒ no wrapper). The wrapper is installed
    # whenever ANY pending task has a fusion_group, so it intercepts unfused tasks too.
    # Rule: a new Capability on ModelAdapter needs a forwarder HERE. Gated by
    # test_gate_every_capability_method_is_forwarded.

    def per_token_logprobs(self, sequences: list[str]) -> list[np.ndarray]:
        return self._inner.per_token_logprobs(sequences)

    def score_variant_llr(self, items: list[dict[str, Any]]) -> list[float]:
        return self._inner.score_variant_llr(items)

    # --- the fused paths ---------------------------------------------------

    def embed(
        self,
        sequences: list[str],
        *,
        layers: str | list[int] | None = "last",
        pool: str = "mean",
    ) -> EmbeddingResult:
        plan = self._plan()
        if plan is None:
            return self._inner.embed(sequences, layers=layers, pool=pool)

        key = (corpus_key(sequences), pool)
        cached = self._embed_cache.get(key)
        if cached is not None:
            self.reuse_stats["embed_hits"] += 1
            logger.info(
                "reuse: serving embed(layers=%r) for %d sequences from the cached forward "
                "pass of group %r — no GPU work",
                layers, len(sequences), plan.name,
            )
            out = _select_layers(cached, layers)
            self._release_embed(key, plan)
            return out

        self.reuse_stats["embed_misses"] += 1
        # Promote to the widest request this GROUP will make, so the later tasks are free.
        # Costs ~1.7% over a single layer; saves a whole forward pass over the same corpus.
        want = "all" if plan.needs_all_layers else layers
        if plan.needs_all_layers and layers != "all":
            logger.info(
                "reuse: promoting embed to layers='all' for group %r (%d sequences) — a later "
                "task in this group needs every layer, and 'last' is a slice of 'all'",
                plan.name, len(sequences),
            )

        # If a logprob task is also coming for this corpus, take BOTH heads off one forward.
        fused = (
            plan.needs_logprob
            and plan.n_logprob_consumers > 0
            and ckey_not_cached(self._logprob_cache, key[0])
            and hasattr(self._inner, "embed_with_logprob")
            and Capability.LOGPROB_EMBEDDING in self._inner.capabilities()
        )
        if fused:
            logger.info(
                "reuse: fusing embed + logprob_embedding for group %r (%d sequences) into ONE "
                "forward pass — they read two heads of the same computation",
                plan.name, len(sequences),
            )
            full, zs = self._inner.embed_with_logprob(sequences, layers=want, pool=pool)
            self._logprob_cache[key[0]] = zs
        else:
            full = self._inner.embed(sequences, layers=want, pool=pool)

        # Only cache if someone else is actually coming for it (else we'd pin a large array).
        if plan.n_embed_consumers > 1:
            self._embed_cache[key] = full
            self._embed_left[key[0]] = plan.n_embed_consumers - 1
        return _select_layers(full, layers)

    def logprob_embedding(self, sequences: list[str]) -> list[np.ndarray]:
        plan = self._plan()
        if plan is None:
            return self._inner.logprob_embedding(sequences)

        ckey = corpus_key(sequences)
        hit = self._logprob_cache.pop(ckey, None)
        if hit is not None:
            self.reuse_stats["logprob_hits"] += 1
            logger.info(
                "reuse: serving logprob_embedding for %d sequences from the cached forward "
                "pass of group %r — no GPU work",
                len(sequences), plan.name,
            )
            return hit
        self.reuse_stats["logprob_misses"] += 1
        return self._inner.logprob_embedding(sequences)

    # --- eviction ----------------------------------------------------------

    def _release_embed(self, key: tuple[str, str], plan: GroupPlan) -> None:
        left = self._embed_left.get(key[0], 0) - 1
        self._embed_left[key[0]] = left
        if left <= 0:
            # Drop the (large) all-layer array as soon as its last consumer has been served.
            self._embed_cache.pop(key, None)
            self._embed_left.pop(key[0], None)
            logger.debug("reuse: evicted cached embedding for group %r", plan.name)


def _select_layers(res: EmbeddingResult, layers: str | list[int] | None) -> EmbeddingResult:
    """Return the requested layers as a view of an already-computed (wider) result.

    ``layers`` follows the layer convention and is resolved against ``res.layers`` — the layer
    IDs actually present. ``"last"`` is the final id, ``"all"`` is everything, a list indexes
    into the id list (negatives allowed). Raises if a requested layer was never computed,
    rather than quietly returning the wrong one.
    """
    have = list(res.layers)
    if layers == "all":
        want = have
    elif layers in ("last", None):
        want = [have[-1]]
    elif isinstance(layers, int):
        want = [have[layers]]
    else:
        want = [have[i] if i < 0 else i for i in layers]  # type: ignore[union-attr]

    missing = [li for li in want if li not in have]
    if missing:
        raise KeyError(
            f"cached forward pass has layers {have}, but layers {missing} were requested — "
            f"refusing to serve a different layer than the one asked for."
        )
    if want == have:
        return res

    idx = [have.index(li) for li in want]
    arrays = [a[idx] if a.ndim == 2 else a[:, idx] for a in res.arrays]
    return EmbeddingResult(
        arrays=arrays,
        layers=want,
        pool=res.pool,
        token_spans=res.token_spans,
        embedding_dim=res.embedding_dim,
    )
