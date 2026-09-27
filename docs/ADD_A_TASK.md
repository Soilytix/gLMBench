# Adding a task

A task is a localized change: **drop one module into `src/glmbench/tasks/`, register
it, add one import line**. Nothing in the runner, the adapters, or the leaderboard
changes. This is composition over inheritance made concrete — the extensibility test in
`tests/unit/test_phase8.py` proves a brand-new task appears in `glmbench tasks list` and
on the board without touching the runner.

## The contract (`tasks/base.py`)

A task subclasses `Task` and fills in three lifecycle methods plus a few class
attributes:

```python
class Task(ABC):
    name: str                                  # registry key, e.g. "rnagym-dms"
    version: str                               # bump when the scoring changes
    required_capabilities: frozenset[Capability]
    higher_is_better: bool = True              # direction of the primary metric
    capability_fallbacks: tuple[frozenset[Capability], ...] = ()  # ordered alternatives

    def prepare(self) -> None: ...             # load + validate data; NO model
    def run(self, adapter: ModelAdapter) -> None: ...  # request data from the adapter
    def score(self) -> TaskResult: ...         # reduce to metrics on CPU
```

`Task.evaluate(adapter)` (provided by the base, do **not** override) drives them and
handles capability negotiation:

1. It picks the **first satisfied** capability set: it tries `required_capabilities`,
   then each set in `capability_fallbacks` in order. If none is a subset of
   `adapter.capabilities()` → returns a first-class
   `TaskResult(status=NA, metadata={"missing_capabilities": [...]})`. No exception, the
   benchmark continues, the leaderboard renders `N/A` in that cell. The chosen set is
   recorded as `metadata["scoring_method"]` — a stable provenance string (e.g.
   `"sequence_loglikelihood"` or `"masked_marginal_llr"`) the leaderboard surfaces as a
   CSV column + a per-task footnote whenever a fallback was used. `capability_fallbacks`
   defaults to `()`, which reproduces the prior single-set behavior exactly (strictly
   additive). A *scoring* (variant-effect) task should declare an ordered fallback to
   `MASKED_MARGINAL_LLR` so masked/encoder LMs without a causal likelihood still land a
   real row — `RnagymDmsTask` does this: primary `{SEQUENCE_LOGLIKELIHOOD}`, fallback
   `(frozenset({MASKED_MARGINAL_LLR}),)` (LLR and log-likelihood both reduce to
   abs-Spearman vs DMS, a method-agnostic rank metric).
2. Otherwise calls `prepare()`, then `run()` + `score()` inside a try; any exception
   becomes `TaskResult(status=ERROR, metadata={"error": ...})`.
   - **`prepare()` sits *outside* the try by design:** a data-integrity failure
     (e.g. a tampered sha256) raises loudly rather than being swallowed as a quiet
     `ERROR`. Put data loading/validation in `prepare()`.

`TaskResult` is flat and JSON-friendly:

```python
@dataclass
class TaskResult:
    task: str
    status: ResultStatus                 # OK | NA | ERROR
    primary_metric: str | None           # which key in `metrics` is THE headline number
    metrics: dict[str, float]            # flat; per-assay + macro live here
    metadata: dict                       # n scored, n missing, truncations, timings, reason
```

The leaderboard's **Overall** table ranks models by the mean of each task's
`primary_metric`; the **per-task** table renders every key in `metrics`. So put the
numbers you want to *compare across models* in `metrics`, and detail/provenance in
`metadata`.

## Recipe

1. **Copy `tasks/rnagym_dms.py`** (a real, batch-first task) or `tasks/dummy.py` (the
   minimal model-free template) to `tasks/<your_task>.py`.
2. **Set the class attributes:** `name`, `version`, `required_capabilities`, and
   `higher_is_better` if a lower primary metric is better.
   Declare *only* the capabilities you actually call — that is what drives the `N/A`
   gating against adapters that lack them. A *scoring* (variant-effect) task may also set
   `capability_fallbacks` — an ordered tuple of alternative capability sets tried when the
   primary is unmet; variant-effect tasks should fall back to `MASKED_MARGINAL_LLR` so
   masked/encoder LMs score instead of going `N/A`. The path actually taken is recorded as
   `metadata["scoring_method"]` provenance (rendered on the board).
3. **Implement `prepare()`** — load + validate your data. Keep a sha256/manifest
   integrity check here if the data is vendored (loud on drift).
4. **Implement `run(adapter)`** — collect **all** the sequences you need to score and
   make **one** batched adapter call (batch-first — never per-item round-trips;
   runner start-up is expensive):

   ```python
   def run(self, adapter):
       all_seqs = [s for assay in self._assays for s in assay.sequences]
       self._scores = adapter.score_sequences(all_seqs, reduction="mean")
   ```

   Honor the length/context contract: an adapter declares `max_context`
   (nucleotides); skip-and-count (with a loud `logger.warning`) any over-context
   sequence — never silently truncate.
5. **Implement `score()`** — reduce to a `TaskResult` on CPU. **Use the vendored metric
   authority, never hand-rolled math:** for RNAGym-family reductions call
   `tasks/metrics.py` (a thin wrapper over the verbatim upstream
   `_vendor/rnagym/performance_fitness.py`). Guard NaN/empty before it can poison
   a statistic.
6. **Register it** with the decorator and **add the import** so the module loads:

   ```python
   # in tasks/<your_task>.py
   from glmbench import registry

   @registry.register("task", "<your-task>")
   class YourTask(Task): ...
   ```

   ```python
   # in tasks/__init__.py — add to the side-effect import block (keep alphabetical)
   from . import your_task  # noqa: F401  (registers <your-task>)
   ```
7. **Put it on a benchmark.** Add the registry key to a `benchmarks/*.yaml` `tasks:`
   list (ordered), e.g.:

   ```yaml
   name: my_bench
   version: "1.0"
   tasks:
     - rnagym-dms
     - <your-task>
   ```

That's it. `glmbench tasks list` now shows it; `glmbench run --benchmark <bench>` runs
it against any adapter, returning `N/A` where the adapter lacks a required capability.

## Before you open a pull request

- [ ] `required_capabilities` declares exactly what `run()` calls; a scoring task adds an
      ordered `capability_fallbacks` (→ `MASKED_MARGINAL_LLR` for variant-effect) where it
      makes sense.
- [ ] `run()` makes **one** batched adapter call (no per-item round-trips).
- [ ] Data integrity/validation lives in `prepare()` (outside `evaluate`'s try).
- [ ] `score()` uses the vendored metric authority, not re-derived math.
- [ ] NaN/empty guarded before any statistic.
- [ ] `metrics` holds the comparable numbers; `primary_metric` names the headline.
- [ ] Registered + imported in `tasks/__init__.py`; on a `benchmarks/*.yaml`.
- [ ] **Best-of-sweep numbers are compared only with best-of-sweep numbers.** A max over
      29 layers has had 29 chances to top a column and a fixed-tap number has had one, so
      mixing them systematically favours the swept models. Sweep every model, or compare at
      one fixed tap (the last layer).
- [ ] **A k-mer floor is MEASURED before the task can headline**
      (`python scripts/kmer_floor.py --benchmark <bench> --task <your-task>`, which runs the
      real task code over a model-free composition vector so the number lands on the same
      axis as any leaderboard cell). Nothing should headline below its floor, and a task whose
      floor sits near the models' scores is testing composition, not learned representation.
      If your task reads the LM head rather than embeddings, a frequency vector is not a
      likelihood model and the honest answer is "not applicable, and here is why" — the
      script records that state rather than omitting the task.
- [ ] A **layer-sweep** task additionally: attaches `sweep_diagnostics(...)` to its metadata
      and selects `best_layer` via `tasks/_depth.choose_best_layer`, so the depth-health
      curves come off the array it already built (free — no second forward pass) and a
      numerically empty tap can never win.
- [ ] No edits to `run.py`, any adapter, or the leaderboard.
