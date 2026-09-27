"""DGEB upstream, vendored as the probe + metric authority.

``evaluators.py`` carries the **torch-free evaluator subset** of upstream
``TattaBio/DGEB``'s ``dgeb/evaluators.py`` — two classes copied char-for-char in their
``__init__``/``__call__``:

* ``logRegClassificationEvaluator`` (sklearn ``LogisticRegression`` probe → macro-``f1`` +
  ``accuracy``), called by ``glmbench.tasks.dgeb_ec_metrics`` for the EC DNA task;
* ``EDSEvaluator`` (paired cosine/manhattan/euclidean distances vs. gold phylogenetic
  distances, correlated with ``scipy.stats.pearsonr`` → ``top_corr``), DGEB's evaluator for
  its phylogeny tasks; no shipped task calls it.

The *only* edit versus upstream is dropping the ``torch`` seeding from the ``Evaluator``
base (inert: the classification result is fixed by ``LogisticRegression(random_state=seed)``
and ``EDSEvaluator`` is fully deterministic) — this keeps the core torch-free.

The wrapper modules *call* these evaluators and contain no statistical math of their own.
See ``NOTICE`` for provenance + license and ``SHA256SUMS`` for the integrity pin (a unit
test fails loudly if this copy is edited).
"""
