"""RNAGym upstream, vendored verbatim — the metric authority.

``performance_fitness.py`` is a byte-identical copy of upstream
``MarksLab-DasLab/RNAGym``'s ``performance_fitness.py``. It defines the exact
per-assay reductions (abs-Spearman, median-split symmetric AUC, abs-MCC) and the
**type-mean** macro that the benchmark is defined by. ``glmbench.tasks.metrics`` is
a thin wrapper that *calls* these functions and contains no statistical math of its
own. See ``NOTICE`` for provenance + license and ``SHA256SUMS`` for the integrity
pin (a unit test fails loudly if the "verbatim" copy is ever edited).
"""
