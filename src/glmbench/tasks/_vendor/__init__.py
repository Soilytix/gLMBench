"""Upstream code vendored VERBATIM (the benchmark authority).

Modules under this package are byte-identical copies of upstream sources. They are
the *source of truth* for their domain (e.g. RNAGym's metric reductions): gLMBench
calls them through thin wrappers (``glmbench.tasks.metrics``) rather than
re-deriving their math, so a leaderboard cell can never drift from the benchmark's
own published definition. **Do not edit a vendored file** — re-vendoring to a newer
upstream commit is the only sanctioned change, and it bumps the benchmark version.
"""
