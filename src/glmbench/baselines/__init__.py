"""Model-free baselines — the floors a language model has to clear to have earned its GPU.

A baseline here is an object that satisfies the *adapter* contract while containing no learned
parameters at all, so the REAL task code (real windowing, real probes, real vendored metrics)
runs over it unchanged and its score lands on exactly the same axis as any leaderboard cell.

Currently one: :class:`~glmbench.baselines.kmer.KmerCompositionAdapter`, whose "embedding" is a
normalized k-mer frequency vector. See ``scripts/kmer_floor.py``.
"""

from __future__ import annotations

from glmbench.baselines.kmer import KmerCompositionAdapter, kmer_frequency_vector

__all__ = ["KmerCompositionAdapter", "kmer_frequency_vector"]
