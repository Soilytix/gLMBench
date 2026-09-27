"""gLMBench — a standalone, model-agnostic benchmark suite for genomic language models.

The core package is dependency-light by design (numpy, pandas, pyyaml, pydantic +
stdlib only). Heavy deep-learning dependencies (torch, transformers, evo2) live behind
adapters and their runner environments and are never imported by core; the torch-free
import gate (``tests/unit/test_phase0.py``) enforces this.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
