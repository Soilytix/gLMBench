"""Every model spec computes the model hash of the record it is documented against.

The model hash is ``sha256(weights digest ‖ hashed config ‖ adapter@version)``. For the nine
external models, the paper record's hash is recomputed from the spec alone (``hf_revision``
digests: nothing is downloaded, nothing runs). For LOAM, the hash covers the sha256 of the
weight file, so it needs the weights: those cases run only when ``GLMBENCH_LOAM_HF_ROOT``
points at a directory holding ``LOAM-*/`` downloads.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from glmbench import registry
from glmbench.config.model_spec import load_model_spec

REPO = Path(__file__).resolve().parents[2]


def _models() -> list[dict[str, str]]:
    with (REPO / "paper" / "models.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


EXTERNAL = [m for m in _models() if not m["model_id"].startswith("LOAM-")]
LOAM = [m for m in _models() if m["model_id"].startswith("LOAM-")]


def _hash(spec_path: Path) -> str:
    spec = load_model_spec(spec_path)
    adapter = registry.resolve("adapter", spec.adapter).from_spec(spec, dry_run=True)
    return adapter.model_hash()


def test_the_paper_lists_thirteen_models() -> None:
    assert len(EXTERNAL) == 9 and len(LOAM) == 4


@pytest.mark.parametrize("model", EXTERNAL, ids=lambda m: m["model_id"])
def test_external_spec_reproduces_the_paper_hash(model: dict[str, str]) -> None:
    assert _hash(REPO / model["spec"]) == model["paper_model_hash"]


@pytest.mark.skipif(
    not os.environ.get("GLMBENCH_LOAM_HF_ROOT"),
    reason="needs the LOAM weights: set GLMBENCH_LOAM_HF_ROOT to a directory of LOAM-*/ downloads",
)
@pytest.mark.parametrize("model", LOAM, ids=lambda m: m["model_id"])
def test_loam_spec_reproduces_the_public_route_hash(model: dict[str, str]) -> None:
    root = Path(os.environ["GLMBENCH_LOAM_HF_ROOT"])
    if not (root / model["model_id"]).is_dir():
        pytest.skip(f"{model['model_id']} is not under {root}")
    assert _hash(REPO / model["spec"]) == model["public_route_model_hash"]
