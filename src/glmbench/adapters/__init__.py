"""The model-agnostic adapter spine.

Re-exports the contract (:class:`ModelAdapter`, :class:`Capability`,
:class:`EmbeddingResult`), the model hash, the wire protocol, and the runner
backends. Importing this package also registers the bundled adapters (the model
adapters plus the model-free :class:`EchoAdapter` test double) so they are
discoverable via the registry. This whole package is core (torch-free) — model
code lives in :mod:`glmbench.runners`, which core only shells out to.
"""

from __future__ import annotations

from .base import (
    CAPABILITY_METHODS,
    COMPARABLE_READOUTS,
    Capability,
    CapabilityNotImplemented,
    EmbeddingResult,
    ModelAdapter,
    Readout,
    missing_capability_methods,
)
from .echo import EchoAdapter
from .evo import EvoAdapter
from .evo2_arc import Evo2ArcAdapter
from .genomeocean import GenomeOceanAdapter
from .glm2 import GLM2Adapter
from .hashing import (
    canonical_config_json,
    compute_weights_digest,
    file_sha256_digest,
    model_hash,
)
from .loam_hf import LOAMHFAdapter
from .ntv3 import NTv3Adapter
from .prokbert import ProkBertAdapter
from .runner_backend import (
    DockerRunner,
    LocalSubprocessRunner,
    RunnerBackend,
    RunnerError,
    RunResult,
)
from .wire import (
    WIRE_VERSION,
    Request,
    Response,
    WireProtocolError,
    WireVersionError,
)

__all__ = [
    "CAPABILITY_METHODS",
    "COMPARABLE_READOUTS",
    "Readout",
    "Capability",
    "CapabilityNotImplemented",
    "EmbeddingResult",
    "ModelAdapter",
    "missing_capability_methods",
    "EchoAdapter",
    "EvoAdapter",
    "Evo2ArcAdapter",
    "GenomeOceanAdapter",
    "GLM2Adapter",
    "LOAMHFAdapter",
    "NTv3Adapter",
    "ProkBertAdapter",
    "canonical_config_json",
    "compute_weights_digest",
    "file_sha256_digest",
    "model_hash",
    "DockerRunner",
    "LocalSubprocessRunner",
    "RunnerBackend",
    "RunnerError",
    "RunResult",
    "WIRE_VERSION",
    "Request",
    "Response",
    "WireProtocolError",
    "WireVersionError",
]
