"""Serializable random-stream specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from distributed_inference.errors import ModelError


@dataclass(frozen=True)
class RandomStreamSpec:
    """Durable metadata for reconstructing a logical random stream."""

    algorithm: str
    seed: int
    stream_id: str
    schema_version: str

    def to_generator(self) -> np.random.Generator:
        """Rehydrate a process-local NumPy generator from restart metadata."""
        match self.algorithm.lower():
            case "numpy.pcg64" | "pcg64":
                return np.random.Generator(np.random.PCG64(self.seed))
            case _:
                msg = f"Unsupported random-stream algorithm {self.algorithm!r}."
                raise ModelError(msg)

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable random-stream manifest."""
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "stream_id": self.stream_id,
        }
