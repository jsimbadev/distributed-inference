"""Serializable inference-run specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from distributed_inference.engines.base import InferenceRun
from distributed_inference.errors import ManifestError
from distributed_inference.model import EvaluationContext
from distributed_inference.persistence.models import ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec


@dataclass(frozen=True)
class InferenceRunSpec:
    """Serializable specification for rehydrating an inference run."""

    name: str
    run_id: str
    replicate_id: str
    attempt_id: str
    model: ModelSpec
    initial_point: Sequence[float]
    random_stream: RandomStreamSpec
    context_metadata: Mapping[str, Any] = field(default_factory=dict)
    record_evaluations: bool = False
    schema_version: str = "1"

    def to_inference_run(self) -> InferenceRun:
        """Rehydrate a runtime inference run from serializable metadata."""
        context = EvaluationContext(
            run_id=self.run_id,
            rng=self.random_stream.to_generator(),
            metadata=dict(self.context_metadata),
        )
        return InferenceRun(
            model=self.model.to_model(),
            initial_point=np.asarray(self.initial_point, dtype=np.float64),
            context=context,
            record_evaluations=self.record_evaluations,
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable inference-run manifest."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "identity": {
                "run_id": self.run_id,
                "replicate_id": self.replicate_id,
                "attempt_id": self.attempt_id,
            },
            "model": self.model.to_manifest(),
            "initial_point": list(self.initial_point),
            "random_stream": self.random_stream.to_manifest(),
            "context": {
                "run_id": self.run_id,
                "metadata": dict(self.context_metadata),
            },
            "record_evaluations": self.record_evaluations,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> InferenceRunSpec:
        """Rehydrate an inference-run spec from a manifest payload."""
        identity = payload["identity"]
        if not isinstance(identity, Mapping):
            msg = "Inference-run manifest identity must be a mapping."
            raise ManifestError(msg)
        model = payload["model"]
        if not isinstance(model, Mapping):
            msg = "Inference-run manifest model must be a mapping."
            raise ManifestError(msg)
        random_stream = payload["random_stream"]
        if not isinstance(random_stream, Mapping):
            msg = "Inference-run manifest random_stream must be a mapping."
            raise ManifestError(msg)
        context = payload.get("context", {})
        if not isinstance(context, Mapping):
            msg = "Inference-run manifest context must be a mapping."
            raise ManifestError(msg)
        context_metadata = context.get("metadata", {})
        if not isinstance(context_metadata, Mapping):
            msg = "Inference-run manifest context metadata must be a mapping."
            raise ManifestError(msg)
        return cls(
            schema_version=str(payload["schema_version"]),
            name=str(payload["name"]),
            run_id=str(identity["run_id"]),
            replicate_id=str(identity["replicate_id"]),
            attempt_id=str(identity["attempt_id"]),
            model=ModelSpec.from_manifest(model),
            initial_point=list(payload["initial_point"]),
            random_stream=RandomStreamSpec.from_manifest(random_stream),
            context_metadata=dict(context_metadata),
            record_evaluations=bool(payload["record_evaluations"]),
        )
