"""Persisted result manifest primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from distributed_inference.engines.base import InferenceResult
from distributed_inference.errors import ManifestError
from distributed_inference.persistence.models import ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec


@dataclass(frozen=True)
class TargetSpec:
    """Serializable description of the target represented by a run."""

    identifier: str
    semantics: str
    dimension: int
    coordinate_space: str

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable target manifest."""
        return {
            "identifier": self.identifier,
            "semantics": self.semantics,
            "dimension": self.dimension,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True)
class EngineSpec:
    """Serializable description of the inference engine configuration."""

    name: str
    version: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable engine manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class ArtifactReference:
    """Reference to a persisted artifact produced by an inference run."""

    uri: str
    media_type: str
    checksum: str

    def to_manifest(self) -> dict[str, str]:
        """Return a serializable artifact reference."""
        return {
            "uri": self.uri,
            "media_type": self.media_type,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class ResultManifestMetadata:
    """Implementation-owned metadata required to persist an inference result."""

    schema_version: str
    workflow_id: str
    replicate_id: str
    attempt_id: str
    model: ModelSpec
    target: TargetSpec
    engine: EngineSpec
    random_stream: RandomStreamSpec
    artifacts: Mapping[str, ArtifactReference]
    status: str
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class ResultManifest:
    """Serializable manifest for a completed inference result."""

    schema_version: str
    workflow_id: str
    run_id: str
    replicate_id: str
    attempt_id: str
    model: ModelSpec
    target: TargetSpec
    engine: EngineSpec
    random_stream: RandomStreamSpec
    diagnostics: Mapping[str, Any]
    artifacts: Mapping[str, ArtifactReference]
    context_metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_result(
        cls,
        result: InferenceResult[Any],
    ) -> ResultManifest:
        """Create a serializable manifest from an in-memory result."""
        context = result.run.context
        metadata = _require_manifest_metadata(result)
        if context is None or not context.run_id:
            msg = "Cannot persist a result whose run has no run_id."
            raise ManifestError(msg)
        return cls(
            schema_version=metadata.schema_version,
            workflow_id=metadata.workflow_id,
            run_id=context.run_id,
            replicate_id=metadata.replicate_id,
            attempt_id=metadata.attempt_id,
            model=metadata.model,
            target=metadata.target,
            engine=metadata.engine,
            random_stream=metadata.random_stream,
            diagnostics=dict(result.diagnostics),
            artifacts=dict(metadata.artifacts),
            context_metadata=dict(context.metadata) if context is not None else {},
            status=metadata.status,
            started_at=metadata.started_at,
            completed_at=metadata.completed_at,
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable result manifest."""
        return {
            "schema_version": self.schema_version,
            "identity": {
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "replicate_id": self.replicate_id,
                "attempt_id": self.attempt_id,
            },
            "model": self.model.to_manifest(),
            "target": self.target.to_manifest(),
            "engine": self.engine.to_manifest(),
            "reproducibility": {
                "random_stream": self.random_stream.to_manifest(),
            },
            "status": self.status,
            "timestamps": {
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            },
            "diagnostics": dict(self.diagnostics),
            "artifacts": {
                name: reference.to_manifest()
                for name, reference in self.artifacts.items()
            },
            "context": {
                "run_id": self.run_id,
                "metadata": dict(self.context_metadata),
            },
        }


def _require_manifest_metadata(result: InferenceResult[Any]) -> ResultManifestMetadata:
    metadata = getattr(result, "manifest_metadata", None)
    if not isinstance(metadata, ResultManifestMetadata):
        msg = "Cannot persist a result without manifest metadata."
        raise ManifestError(msg)
    return metadata
