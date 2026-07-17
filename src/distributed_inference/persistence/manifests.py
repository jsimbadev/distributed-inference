"""Persisted result manifest primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

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

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> TargetSpec:
        """Rehydrate a target spec from a manifest payload."""
        return cls(
            identifier=str(payload["identifier"]),
            semantics=str(payload["semantics"]),
            dimension=int(payload["dimension"]),
            coordinate_space=str(payload["coordinate_space"]),
        )


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

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> EngineSpec:
        """Rehydrate an engine spec from a manifest payload."""
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            msg = "Engine manifest config must be a mapping."
            raise ManifestError(msg)
        return cls(
            name=str(payload["name"]),
            version=str(payload["version"]),
            config=dict(config),
        )


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

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ArtifactReference:
        """Rehydrate an artifact reference from a manifest payload."""
        return cls(
            uri=str(payload["uri"]),
            media_type=str(payload["media_type"]),
            checksum=str(payload["checksum"]),
        )


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
    checkpoints: Mapping[str, ArtifactReference] = field(default_factory=dict)
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
    checkpoints: Mapping[str, ArtifactReference] = field(default_factory=dict)
    context_metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_result(
        cls,
        result: InferenceResult[Any],
        metadata: ResultManifestMetadata | None = None,
    ) -> ResultManifest:
        """Create a serializable manifest from an in-memory result."""
        context = result.run.context
        metadata = metadata or _require_manifest_metadata(result)
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
            checkpoints=dict(metadata.checkpoints),
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
            "checkpoints": {
                name: reference.to_manifest()
                for name, reference in self.checkpoints.items()
            },
            "context": {
                "run_id": self.run_id,
                "metadata": dict(self.context_metadata),
            },
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ResultManifest:
        """Rehydrate a result manifest from a manifest payload."""
        identity = payload["identity"]
        if not isinstance(identity, Mapping):
            msg = "Result manifest identity must be a mapping."
            raise ManifestError(msg)
        model = payload["model"]
        target = payload["target"]
        engine = payload["engine"]
        reproducibility = payload["reproducibility"]
        artifacts = payload["artifacts"]
        checkpoints = payload.get("checkpoints", {})
        timestamps = payload.get("timestamps", {})
        context = payload.get("context", {})
        for name, value in {
            "model": model,
            "target": target,
            "engine": engine,
            "reproducibility": reproducibility,
            "artifacts": artifacts,
            "checkpoints": checkpoints,
            "timestamps": timestamps,
            "context": context,
        }.items():
            if not isinstance(value, Mapping):
                msg = f"Result manifest {name} must be a mapping."
                raise ManifestError(msg)
        context_metadata = context.get("metadata", {})
        if not isinstance(context_metadata, Mapping):
            msg = "Result manifest context metadata must be a mapping."
            raise ManifestError(msg)
        return cls(
            schema_version=str(payload["schema_version"]),
            workflow_id=str(identity["workflow_id"]),
            run_id=str(identity["run_id"]),
            replicate_id=str(identity["replicate_id"]),
            attempt_id=str(identity["attempt_id"]),
            model=ModelSpec.from_manifest(model),
            target=TargetSpec.from_manifest(target),
            engine=EngineSpec.from_manifest(engine),
            random_stream=RandomStreamSpec.from_manifest(
                _require_mapping(reproducibility["random_stream"]),
            ),
            diagnostics=dict(_require_mapping(payload["diagnostics"])),
            artifacts={
                str(name): ArtifactReference.from_manifest(
                    _require_mapping(reference),
                )
                for name, reference in artifacts.items()
            },
            checkpoints={
                str(name): ArtifactReference.from_manifest(
                    _require_mapping(reference),
                )
                for name, reference in checkpoints.items()
            },
            context_metadata=dict(context_metadata),
            status=str(payload["status"]),
            started_at=_optional_str(timestamps.get("started_at")),
            completed_at=_optional_str(timestamps.get("completed_at")),
        )


def _require_manifest_metadata(result: InferenceResult[Any]) -> ResultManifestMetadata:
    metadata = getattr(result, "manifest_metadata", None)
    if not isinstance(metadata, ResultManifestMetadata):
        msg = "Cannot persist a result without manifest metadata."
        raise ManifestError(msg)
    return metadata


def _require_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = "Manifest field must be a mapping."
        raise ManifestError(msg)
    return cast(Mapping[str, Any], value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
