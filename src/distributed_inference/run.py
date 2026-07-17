"""High-level execution and persistence facade."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from distributed_inference.engines.base import InferenceEngine
from distributed_inference.execution import (
    ExecutedInference,
    ExecutionBackend,
    LocalExecutionBackend,
)
from distributed_inference.persistence.artifacts import write_result_artifacts
from distributed_inference.persistence.local import (
    LocalInferenceStore,
    PersistedInferenceFiles,
)
from distributed_inference.persistence.manifests import (
    ArtifactReference,
    EngineSpec,
    ResultManifest,
    ResultManifestMetadata,
    TargetSpec,
)
from distributed_inference.persistence.runs import InferenceRunSpec


@dataclass(frozen=True)
class PersistedInference:
    """Completed inference execution plus persisted local manifest paths."""

    executed: ExecutedInference
    result_manifest: ResultManifest
    files: PersistedInferenceFiles
    artifacts: Mapping[str, ArtifactReference]
    checkpoints: Mapping[str, ArtifactReference] = field(default_factory=dict)


def run_inference(
    *,
    run: InferenceRunSpec,
    engine: InferenceEngine,
    store: LocalInferenceStore,
    target: TargetSpec | None = None,
    backend: ExecutionBackend | None = None,
    attempt_number: int = 1,
    checkpoints: Mapping[str, ArtifactReference] | None = None,
    execution_metadata: Mapping[str, Any] | None = None,
) -> PersistedInference:
    """Run inference, write artifacts, and persist run/result/execution manifests."""
    runtime_run = run.to_inference_run()
    execution_backend = backend or LocalExecutionBackend()
    executed = execution_backend.execute(
        runtime_run,
        engine,
        attempt_number=attempt_number,
        metadata=execution_metadata,
    )
    artifacts = write_result_artifacts(store, executed.result)
    checkpoint_references = dict(checkpoints or {})
    result_manifest = ResultManifest.from_result(
        executed.result,
        ResultManifestMetadata(
            schema_version="1",
            attempt_number=executed.execution.attempt.attempt_number,
            model=run.model,
            target=target or _default_target(run, executed),
            engine=_engine_spec(engine),
            random_stream=run.random_stream,
            artifacts=artifacts,
            checkpoints=checkpoint_references,
            status=executed.execution.status,
            started_at=executed.execution.attempt.started_at,
            completed_at=executed.execution.attempt.completed_at,
        ),
    )
    files = store.write_inference(
        run=run,
        result=result_manifest,
        execution=executed.execution,
    )
    return PersistedInference(
        executed=executed,
        result_manifest=result_manifest,
        files=files,
        artifacts=artifacts,
        checkpoints=checkpoint_references,
    )


def _default_target(
    run: InferenceRunSpec,
    executed: ExecutedInference,
) -> TargetSpec:
    model_info = executed.result.run.model.info
    return TargetSpec(
        identifier=f"{run.name}.{model_info.name}",
        semantics=str(run.context_metadata.get("target", "target-density")),
        dimension=model_info.dimension,
        coordinate_space=model_info.input_space.value,
    )


def _engine_spec(engine: InferenceEngine) -> EngineSpec:
    return EngineSpec(
        name=engine.name,
        version=_engine_version(engine),
        config=_engine_config(engine),
    )


def _engine_version(engine: InferenceEngine) -> str:
    engine_version = getattr(engine, "version", None)
    if engine_version is not None:
        return str(engine_version)
    if engine.name == "pyvbmc":
        try:
            return version("pyvbmc")
        except PackageNotFoundError:
            return "unknown"
    return "unknown"


def _engine_config(engine: InferenceEngine) -> Mapping[str, Any]:
    config = getattr(engine, "config", None)
    if isinstance(config, Mapping):
        return dict(config)
    options = getattr(engine, "options", None)
    raw_options = getattr(options, "raw_options", None)
    if isinstance(raw_options, Mapping):
        return dict(raw_options)
    return {}
