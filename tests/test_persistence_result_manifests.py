import json
from dataclasses import dataclass

import numpy as np
import pytest

from distributed_inference import CallableModel, EvaluationContext, InferenceRun
from distributed_inference._validation import FloatArray
from distributed_inference.engines.base import InferenceResult
from distributed_inference.errors import ManifestError
from distributed_inference.persistence.manifests import (
    ArtifactReference,
    EngineSpec,
    ResultManifest,
    ResultManifestMetadata,
    TargetSpec,
)
from distributed_inference.persistence.models import ModelSpec, PythonCallableSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec


def _log_density(x: FloatArray, context: EvaluationContext | None) -> float:
    return -0.5 * float(np.dot(x, x))


@dataclass(frozen=True)
class FakeInferenceResult(InferenceResult[object]):
    manifest_metadata: ResultManifestMetadata | None = None


class FakeInferenceEngine:
    def __init__(self, manifest_metadata: ResultManifestMetadata) -> None:
        self.manifest_metadata = manifest_metadata

    @property
    def name(self) -> str:
        return "fake"

    def run_inference(self, run: InferenceRun) -> FakeInferenceResult:
        return FakeInferenceResult(
            engine_name=self.name,
            run=run,
            posterior=object(),
            diagnostics={"n_evaluations": 12},
            manifest_metadata=self.manifest_metadata,
        )


@pytest.fixture
def inference_run() -> InferenceRun:
    model = CallableModel(name="gaussian", dimension=1, fn=_log_density)
    context = EvaluationContext(
        run_id="run-001",
        rng=np.random.default_rng(42),
        metadata={"target": "full-posterior"},
        cache={"runtime-only": object()},
    )
    return InferenceRun(
        name="gaussian-smoke",
        model=model,
        initial_point=np.array([0.0]),
        context=context,
        record_evaluations=True,
    )


@pytest.fixture
def model_spec() -> ModelSpec:
    return ModelSpec(
        callable=PythonCallableSpec(
            module="examples.gaussian",
            qualname="log_density",
            project_root="/example-project",
            source_path="models/gaussian.py",
        ),
        version="1",
        config={"dimension": 1},
    )


@pytest.fixture
def target_spec() -> TargetSpec:
    return TargetSpec(
        identifier="examples.gaussian.full_posterior",
        semantics="full-posterior",
        dimension=1,
        coordinate_space="unconstrained",
    )


@pytest.fixture
def engine_spec() -> EngineSpec:
    return EngineSpec(
        name="fake",
        version="0.1",
        config={"max_fun_evals": 100},
    )


@pytest.fixture
def random_stream_spec() -> RandomStreamSpec:
    return RandomStreamSpec(
        algorithm="numpy.pcg64",
        seed=42,
        stream_id="stream-000",
        schema_version="1",
    )


@pytest.fixture
def artifact_references() -> dict[str, ArtifactReference]:
    return {
        "posterior": ArtifactReference(
            uri="artifacts/posterior.npz",
            media_type="application/x-npz",
            checksum="sha256:posterior",
        ),
    }


@pytest.fixture
def checkpoint_references() -> dict[str, ArtifactReference]:
    return {
        "checkpoint": ArtifactReference(
            uri="checkpoints/engine-state.json",
            media_type="application/json",
            checksum="sha256:checkpoint",
        ),
    }


@pytest.fixture
def manifest_metadata(
    model_spec: ModelSpec,
    target_spec: TargetSpec,
    engine_spec: EngineSpec,
    random_stream_spec: RandomStreamSpec,
    artifact_references: dict[str, ArtifactReference],
    checkpoint_references: dict[str, ArtifactReference],
) -> ResultManifestMetadata:
    return ResultManifestMetadata(
        schema_version="1",
        attempt_number=1,
        model=model_spec,
        target=target_spec,
        engine=engine_spec,
        random_stream=random_stream_spec,
        artifacts=artifact_references,
        checkpoints=checkpoint_references,
        status="completed",
        started_at="2026-07-17T09:00:00Z",
        completed_at="2026-07-17T09:05:00Z",
    )


@pytest.fixture
def inference_result(
    inference_run: InferenceRun,
    manifest_metadata: ResultManifestMetadata,
) -> FakeInferenceResult:
    return FakeInferenceEngine(manifest_metadata).run_inference(inference_run)


@pytest.fixture
def unidentified_inference_result(
    manifest_metadata: ResultManifestMetadata,
) -> FakeInferenceResult:
    model = CallableModel(name="gaussian", dimension=1, fn=_log_density)
    run = InferenceRun(
        name="gaussian-smoke",
        model=model,
        initial_point=np.array([0.0]),
        context=EvaluationContext(),
    )
    return FakeInferenceEngine(manifest_metadata).run_inference(run)


@pytest.fixture
def result_manifest(
    inference_result: InferenceResult[object],
) -> ResultManifest:
    return ResultManifest.from_result(inference_result)


def test_result_manifest_has_explicit_schema_version(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["schema_version"] == "1"


def test_result_manifest_requires_run_identity(
    unidentified_inference_result: InferenceResult[object],
) -> None:
    with pytest.raises(ManifestError):
        ResultManifest.from_result(unidentified_inference_result)


def test_result_manifest_keeps_run_and_attempt_identity_distinct(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["identity"] == {
        "name": "gaussian-smoke",
        "run_id": "run-001",
        "attempt_number": 1,
    }


def test_result_manifest_records_versioned_random_stream_spec(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["reproducibility"]["random_stream"]["schema_version"] == "1"


def test_result_manifest_references_posterior_artifact(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["artifacts"]["posterior"]["uri"] == "artifacts/posterior.npz"


def test_result_manifest_uses_callable_model_reference(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["model"]["callable"]["qualname"] == "log_density"


def test_result_manifest_references_checkpoint_separately(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()
    checkpoint = payload["checkpoints"]["checkpoint"]

    assert checkpoint["uri"] == "checkpoints/engine-state.json"


def test_result_manifest_keeps_checkpoint_out_of_completed_artifacts(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert "checkpoint" not in payload["artifacts"]


def test_result_manifest_does_not_serialize_runtime_context_cache(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert "cache" not in payload["context"]


def test_result_manifest_is_json_serializable(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert isinstance(json.dumps(payload), str)


def test_result_manifest_round_trips_through_json(
    result_manifest: ResultManifest,
) -> None:
    payload = json.loads(json.dumps(result_manifest.to_manifest()))

    assert ResultManifest.from_manifest(payload).to_manifest() == payload
