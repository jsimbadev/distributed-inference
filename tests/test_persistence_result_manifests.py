import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from distributed_inference import CallableModel, EvaluationContext, InferenceRun
from distributed_inference._validation import FloatArray
from distributed_inference.engines.base import InferenceResult
from distributed_inference.persistence.manifests import (
    ArtifactReference,
    EngineSpec,
    ResultManifest,
    TargetSpec,
)
from distributed_inference.persistence.models import ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec

PROJECT_ROOT = Path.cwd()


def _log_density(x: FloatArray, context: EvaluationContext | None) -> float:
    return -0.5 * float(np.dot(x, x))


def _build_gaussian_model(config: Mapping[str, Any]) -> CallableModel:
    dimension = int(config["dimension"])
    return CallableModel(name="gaussian", dimension=dimension, fn=_log_density)


@pytest.fixture
def inference_result() -> InferenceResult[object]:
    model = CallableModel(name="gaussian", dimension=1, fn=_log_density)
    context = EvaluationContext(
        run_id="run-001",
        rng=np.random.default_rng(42),
        metadata={"target": "full-posterior"},
        cache={"runtime-only": object()},
    )
    run = InferenceRun(
        model=model,
        initial_point=np.array([0.0]),
        context=context,
        record_evaluations=True,
    )
    return InferenceResult(
        engine_name="pyvbmc",
        run=run,
        posterior=object(),
        diagnostics={"n_evaluations": 12},
    )


@pytest.fixture
def result_manifest(inference_result: InferenceResult[object]) -> ResultManifest:
    return ResultManifest.from_result(
        inference_result,
        schema_version="1",
        workflow_id="workflow-001",
        task_id="task-000",
        replicate_id="replicate-000",
        attempt_id="attempt-000",
        model=ModelSpec.from_callable(
            _build_gaussian_model,
            config={"dimension": 1},
            project_root=PROJECT_ROOT,
            version="1",
        ),
        target=TargetSpec(
            identifier="examples.gaussian.full_posterior",
            semantics="full-posterior",
            dimension=1,
            coordinate_space="unconstrained",
        ),
        engine=EngineSpec(
            name="pyvbmc",
            version="0.1",
            config={"max_fun_evals": 100},
        ),
        random_stream=RandomStreamSpec(
            algorithm="numpy.pcg64",
            seed=42,
            stream_id="replicate-000",
            schema_version="1",
        ),
        artifacts={
            "posterior": ArtifactReference(
                uri="artifacts/posterior.npz",
                media_type="application/x-npz",
                checksum="sha256:posterior",
            ),
            "checkpoint": ArtifactReference(
                uri="checkpoints/engine-state.msgpack",
                media_type="application/msgpack",
                checksum="sha256:checkpoint",
            ),
        },
        status="completed",
        started_at="2026-07-17T09:00:00Z",
        completed_at="2026-07-17T09:05:00Z",
    )


def test_result_manifest_has_explicit_schema_version(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["schema_version"] == "1"


def test_result_manifest_defaults_missing_run_id_to_uuid4() -> None:
    model = CallableModel(name="gaussian", dimension=1, fn=_log_density)
    run = InferenceRun(
        model=model,
        initial_point=np.array([0.0]),
        context=EvaluationContext(),
    )
    result = InferenceResult(
        engine_name="pyvbmc",
        run=run,
        posterior=object(),
        diagnostics={},
    )
    manifest = ResultManifest.from_result(
        result,
        schema_version="1",
        workflow_id="workflow-001",
        task_id="task-000",
        replicate_id="replicate-000",
        attempt_id="attempt-000",
        model=ModelSpec.from_callable(
            _build_gaussian_model,
            config={"dimension": 1},
            project_root=PROJECT_ROOT,
            version="1",
        ),
        target=TargetSpec(
            identifier="examples.gaussian.full_posterior",
            semantics="full-posterior",
            dimension=1,
            coordinate_space="unconstrained",
        ),
        engine=EngineSpec(name="pyvbmc", version="0.1"),
        random_stream=RandomStreamSpec(
            algorithm="numpy.pcg64",
            seed=42,
            stream_id="replicate-000",
            schema_version="1",
        ),
        artifacts={},
        status="completed",
    )

    assert UUID(manifest.to_manifest()["identity"]["run_id"]).version == 4


def test_result_manifest_keeps_task_replicate_and_attempt_distinct(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()

    assert payload["identity"] == {
        "workflow_id": "workflow-001",
        "run_id": "run-001",
        "task_id": "task-000",
        "replicate_id": "replicate-000",
        "attempt_id": "attempt-000",
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

    assert payload["model"]["callable"]["qualname"] == "_build_gaussian_model"


def test_result_manifest_references_checkpoint_separately(
    result_manifest: ResultManifest,
) -> None:
    payload = result_manifest.to_manifest()
    checkpoint = payload["artifacts"]["checkpoint"]

    assert checkpoint["uri"] == "checkpoints/engine-state.msgpack"


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
