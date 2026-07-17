import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from distributed_inference import CallableModel, EvaluationContext
from distributed_inference._validation import FloatArray
from distributed_inference.persistence.models import ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec
from distributed_inference.persistence.runs import InferenceRunSpec

PROJECT_ROOT = Path.cwd()


def _build_prng_model(config: Mapping[str, Any]) -> CallableModel:
    dimension = int(config["dimension"])

    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        if context is None or context.rng is None:
            msg = "prng-model requires an EvaluationContext with an rng"
            raise RuntimeError(msg)
        return -0.5 * float(np.dot(x, x)) + float(context.rng.normal())

    return CallableModel(
        name="prng-gaussian",
        dimension=dimension,
        fn=log_density,
    )


@pytest.fixture
def inference_run_spec() -> InferenceRunSpec:
    return InferenceRunSpec(
        name="local-prng-smoke",
        run_id="run-001",
        replicate_id="replicate-000",
        attempt_id="attempt-000",
        model=ModelSpec.from_callable(
            _build_prng_model,
            config={"dimension": 1},
            project_root=PROJECT_ROOT,
            version="1",
        ),
        initial_point=[0.0],
        random_stream=RandomStreamSpec(
            algorithm="numpy.pcg64",
            seed=42,
            stream_id="replicate-000",
            schema_version="1",
        ),
        context_metadata={"target": "full-posterior"},
        record_evaluations=True,
    )


def test_inference_run_spec_rehydrates_initial_point(
    inference_run_spec: InferenceRunSpec,
) -> None:
    run = inference_run_spec.to_inference_run()

    np.testing.assert_allclose(run.initial_point, np.array([0.0]))


def test_inference_run_spec_rehydrates_runtime_context(
    inference_run_spec: InferenceRunSpec,
) -> None:
    run = inference_run_spec.to_inference_run()
    context = cast(EvaluationContext, run.context)

    assert context.run_id == "run-001"


def test_inference_run_spec_rehydrates_random_stream(
    inference_run_spec: InferenceRunSpec,
) -> None:
    run = inference_run_spec.to_inference_run()
    same_run = inference_run_spec.to_inference_run()
    context = cast(EvaluationContext, run.context)
    same_context = cast(EvaluationContext, same_run.context)

    assert run.model(run.initial_point, context) == pytest.approx(
        same_run.model(same_run.initial_point, same_context)
    )


def test_inference_run_spec_manifest_is_json_serializable(
    inference_run_spec: InferenceRunSpec,
) -> None:
    payload = inference_run_spec.to_manifest()

    assert isinstance(json.dumps(payload), str)


def test_inference_run_spec_manifest_does_not_flatten_live_rng(
    inference_run_spec: InferenceRunSpec,
) -> None:
    payload = inference_run_spec.to_manifest()

    assert "rng" not in payload["context"]


def test_inference_run_spec_manifest_uses_callable_model_reference(
    inference_run_spec: InferenceRunSpec,
) -> None:
    payload = inference_run_spec.to_manifest()

    assert payload["model"]["kind"] == "python-callable"
