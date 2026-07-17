import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from distributed_inference import EvaluationContext
from distributed_inference.persistence.models import ModelBuilder, ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec
from distributed_inference.persistence.runs import InferenceRunSpec

TEMPORARY_RUN_MODEL_SOURCE = """
import numpy as np


from distributed_inference import CallableModel


def build_prng_model(config):
    dimension = int(config["dimension"])

    def log_density(x, context):
        if context is None or context.rng is None:
            msg = "prng-model requires an EvaluationContext with an rng"
            raise RuntimeError(msg)
        return -0.5 * float(np.dot(x, x)) + float(context.rng.normal())

    return CallableModel(
        name="prng-gaussian",
        dimension=dimension,
        fn=log_density,
    )
"""


@pytest.fixture
def temporary_project(tmp_path: Path) -> Path:
    (tmp_path / "temporary_run_model.py").write_text(
        TEMPORARY_RUN_MODEL_SOURCE,
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def prng_model_builder(temporary_project: Path) -> Iterator[ModelBuilder]:
    module = _load_module(
        "temporary_run_model",
        temporary_project / "temporary_run_model.py",
    )
    sys.modules[module.__name__] = module
    yield module.build_prng_model
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def inference_run_spec(
    temporary_project: Path,
    prng_model_builder: ModelBuilder,
) -> InferenceRunSpec:
    return InferenceRunSpec(
        name="local-prng-smoke",
        run_id="run-001",
        replicate_id="replicate-000",
        attempt_id="attempt-000",
        model=ModelSpec.from_callable(
            prng_model_builder,
            config={"dimension": 1},
            project_root=temporary_project,
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


def test_inference_run_spec_manifest_separates_run_replicate_and_attempt(
    inference_run_spec: InferenceRunSpec,
) -> None:
    payload = inference_run_spec.to_manifest()

    assert payload["identity"] == {
        "run_id": "run-001",
        "replicate_id": "replicate-000",
        "attempt_id": "attempt-000",
    }


def test_inference_run_spec_round_trips_through_json(
    inference_run_spec: InferenceRunSpec,
) -> None:
    payload = json.loads(json.dumps(inference_run_spec.to_manifest()))

    assert InferenceRunSpec.from_manifest(payload).to_manifest() == payload


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
