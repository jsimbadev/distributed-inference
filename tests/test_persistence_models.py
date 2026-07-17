import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from distributed_inference import CallableModel, EvaluationContext
from distributed_inference._validation import FloatArray
from distributed_inference.persistence.models import ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec

PROJECT_ROOT = Path.cwd()


def _build_prng_model(config: Mapping[str, Any]) -> CallableModel:
    dimension = int(config["dimension"])
    noise_scale = float(config["noise_scale"])

    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        if context is None or context.rng is None:
            msg = "prng-model requires an EvaluationContext with an rng"
            raise RuntimeError(msg)
        deterministic_part = -0.5 * float(np.dot(x, x))
        stochastic_part = noise_scale * float(context.rng.normal())
        return deterministic_part + stochastic_part

    return CallableModel(
        name="prng-gaussian",
        dimension=dimension,
        fn=log_density,
    )


@pytest.fixture
def prng_model_spec() -> ModelSpec:
    return ModelSpec.from_callable(
        _build_prng_model,
        config={"dimension": 1, "noise_scale": 0.25},
        project_root=PROJECT_ROOT,
        version="1",
    )


def test_model_spec_captures_importable_callable(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["module"] == _build_prng_model.__module__


def test_model_spec_captures_callable_qualname(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["qualname"] == "_build_prng_model"


def test_model_spec_captures_project_relative_source_path(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["source_path"] == "tests/test_persistence_models.py"


def test_model_spec_manifest_is_json_serializable(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert isinstance(json.dumps(payload), str)


def test_rehydrated_model_uses_context_random_stream(
    prng_model_spec: ModelSpec,
) -> None:
    model = prng_model_spec.to_model()
    same_model = prng_model_spec.to_model()
    stream = RandomStreamSpec(
        algorithm="numpy.pcg64",
        seed=42,
        stream_id="replicate-0",
        schema_version="1",
    )
    context = EvaluationContext(rng=stream.to_generator())
    same_context = EvaluationContext(rng=stream.to_generator())

    assert model(np.array([0.0]), context) == pytest.approx(
        same_model(np.array([0.0]), same_context)
    )


def test_rehydrated_model_requires_runtime_random_stream(
    prng_model_spec: ModelSpec,
) -> None:
    model = prng_model_spec.to_model()

    with pytest.raises(RuntimeError):
        model(np.array([0.0]), EvaluationContext())
