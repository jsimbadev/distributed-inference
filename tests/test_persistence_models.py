import importlib.util
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from distributed_inference import EvaluationContext
from distributed_inference.errors import ModelError
from distributed_inference.model import Model
from distributed_inference.persistence.models import (
    ModelBuilder,
    ModelSpec,
    PythonCallableSpec,
)
from distributed_inference.persistence.random_streams import RandomStreamSpec

TEMPORARY_MODEL_SOURCE = """
import numpy as np

from distributed_inference import CallableModel


def build_prng_model(config):
    dimension = int(config["dimension"])
    noise_scale = float(config["noise_scale"])

    def log_density(x, context):
        if context is None or context.rng is None:
            msg = "prng-model requires an EvaluationContext with an rng"
            raise RuntimeError(msg)
        deterministic_part = -0.5 * float(np.dot(x, x))
        stochastic_part = noise_scale * float(context.rng.normal())
        return deterministic_part + stochastic_part

    return CallableModel(
        name="temporary-prng-gaussian",
        dimension=dimension,
        fn=log_density,
    )
"""


@pytest.fixture
def temporary_project() -> Iterator[Path]:
    with TemporaryDirectory() as directory:
        project_root = Path(directory)
        (project_root / "temporary_model.py").write_text(
            TEMPORARY_MODEL_SOURCE,
            encoding="utf-8",
        )
        yield project_root


@pytest.fixture
def prng_model_builder(temporary_project: Path) -> Iterator[ModelBuilder]:
    module = _load_module(temporary_project / "temporary_model.py")
    sys.modules[module.__name__] = module
    yield module.build_prng_model
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def prng_model_config() -> Mapping[str, Any]:
    return {"dimension": 1, "noise_scale": 0.25}


@pytest.fixture
def prng_model_spec(
    temporary_project: Path,
    prng_model_builder: ModelBuilder,
    prng_model_config: Mapping[str, Any],
) -> ModelSpec:
    return ModelSpec.from_callable(
        prng_model_builder,
        config=prng_model_config,
        project_root=temporary_project,
        version="1",
    )


def test_model_spec_captures_importable_callable(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["module"] == "temporary_model"


def test_model_spec_captures_callable_qualname(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["qualname"] == "build_prng_model"


def test_model_spec_captures_project_relative_source_path(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert payload["callable"]["source_path"] == "temporary_model.py"


def test_model_spec_manifest_is_json_serializable(
    prng_model_spec: ModelSpec,
) -> None:
    payload = prng_model_spec.to_manifest()

    assert isinstance(json.dumps(payload), str)


def test_model_spec_round_trips_through_json(
    prng_model_spec: ModelSpec,
) -> None:
    payload = json.loads(json.dumps(prng_model_spec.to_manifest()))

    assert ModelSpec.from_manifest(payload).to_manifest() == payload


def test_rehydrated_model_matches_raw_callable_evaluation(
    prng_model_builder: ModelBuilder,
    prng_model_config: Mapping[str, Any],
    prng_model_spec: ModelSpec,
) -> None:
    raw_model = prng_model_builder(prng_model_config)
    hydrated_model = prng_model_spec.to_model()

    assert _evaluate_with_seed(hydrated_model, seed=42) == pytest.approx(
        _evaluate_with_seed(raw_model, seed=42)
    )


def test_rehydrated_model_requires_runtime_random_stream(
    prng_model_spec: ModelSpec,
) -> None:
    model = prng_model_spec.to_model()

    with pytest.raises(RuntimeError):
        model(np.array([0.0]), EvaluationContext())


def test_model_spec_rejects_missing_source_file(
    temporary_project: Path,
) -> None:
    spec = ModelSpec(
        callable=PythonCallableSpec(
            module="missing_model",
            qualname="build_model",
            project_root=str(temporary_project),
            source_path="missing_model.py",
        ),
        version="1",
        config={},
    )

    with pytest.raises(ModelError):
        spec.to_model()


def test_model_spec_rejects_missing_qualname(
    temporary_project: Path,
) -> None:
    spec = ModelSpec(
        callable=PythonCallableSpec(
            module="temporary_model",
            qualname="missing_builder",
            project_root=str(temporary_project),
            source_path="temporary_model.py",
        ),
        version="1",
        config={},
    )

    with pytest.raises(ModelError):
        spec.to_model()


def _evaluate_with_seed(model: Model, *, seed: int) -> float:
    stream = RandomStreamSpec(
        algorithm="numpy.pcg64",
        seed=seed,
        stream_id="stream-0",
        schema_version="1",
    )
    context = EvaluationContext(rng=stream.to_generator())
    return model(np.array([0.0]), context)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("temporary_model", path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
