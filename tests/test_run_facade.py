import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from distributed_inference import run_inference
from distributed_inference.engines.dummy import DummyInferenceEngine
from distributed_inference.persistence import (
    LocalInferenceStore,
    ModelSpec,
    RandomStreamSpec,
    TargetSpec,
)
from distributed_inference.persistence.models import ModelBuilder
from distributed_inference.persistence.runs import InferenceRunSpec

TEMPORARY_MODEL_SOURCE = """
import numpy as np

from distributed_inference import CallableModel


def build_model(config):
    dimension = int(config["dimension"])

    def log_density(x, context):
        return -0.5 * float(np.dot(x, x))

    return CallableModel(
        name="temporary-gaussian",
        dimension=dimension,
        fn=log_density,
    )
"""


@pytest.fixture
def temporary_project(tmp_path: Path) -> Path:
    (tmp_path / "temporary_model.py").write_text(
        TEMPORARY_MODEL_SOURCE,
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def model_builder(temporary_project: Path) -> Iterator[ModelBuilder]:
    module = _load_module("temporary_model", temporary_project / "temporary_model.py")
    sys.modules[module.__name__] = module
    yield module.build_model
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def run_spec(
    temporary_project: Path,
    model_builder: ModelBuilder,
) -> InferenceRunSpec:
    return InferenceRunSpec(
        name="gaussian-facade",
        run_id="run-001",
        model=ModelSpec.from_callable(
            model_builder,
            config={"dimension": 2},
            project_root=temporary_project,
            version="1",
        ),
        initial_point=[0.0, 0.0],
        random_stream=RandomStreamSpec(
            algorithm="numpy.pcg64",
            seed=42,
            stream_id="stream-001",
            schema_version="1",
        ),
        context_metadata={"target": "full-posterior"},
        record_evaluations=True,
    )


@pytest.fixture
def target() -> TargetSpec:
    return TargetSpec(
        identifier="temporary-gaussian.full-posterior",
        semantics="full-posterior",
        dimension=2,
        coordinate_space="unconstrained",
    )


def test_run_inference_persists_core_manifests(
    tmp_path: Path,
    run_spec: InferenceRunSpec,
    target: TargetSpec,
) -> None:
    persisted = run_inference(
        run=run_spec,
        engine=DummyInferenceEngine(),
        store=LocalInferenceStore(tmp_path),
        target=target,
    )

    assert persisted.files.result_manifest.exists()


def test_run_inference_uses_target_metadata(
    tmp_path: Path,
    run_spec: InferenceRunSpec,
    target: TargetSpec,
) -> None:
    persisted = run_inference(
        run=run_spec,
        engine=DummyInferenceEngine(),
        store=LocalInferenceStore(tmp_path),
        target=target,
    )

    assert persisted.result_manifest.target.identifier == target.identifier


def test_run_inference_records_artifacts(
    tmp_path: Path,
    run_spec: InferenceRunSpec,
    target: TargetSpec,
) -> None:
    persisted = run_inference(
        run=run_spec,
        engine=DummyInferenceEngine(),
        store=LocalInferenceStore(tmp_path),
        target=target,
    )

    assert sorted(persisted.artifacts) == [
        "diagnostics",
        "evaluations",
        "posterior",
    ]


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
