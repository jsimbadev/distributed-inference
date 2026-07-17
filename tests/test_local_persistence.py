import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from distributed_inference.engines.dummy import DummyInferenceEngine
from distributed_inference.execution import LocalExecutionBackend
from distributed_inference.persistence.local import LocalInferenceStore
from distributed_inference.persistence.manifests import (
    EngineSpec,
    ResultManifest,
    ResultManifestMetadata,
    TargetSpec,
)
from distributed_inference.persistence.models import ModelBuilder, ModelSpec
from distributed_inference.persistence.random_streams import RandomStreamSpec
from distributed_inference.persistence.runs import InferenceRunSpec

TEMPORARY_LOCAL_MODEL_SOURCE = """
import numpy as np


from distributed_inference import CallableModel


def build_gaussian_model(config):
    dimension = int(config["dimension"])

    def log_density(x, context):
        return -0.5 * float(np.dot(x, x))

    return CallableModel(
        name="temporary-gaussian",
        dimension=dimension,
        fn=log_density,
    )
"""


@dataclass(frozen=True)
class LocalSmokeRun:
    store: LocalInferenceStore
    result_manifest: ResultManifest
    run_spec: InferenceRunSpec


@pytest.fixture
def temporary_project(tmp_path: Path) -> Path:
    (tmp_path / "temporary_local_model.py").write_text(
        TEMPORARY_LOCAL_MODEL_SOURCE,
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def gaussian_model_builder(temporary_project: Path) -> Iterator[ModelBuilder]:
    module = _load_module(
        "temporary_local_model",
        temporary_project / "temporary_local_model.py",
    )
    sys.modules[module.__name__] = module
    yield module.build_gaussian_model
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def run_spec(
    temporary_project: Path,
    gaussian_model_builder: ModelBuilder,
) -> InferenceRunSpec:
    return InferenceRunSpec(
        name="local-dummy-smoke",
        run_id="run-001",
        replicate_id="replicate-000",
        attempt_id="attempt-000",
        model=ModelSpec.from_callable(
            gaussian_model_builder,
            config={"dimension": 2},
            project_root=temporary_project,
            version="1",
        ),
        initial_point=[1.0, 2.0],
        random_stream=RandomStreamSpec(
            algorithm="numpy.pcg64",
            seed=42,
            stream_id="replicate-000",
            schema_version="1",
        ),
        context_metadata={"target": "full-posterior"},
        record_evaluations=True,
    )


@pytest.fixture
def local_smoke_run(tmp_path: Path, run_spec: InferenceRunSpec) -> LocalSmokeRun:
    store = LocalInferenceStore(tmp_path)
    engine = DummyInferenceEngine()
    run = run_spec.to_inference_run()
    executed = LocalExecutionBackend().execute(
        run,
        engine,
        attempt_id=run_spec.attempt_id,
    )
    artifacts = store.write_result_artifacts(executed.result)
    checkpoint = store.write_json_artifact(
        "checkpoints/dummy-state.json",
        {"kind": "dummy-checkpoint", "state": {}},
    )
    result_manifest = ResultManifest.from_result(
        executed.result,
        ResultManifestMetadata(
            schema_version="1",
            workflow_id="workflow-001",
            replicate_id=run_spec.replicate_id,
            attempt_id=run_spec.attempt_id,
            model=run_spec.model,
            target=TargetSpec(
                identifier="temporary-gaussian.full-posterior",
                semantics="full-posterior",
                dimension=2,
                coordinate_space="unconstrained",
            ),
            engine=EngineSpec(
                name=engine.name,
                version=engine.version,
                config=engine.config,
            ),
            random_stream=run_spec.random_stream,
            artifacts=artifacts,
            checkpoints={"engine_state": checkpoint},
            status="completed",
            started_at=executed.execution.attempt.started_at,
            completed_at=executed.execution.attempt.completed_at,
        ),
    )
    store.write_inference(
        run=run_spec,
        result=result_manifest,
        execution=executed.execution,
    )
    return LocalSmokeRun(
        store=store,
        result_manifest=result_manifest,
        run_spec=run_spec,
    )


def test_local_store_persists_core_manifests(
    local_smoke_run: LocalSmokeRun,
) -> None:
    files = {path.name for path in local_smoke_run.store.root.glob("*.json")}

    assert files == {"execution.json", "result.json", "run.json"}


def test_local_store_result_manifest_references_posterior_artifact(
    local_smoke_run: LocalSmokeRun,
) -> None:
    payload = local_smoke_run.store.read_json("result.json")

    assert payload["artifacts"]["posterior"]["uri"] == "artifacts/posterior.json"


def test_local_store_result_manifest_references_checkpoint_separately(
    local_smoke_run: LocalSmokeRun,
) -> None:
    payload = local_smoke_run.store.read_json("result.json")

    assert payload["checkpoints"]["engine_state"]["uri"] == (
        "checkpoints/dummy-state.json"
    )


def test_local_store_result_manifest_does_not_serialize_runtime_cache(
    local_smoke_run: LocalSmokeRun,
) -> None:
    payload = local_smoke_run.store.read_json("result.json")

    assert "cache" not in payload["context"]


def test_local_store_result_manifest_round_trips(
    local_smoke_run: LocalSmokeRun,
) -> None:
    payload = json.loads(json.dumps(local_smoke_run.store.read_json("result.json")))

    assert ResultManifest.from_manifest(payload).to_manifest() == payload


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {path}."
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
