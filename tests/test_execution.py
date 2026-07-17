import json
from typing import cast

import numpy as np
import pytest

from distributed_inference import (
    CallableModel,
    EvaluationContext,
    ExecutionError,
    InferenceRun,
    LocalExecutionBackend,
)
from distributed_inference._validation import FloatArray
from distributed_inference.engines.dummy import DummyInferenceEngine
from distributed_inference.execution import ExecutionRecord


def _log_density(x: FloatArray, context: EvaluationContext | None) -> float:
    return -0.5 * float(np.dot(x, x))


def test_local_execution_returns_result_provenance() -> None:
    executed = LocalExecutionBackend().execute(
        _run(),
        DummyInferenceEngine(),
        attempt_id="attempt-000",
    )

    assert executed.execution.attempt.attempt_id == "attempt-000"


def test_local_execution_record_is_json_serializable() -> None:
    executed = LocalExecutionBackend().execute(
        _run(),
        DummyInferenceEngine(),
        attempt_id="attempt-000",
    )

    assert isinstance(json.dumps(executed.execution.to_manifest()), str)


def test_local_execution_record_round_trips_through_json() -> None:
    executed = LocalExecutionBackend().execute(
        _run(),
        DummyInferenceEngine(),
        attempt_id="attempt-000",
    )
    payload = json.loads(json.dumps(executed.execution.to_manifest()))

    assert ExecutionRecord.from_manifest(payload).to_manifest() == payload


def test_local_execution_failure_exposes_failed_record() -> None:
    with pytest.raises(ExecutionError) as error:
        LocalExecutionBackend().execute(
            _failing_run(),
            DummyInferenceEngine(),
            attempt_id="attempt-000",
        )

    assert cast(ExecutionRecord, error.value.record).status == "failed"


def test_local_execution_metadata_does_not_contain_backend_object() -> None:
    executed = LocalExecutionBackend().execute(
        _run(),
        DummyInferenceEngine(),
        attempt_id="attempt-000",
    )

    assert "backend" not in executed.execution.metadata


def _run() -> InferenceRun:
    return InferenceRun(
        model=CallableModel(name="gaussian", dimension=2, fn=_log_density),
        initial_point=np.array([1.0, 2.0]),
        context=EvaluationContext(run_id="run-001"),
    )


def _failing_run() -> InferenceRun:
    def fail(x: FloatArray, context: EvaluationContext | None) -> float:
        raise RuntimeError("model failed")

    return InferenceRun(
        model=CallableModel(name="failing", dimension=2, fn=fail),
        initial_point=np.array([1.0, 2.0]),
        context=EvaluationContext(run_id="run-001"),
    )
