import json

import numpy as np

from distributed_inference import CallableModel, EvaluationContext, InferenceRun
from distributed_inference._validation import FloatArray
from distributed_inference.engines.dummy import DummyInferenceEngine


def _log_density(x: FloatArray, context: EvaluationContext | None) -> float:
    return -0.5 * float(np.dot(x, x))


def test_dummy_engine_evaluates_initial_point() -> None:
    result = DummyInferenceEngine().run_inference(_run(record_evaluations=False))

    assert result.diagnostics["initial_value"] == -2.5


def test_dummy_engine_records_one_evaluation_when_requested() -> None:
    result = DummyInferenceEngine().run_inference(_run(record_evaluations=True))

    assert len(result.evaluations) == 1


def test_dummy_engine_omits_evaluations_when_not_requested() -> None:
    result = DummyInferenceEngine().run_inference(_run(record_evaluations=False))

    assert result.evaluations == ()


def test_dummy_engine_result_payload_is_json_serializable() -> None:
    result = DummyInferenceEngine().run_inference(_run(record_evaluations=True))

    assert isinstance(json.dumps(result.posterior), str)


def _run(*, record_evaluations: bool) -> InferenceRun:
    return InferenceRun(
        model=CallableModel(name="gaussian", dimension=2, fn=_log_density),
        initial_point=np.array([1.0, 2.0]),
        context=EvaluationContext(run_id="run-001"),
        record_evaluations=record_evaluations,
    )
