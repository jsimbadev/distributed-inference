"""Fast deterministic inference engine for local smoke runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from distributed_inference._validation import require_dimension
from distributed_inference.engines.base import (
    EvaluationRecorder,
    InferenceResult,
    InferenceRun,
    ModelEvaluation,
)


@dataclass(frozen=True)
class DummyInferenceResult(InferenceResult[Mapping[str, Any]]):
    """Result returned by the dummy inference engine."""


@dataclass(frozen=True)
class DummyInferenceEngine:
    """Evaluate the model once at the initial point."""

    version: str = "0.1.0"
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the engine name."""
        return "dummy"

    def run_inference(self, run: InferenceRun) -> DummyInferenceResult:
        """Run a single model evaluation and return a serializable posterior."""
        require_dimension(
            run.initial_point,
            run.model.info.dimension,
            name="initial_point",
        )
        recorder = EvaluationRecorder() if run.record_evaluations else None
        value = run.model(run.initial_point, run.context)
        if recorder is not None:
            recorder.record(ModelEvaluation(x=run.initial_point.copy(), value=value))

        posterior = {
            "kind": "point-evaluation",
            "x": run.initial_point.tolist(),
            "log_density": float(value),
        }
        diagnostics = {
            "engine_status": "completed",
            "initial_value": float(value),
            "n_evaluations": 1,
        }
        return DummyInferenceResult(
            engine_name=self.name,
            run=run,
            posterior=posterior,
            diagnostics=diagnostics,
            evaluations=recorder.evaluations if recorder is not None else (),
        )
