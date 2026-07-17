"""Engine-neutral inference run and result abstractions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from distributed_inference._validation import FloatArray
from distributed_inference.model import EvaluationContext, Model

PosteriorT = TypeVar("PosteriorT")


@dataclass(frozen=True)
class ModelEvaluation:
    """A single model evaluation recorded during an inference run."""

    x: FloatArray
    value: float


@dataclass
class EvaluationRecorder:
    """In-memory store for model evaluations."""

    _evaluations: list[ModelEvaluation] = field(default_factory=list)

    def record(self, evaluation: ModelEvaluation) -> None:
        """Append a model evaluation."""
        self._evaluations.append(evaluation)

    @property
    def evaluations(self) -> tuple[ModelEvaluation, ...]:
        """Return recorded evaluations as an immutable tuple."""
        return tuple(self._evaluations)


@dataclass(frozen=True)
class InferenceRun:
    """Input data required to run an inference engine."""

    name: str
    model: Model
    initial_point: FloatArray
    context: EvaluationContext | None = None
    record_evaluations: bool = False


@dataclass(frozen=True)
class InferenceResult(Generic[PosteriorT]):
    """Engine-neutral result returned by an inference engine."""

    engine_name: str
    run: InferenceRun
    posterior: PosteriorT
    diagnostics: Mapping[str, Any]
    evaluations: tuple[ModelEvaluation, ...] = ()


class InferenceEngine(Protocol):
    """Inference engine protocol."""

    @property
    def name(self) -> str: ...

    def run_inference(self, run: InferenceRun) -> InferenceResult[Any]: ...
