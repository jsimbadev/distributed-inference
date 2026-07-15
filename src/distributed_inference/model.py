"""Core model abstractions for distributed inference workflows."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from distributed_inference._validation import FloatArray, as_vector, require_dimension
from distributed_inference.errors import ModelCapabilityError

LogDensityFn = Callable[[FloatArray, "EvaluationContext | None"], float]
GradientFn = Callable[
    [FloatArray, "EvaluationContext | None"],
    tuple[float, FloatArray],
]


class ParameterSpace(StrEnum):
    """Coordinate space in which a model accepts parameters."""

    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for a model."""

    name: str
    dimension: int
    input_space: ParameterSpace
    supports_gradient: bool = False


@dataclass
class EvaluationContext:
    """Per-evaluation context supplied by runners or engines.

    Models do not own randomness. If stochastic behavior is required, pass an
    RNG through this context for the operation that needs it.
    """

    run_id: str | None = None
    rng: np.random.Generator | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cache: MutableMapping[str, Any] = field(default_factory=dict)


class Model(Protocol):
    """Callable log-density model."""

    @property
    def info(self) -> ModelInfo: ...

    def __call__(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> float: ...


class DifferentiableModel(Model, Protocol):
    """Model with first derivative support."""

    def log_density_and_gradient(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> tuple[float, FloatArray]: ...


@dataclass(frozen=True)
class CallableModel:
    """Concrete model backed by a Python callable."""

    name: str
    dimension: int
    fn: LogDensityFn
    input_space: ParameterSpace = ParameterSpace.UNCONSTRAINED
    gradient_fn: GradientFn | None = None

    @property
    def info(self) -> ModelInfo:
        """Return static model metadata."""
        return ModelInfo(
            name=self.name,
            dimension=self.dimension,
            input_space=self.input_space,
            supports_gradient=self.gradient_fn is not None,
        )

    def __call__(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> float:
        """Evaluate the model log density."""
        vector = self._validate_x(x)
        return float(self.fn(vector, context))

    def log_density_and_gradient(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> tuple[float, FloatArray]:
        """Evaluate log density and gradient when a gradient function exists."""
        if self.gradient_fn is None:
            msg = f"Model {self.name!r} does not provide gradients."
            raise ModelCapabilityError(msg)

        vector = self._validate_x(x)
        value, gradient = self.gradient_fn(vector, context)
        gradient_vector = as_vector(gradient, name="gradient")
        require_dimension(gradient_vector, self.dimension, name="gradient")
        return float(value), gradient_vector

    def _validate_x(self, x: ArrayLike) -> FloatArray:
        vector = as_vector(x, name="x")
        require_dimension(vector, self.dimension, name="x")
        return vector
