"""Model abstractions for distributed inference workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
LogDensityFn = Callable[[FloatArray, "EvaluationContext | None"], float]
GradientFn = Callable[
    [FloatArray, "EvaluationContext | None"],
    tuple[float, FloatArray],
]


class ModelError(ValueError):
    """Base error raised by model abstractions."""


class ModelCapabilityError(ModelError):
    """Raised when a model does not support a requested capability."""


class ParameterSpace(StrEnum):
    """Coordinate space in which a model accepts parameters."""

    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"


@dataclass(frozen=True)
class Bounds:
    """Box bounds for a model in a specific parameter space."""

    lower: FloatArray
    upper: FloatArray
    plausible_lower: FloatArray | None = None
    plausible_upper: FloatArray | None = None

    def __post_init__(self) -> None:
        lower = _as_vector(self.lower, name="lower")
        upper = _as_vector(self.upper, name="upper")
        _require_same_shape(lower, upper, left_name="lower", right_name="upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

        if self.plausible_lower is not None:
            plausible_lower = _as_vector(self.plausible_lower, name="plausible_lower")
            _require_same_shape(
                lower,
                plausible_lower,
                left_name="lower",
                right_name="plausible_lower",
            )
            object.__setattr__(self, "plausible_lower", plausible_lower)

        if self.plausible_upper is not None:
            plausible_upper = _as_vector(self.plausible_upper, name="plausible_upper")
            _require_same_shape(
                lower,
                plausible_upper,
                left_name="lower",
                right_name="plausible_upper",
            )
            object.__setattr__(self, "plausible_upper", plausible_upper)


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

    def bounds(self) -> Bounds | None: ...


class DifferentiableModel(Model, Protocol):
    """Model with first derivative support."""

    def log_density_and_gradient(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> tuple[float, FloatArray]: ...


class ParameterTransform(Protocol):
    """Transform between constrained and unconstrained parameter spaces."""

    @property
    def constrained_dimension(self) -> int: ...

    @property
    def unconstrained_dimension(self) -> int: ...

    def to_unconstrained(self, x: ArrayLike) -> FloatArray: ...

    def to_constrained(self, z: ArrayLike) -> FloatArray: ...

    def log_abs_det_jacobian(self, z: ArrayLike) -> float: ...


@dataclass(frozen=True)
class CallableModel:
    """Concrete model backed by a Python callable."""

    name: str
    dimension: int
    fn: LogDensityFn
    input_space: ParameterSpace = ParameterSpace.UNCONSTRAINED
    model_bounds: Bounds | None = None
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

    def bounds(self) -> Bounds | None:
        """Return model bounds in the model input space."""
        return self.model_bounds

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
        gradient_vector = _as_vector(gradient, name="gradient")
        _require_dimension(gradient_vector, self.dimension, name="gradient")
        return float(value), gradient_vector

    def _validate_x(self, x: ArrayLike) -> FloatArray:
        vector = _as_vector(x, name="x")
        _require_dimension(vector, self.dimension, name="x")
        return vector


@dataclass(frozen=True)
class TransformedModel:
    """Expose a constrained-space model in unconstrained coordinates."""

    base_model: Model
    transform: ParameterTransform
    model_bounds: Bounds | None = None

    @property
    def info(self) -> ModelInfo:
        """Return static model metadata for the transformed model."""
        return ModelInfo(
            name=f"{self.base_model.info.name}.unconstrained",
            dimension=self.transform.unconstrained_dimension,
            input_space=ParameterSpace.UNCONSTRAINED,
            supports_gradient=False,
        )

    def __call__(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> float:
        """Evaluate the transformed log density with Jacobian correction."""
        z = _as_vector(x, name="x")
        _require_dimension(z, self.info.dimension, name="x")
        constrained = self.transform.to_constrained(z)
        value = self.base_model(constrained, context)
        return float(value + self.transform.log_abs_det_jacobian(z))

    def bounds(self) -> Bounds | None:
        """Return bounds in the transformed model input space."""
        return self.model_bounds


def _as_vector(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be a one-dimensional array."
        raise ModelError(msg)
    return array


def _require_same_shape(
    left: FloatArray,
    right: FloatArray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        msg = f"{left_name} and {right_name} must have matching shapes."
        raise ModelError(msg)


def _require_dimension(value: FloatArray, dimension: int, *, name: str) -> None:
    if value.shape != (dimension,):
        msg = f"{name} must have dimension {dimension}; got {value.shape[0]}."
        raise ModelError(msg)


def bounds_mapping(bounds: Bounds | None) -> Mapping[str, FloatArray | None]:
    """Return a serializable mapping for display and diagnostics."""
    if bounds is None:
        return {
            "lower": None,
            "upper": None,
            "plausible_lower": None,
            "plausible_upper": None,
        }
    return {
        "lower": bounds.lower,
        "upper": bounds.upper,
        "plausible_lower": bounds.plausible_lower,
        "plausible_upper": bounds.plausible_upper,
    }
