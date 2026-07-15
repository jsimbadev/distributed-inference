"""Parameter-space transform abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from distributed_inference._validation import FloatArray, as_vector, require_dimension
from distributed_inference.model import (
    EvaluationContext,
    Model,
    ModelInfo,
    ParameterSpace,
)


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
class TransformedModel:
    """Expose a constrained-space model in unconstrained coordinates."""

    base_model: Model
    transform: ParameterTransform

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
        z = as_vector(x, name="x")
        require_dimension(z, self.info.dimension, name="x")
        constrained = self.transform.to_constrained(z)
        value = self.base_model(constrained, context)
        return float(value + self.transform.log_abs_det_jacobian(z))
