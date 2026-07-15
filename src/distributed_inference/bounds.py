"""Optional bounds capability for models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike

from distributed_inference._validation import (
    FloatArray,
    as_vector,
    require_less_equal,
    require_same_shape,
)
from distributed_inference.model import EvaluationContext, Model, ModelInfo


@dataclass(frozen=True)
class Bounds:
    """Box bounds for a model in a specific parameter space."""

    lower: FloatArray
    upper: FloatArray
    plausible_lower: FloatArray | None = None
    plausible_upper: FloatArray | None = None

    def __post_init__(self) -> None:
        lower = as_vector(self.lower, name="lower")
        upper = as_vector(self.upper, name="upper")
        require_same_shape(lower, upper, left_name="lower", right_name="upper")
        require_less_equal(lower, upper, left_name="lower", right_name="upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

        if self.plausible_lower is not None:
            plausible_lower = as_vector(self.plausible_lower, name="plausible_lower")
            require_same_shape(
                lower,
                plausible_lower,
                left_name="lower",
                right_name="plausible_lower",
            )
            require_less_equal(
                lower,
                plausible_lower,
                left_name="lower",
                right_name="plausible_lower",
            )
            object.__setattr__(self, "plausible_lower", plausible_lower)

        if self.plausible_upper is not None:
            plausible_upper = as_vector(self.plausible_upper, name="plausible_upper")
            require_same_shape(
                lower,
                plausible_upper,
                left_name="lower",
                right_name="plausible_upper",
            )
            require_less_equal(
                plausible_upper,
                upper,
                left_name="plausible_upper",
                right_name="upper",
            )
            object.__setattr__(self, "plausible_upper", plausible_upper)

        if self.plausible_lower is not None and self.plausible_upper is not None:
            require_less_equal(
                self.plausible_lower,
                self.plausible_upper,
                left_name="plausible_lower",
                right_name="plausible_upper",
            )


class BoundedModel(Model, Protocol):
    """Model with box bounds in its input space."""

    def bounds(self) -> Bounds | None: ...


@dataclass(frozen=True)
class WithBounds:
    """Attach bounds to any model by composition."""

    model: Model
    model_bounds: Bounds

    @property
    def info(self) -> ModelInfo:
        """Return static model metadata."""
        return self.model.info

    def __call__(
        self,
        x: ArrayLike,
        context: EvaluationContext | None = None,
    ) -> float:
        """Evaluate the wrapped model."""
        return self.model(x, context)

    def bounds(self) -> Bounds:
        """Return bounds in the wrapped model input space."""
        return self.model_bounds


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
