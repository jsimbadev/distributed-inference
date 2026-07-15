"""Tools for repeated inference runs on local and distributed infrastructure."""

from distributed_inference.model import (
    BoundedModel,
    Bounds,
    CallableModel,
    DifferentiableModel,
    EvaluationContext,
    Model,
    ModelCapabilityError,
    ModelError,
    ModelInfo,
    ParameterSpace,
    ParameterTransform,
    TransformedModel,
    WithBounds,
)

__all__ = [
    "BoundedModel",
    "Bounds",
    "CallableModel",
    "DifferentiableModel",
    "EvaluationContext",
    "Model",
    "ModelCapabilityError",
    "ModelError",
    "ModelInfo",
    "ParameterSpace",
    "ParameterTransform",
    "TransformedModel",
    "WithBounds",
    "__version__",
]

__version__ = "0.1.0"
