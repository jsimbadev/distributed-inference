"""Tools for repeated inference runs on local and distributed infrastructure."""

from distributed_inference.bounds import (
    BoundedModel,
    Bounds,
    WithBounds,
    bounds_mapping,
)
from distributed_inference.errors import (
    DistributedInferenceError,
    ModelCapabilityError,
    ModelError,
)
from distributed_inference.model import (
    CallableModel,
    DifferentiableModel,
    EvaluationContext,
    Model,
    ModelInfo,
    ParameterSpace,
)
from distributed_inference.transforms import (
    ParameterTransform,
    TransformedModel,
)

__all__ = [
    "BoundedModel",
    "Bounds",
    "CallableModel",
    "DifferentiableModel",
    "DistributedInferenceError",
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
    "bounds_mapping",
]

__version__ = "0.1.0"
