"""Tools for repeated inference runs on local and distributed infrastructure."""

from distributed_inference.bounds import (
    BoundedModel,
    Bounds,
    WithBounds,
    bounds_mapping,
)
from distributed_inference.engines import (
    EvaluationRecorder,
    InferenceEngine,
    InferenceResult,
    InferenceRun,
    ModelEvaluation,
)
from distributed_inference.errors import (
    DistributedInferenceError,
    ManifestError,
    ModelCapabilityError,
    ModelError,
)
from distributed_inference.model import (
    CallableDifferentiableModel,
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
    "CallableDifferentiableModel",
    "CallableModel",
    "DifferentiableModel",
    "DistributedInferenceError",
    "EvaluationContext",
    "EvaluationRecorder",
    "InferenceEngine",
    "InferenceResult",
    "InferenceRun",
    "ManifestError",
    "Model",
    "ModelCapabilityError",
    "ModelError",
    "ModelEvaluation",
    "ModelInfo",
    "ParameterSpace",
    "ParameterTransform",
    "TransformedModel",
    "WithBounds",
    "__version__",
    "bounds_mapping",
]

__version__ = "0.1.0"
