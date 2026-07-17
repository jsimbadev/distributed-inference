"""Tools for repeated inference runs on local and distributed infrastructure."""

from distributed_inference.bounds import (
    BoundedModel,
    Bounds,
    WithBounds,
    bounds_mapping,
)
from distributed_inference.engines import (
    DummyInferenceEngine,
    DummyInferenceResult,
    EvaluationRecorder,
    InferenceEngine,
    InferenceResult,
    InferenceRun,
    ModelEvaluation,
)
from distributed_inference.errors import (
    DistributedInferenceError,
    ExecutionError,
    ManifestError,
    ModelCapabilityError,
    ModelError,
)
from distributed_inference.execution import (
    ExecutedInference,
    ExecutionAttempt,
    ExecutionBackend,
    ExecutionIdentity,
    ExecutionRecord,
    ExecutionSpec,
    LocalExecutionBackend,
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
    "DummyInferenceEngine",
    "DummyInferenceResult",
    "EvaluationContext",
    "EvaluationRecorder",
    "ExecutedInference",
    "ExecutionAttempt",
    "ExecutionBackend",
    "ExecutionError",
    "ExecutionIdentity",
    "ExecutionRecord",
    "ExecutionSpec",
    "InferenceEngine",
    "InferenceResult",
    "InferenceRun",
    "LocalExecutionBackend",
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
