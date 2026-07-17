"""Error types for distributed inference."""


class DistributedInferenceError(ValueError):
    """Base error raised by distributed inference."""


class ModelError(DistributedInferenceError):
    """Base error raised by model abstractions."""


class ModelCapabilityError(ModelError):
    """Raised when a model does not support a requested capability."""


class ManifestError(DistributedInferenceError):
    """Raised when a persisted manifest cannot be constructed."""
