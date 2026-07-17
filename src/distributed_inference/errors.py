"""Error types for distributed inference."""

from __future__ import annotations

from typing import Any


class DistributedInferenceError(ValueError):
    """Base error raised by distributed inference."""


class ModelError(DistributedInferenceError):
    """Base error raised by model abstractions."""


class ModelCapabilityError(ModelError):
    """Raised when a model does not support a requested capability."""


class ManifestError(DistributedInferenceError):
    """Raised when a persisted manifest cannot be constructed."""


class ExecutionError(DistributedInferenceError):
    """Raised when an execution backend cannot complete a run.

    The optional record contains failed-run provenance when a backend can
    construct it before raising.
    """

    def __init__(self, message: str, *, record: Any | None = None) -> None:
        super().__init__(message)
        self.record = record
