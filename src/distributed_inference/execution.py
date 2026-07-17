"""Execution backend abstractions and local execution provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from distributed_inference.engines.base import (
    InferenceEngine,
    InferenceResult,
    InferenceRun,
)
from distributed_inference.errors import ExecutionError, ManifestError


@dataclass(frozen=True)
class ExecutionSpec:
    """Serializable description of an execution backend configuration."""

    backend: str
    backend_version: str
    config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable execution backend manifest."""
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "config": dict(self.config),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ExecutionSpec:
        """Rehydrate an execution backend spec from a manifest payload."""
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            msg = "Execution spec config must be a mapping."
            raise ManifestError(msg)
        return cls(
            schema_version=str(payload["schema_version"]),
            backend=str(payload["backend"]),
            backend_version=str(payload["backend_version"]),
            config=dict(config),
        )


@dataclass(frozen=True)
class ExecutionIdentity:
    """Serializable identity for a backend execution of a run."""

    name: str
    run_id: str

    def to_manifest(self) -> dict[str, str]:
        """Return a serializable execution identity."""
        return {"name": self.name, "run_id": self.run_id}

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ExecutionIdentity:
        """Rehydrate an execution identity from a manifest payload."""
        return cls(name=str(payload["name"]), run_id=str(payload["run_id"]))


@dataclass(frozen=True)
class ExecutionAttempt:
    """Serializable provenance for one infrastructure execution attempt."""

    attempt_number: int
    started_at: str | None = None
    completed_at: str | None = None

    def to_manifest(self) -> dict[str, int | str | None]:
        """Return a serializable execution attempt."""
        return {
            "attempt_number": self.attempt_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ExecutionAttempt:
        """Rehydrate an execution attempt from a manifest payload."""
        return cls(
            attempt_number=int(payload["attempt_number"]),
            started_at=_optional_str(payload.get("started_at")),
            completed_at=_optional_str(payload.get("completed_at")),
        )


@dataclass(frozen=True)
class ExecutionRecord:
    """Serializable provenance produced by an execution backend."""

    spec: ExecutionSpec
    identity: ExecutionIdentity
    attempt: ExecutionAttempt
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1"

    def to_manifest(self) -> dict[str, Any]:
        """Return a serializable execution manifest."""
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_manifest(),
            "backend": self.spec.to_manifest(),
            "attempt": self.attempt.to_manifest(),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> ExecutionRecord:
        """Rehydrate an execution record from a manifest payload."""
        identity = payload["identity"]
        backend = payload["backend"]
        attempt = payload["attempt"]
        metadata = payload.get("metadata", {})
        for name, value in {
            "identity": identity,
            "backend": backend,
            "attempt": attempt,
            "metadata": metadata,
        }.items():
            if not isinstance(value, Mapping):
                msg = f"Execution manifest {name} must be a mapping."
                raise ManifestError(msg)
        return cls(
            schema_version=str(payload["schema_version"]),
            identity=ExecutionIdentity.from_manifest(identity),
            spec=ExecutionSpec.from_manifest(backend),
            attempt=ExecutionAttempt.from_manifest(attempt),
            status=str(payload["status"]),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ExecutedInference:
    """In-memory result plus the process-independent execution provenance."""

    result: InferenceResult[Any]
    execution: ExecutionRecord


class ExecutionBackend(Protocol):
    """Backend protocol for executing an inference engine against a run."""

    @property
    def name(self) -> str: ...

    def execute(
        self,
        run: InferenceRun,
        engine: InferenceEngine,
        *,
        attempt_number: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutedInference: ...


@dataclass(frozen=True)
class LocalExecutionBackend:
    """Synchronous in-process execution backend."""

    version: str = "0.1.0"
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "local"

    def execute(
        self,
        run: InferenceRun,
        engine: InferenceEngine,
        *,
        attempt_number: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutedInference:
        """Execute an inference run synchronously in the current process."""
        _require_positive_attempt_number(attempt_number)
        started_at = _utc_timestamp()
        record_metadata = dict(metadata or {})
        try:
            result = engine.run_inference(run)
        except Exception as exc:
            completed_at = _utc_timestamp()
            failed_record = self._record(
                run=run,
                attempt_number=attempt_number,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                metadata={
                    **record_metadata,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            raise ExecutionError(
                "Local execution failed.",
                record=failed_record,
            ) from exc

        completed_at = _utc_timestamp()
        record = self._record(
            run=run,
            attempt_number=attempt_number,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            metadata=record_metadata,
        )
        return ExecutedInference(result=result, execution=record)

    def _record(
        self,
        *,
        run: InferenceRun,
        attempt_number: int,
        status: str,
        started_at: str,
        completed_at: str,
        metadata: Mapping[str, Any],
    ) -> ExecutionRecord:
        return ExecutionRecord(
            spec=ExecutionSpec(
                backend=self.name,
                backend_version=self.version,
                config=dict(self.config),
            ),
            identity=ExecutionIdentity(name=run.name, run_id=_run_id(run)),
            attempt=ExecutionAttempt(
                attempt_number=attempt_number,
                started_at=started_at,
                completed_at=completed_at,
            ),
            status=status,
            metadata=dict(metadata),
        )


def _run_id(run: InferenceRun) -> str:
    if run.context is not None and run.context.run_id:
        return run.context.run_id
    return str(uuid4())


def _require_positive_attempt_number(attempt_number: int) -> None:
    if attempt_number < 1:
        msg = "attempt_number must be positive."
        raise ValueError(msg)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
