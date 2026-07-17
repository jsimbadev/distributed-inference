"""Local filesystem persistence for inference manifests and JSON artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distributed_inference.engines.base import InferenceResult, ModelEvaluation
from distributed_inference.errors import ManifestError
from distributed_inference.execution import ExecutionRecord
from distributed_inference.persistence.manifests import (
    ArtifactReference,
    ResultManifest,
)
from distributed_inference.persistence.runs import InferenceRunSpec


@dataclass(frozen=True)
class PersistedInferenceFiles:
    """Paths written for one locally persisted inference run."""

    run_manifest: Path
    result_manifest: Path
    execution_manifest: Path


@dataclass(frozen=True)
class LocalInferenceStore:
    """Write process-independent inference records to a local directory."""

    root: Path

    def write_run(self, run: InferenceRunSpec) -> Path:
        """Write an inference-run manifest."""
        return self._write_json("run.json", run.to_manifest())

    def write_result(self, result: ResultManifest) -> Path:
        """Write a result manifest."""
        return self._write_json("result.json", result.to_manifest())

    def write_execution(self, execution: ExecutionRecord) -> Path:
        """Write an execution manifest."""
        return self._write_json("execution.json", execution.to_manifest())

    def write_inference(
        self,
        *,
        run: InferenceRunSpec,
        result: ResultManifest,
        execution: ExecutionRecord,
    ) -> PersistedInferenceFiles:
        """Write run, result, and execution manifests for one inference run."""
        return PersistedInferenceFiles(
            run_manifest=self.write_run(run),
            result_manifest=self.write_result(result),
            execution_manifest=self.write_execution(execution),
        )

    def write_result_artifacts(
        self,
        result: InferenceResult[Any],
    ) -> dict[str, ArtifactReference]:
        """Write generic JSON artifacts for a completed result."""
        artifacts = {
            "posterior": self.write_json_artifact(
                "artifacts/posterior.json",
                result.posterior,
            ),
            "diagnostics": self.write_json_artifact(
                "artifacts/diagnostics.json",
                dict(result.diagnostics),
            ),
        }
        if result.evaluations:
            artifacts["evaluations"] = self.write_json_artifact(
                "artifacts/evaluations.json",
                _evaluations_payload(result.evaluations),
            )
        return artifacts

    def write_json_artifact(
        self,
        relative_path: str,
        payload: Any,
        *,
        media_type: str = "application/json",
    ) -> ArtifactReference:
        """Write a JSON artifact and return its manifest reference."""
        data = _json_bytes(payload)
        self._write_bytes(relative_path, data)
        return ArtifactReference(
            uri=relative_path,
            media_type=media_type,
            checksum=_sha256(data),
        )

    def read_json(self, relative_path: str) -> Any:
        """Read a JSON payload from the store."""
        path = self.root / relative_path
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_json(self, relative_path: str, payload: Any) -> Path:
        return self._write_bytes(relative_path, _json_bytes(payload))

    def _write_bytes(self, relative_path: str, data: bytes) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


def _evaluations_payload(
    evaluations: tuple[ModelEvaluation, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "x": evaluation.x.tolist(),
            "value": float(evaluation.value),
        }
        for evaluation in evaluations
    ]


def _json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    except TypeError as exc:
        msg = "Persistence payload contains non-serializable values."
        raise ManifestError(msg) from exc


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
