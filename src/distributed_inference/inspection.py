"""Read-only inspection helpers for persisted inference records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class RunSummary:
    """Summary of a persisted run invocation read from JSON manifests."""

    name: str
    run_id: str
    status: str
    engine: str
    target: str
    artifact_count: int
    checkpoint_count: int
    started_at: str | None
    completed_at: str | None
    path: Path


def iter_run_summaries(root: Path) -> tuple[RunSummary, ...]:
    """Return summaries for persisted invocations without importing model code."""
    runs_root = root / "runs"
    if not runs_root.exists():
        return ()
    return tuple(
        _summary_from_result_manifest(path)
        for path in sorted(runs_root.glob("*/*/result.json"))
    )


def format_run_summaries(summaries: Iterable[RunSummary]) -> str:
    """Format run summaries as a compact plain-text table."""
    rows = [
        [
            "name",
            "run_id",
            "status",
            "engine",
            "target",
            "artifacts",
            "checkpoints",
        ]
    ]
    for summary in summaries:
        rows.append(
            [
                summary.name,
                summary.run_id,
                summary.status,
                summary.engine,
                summary.target,
                str(summary.artifact_count),
                str(summary.checkpoint_count),
            ]
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _summary_from_result_manifest(path: Path) -> RunSummary:
    payload = _read_json(path)
    identity = _mapping(payload["identity"])
    engine = _mapping(payload["engine"])
    target = _mapping(payload["target"])
    timestamps = _mapping(payload.get("timestamps", {}))
    artifacts = _mapping(payload.get("artifacts", {}))
    checkpoints = _mapping(payload.get("checkpoints", {}))
    return RunSummary(
        name=str(identity["name"]),
        run_id=str(identity["run_id"]),
        status=str(payload["status"]),
        engine=str(engine["name"]),
        target=str(target["identifier"]),
        artifact_count=len(artifacts),
        checkpoint_count=len(checkpoints),
        started_at=_optional_str(timestamps.get("started_at")),
        completed_at=_optional_str(timestamps.get("completed_at")),
        path=path,
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = "Expected manifest field to be a JSON object."
        raise ValueError(msg)
    return cast(dict[str, Any], value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
