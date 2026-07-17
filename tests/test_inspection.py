import json
from pathlib import Path

from distributed_inference.inspection import format_run_summaries, iter_run_summaries


def test_iter_run_summaries_reads_result_manifest_without_model_import(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "runs" / "gaussian" / "run-001" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(_result_manifest()), encoding="utf-8")

    summaries = iter_run_summaries(tmp_path)

    assert summaries[0].target == "gaussian.full-posterior"


def test_format_run_summaries_includes_run_identity(tmp_path: Path) -> None:
    result_path = tmp_path / "runs" / "gaussian" / "run-001" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(_result_manifest()), encoding="utf-8")

    table = format_run_summaries(iter_run_summaries(tmp_path))

    assert "run-001" in table


def _result_manifest() -> dict[str, object]:
    return {
        "schema_version": "1",
        "identity": {
            "name": "gaussian",
            "run_id": "run-001",
            "attempt_number": 1,
        },
        "model": {
            "kind": "python-callable",
            "version": "1",
            "callable": {
                "module": "missing_model",
                "qualname": "build_model",
                "project_root": "/does/not/exist",
                "source_path": "missing_model.py",
            },
            "config": {},
        },
        "target": {
            "identifier": "gaussian.full-posterior",
            "semantics": "full-posterior",
            "dimension": 2,
            "coordinate_space": "unconstrained",
        },
        "engine": {
            "name": "dummy",
            "version": "0.1.0",
            "config": {},
        },
        "reproducibility": {
            "random_stream": {
                "schema_version": "1",
                "algorithm": "numpy.pcg64",
                "seed": 42,
                "stream_id": "stream-001",
            }
        },
        "status": "completed",
        "timestamps": {
            "started_at": "2026-07-17T09:00:00Z",
            "completed_at": "2026-07-17T09:01:00Z",
        },
        "diagnostics": {},
        "artifacts": {
            "diagnostics": {
                "uri": "artifacts/diagnostics.json",
                "media_type": "application/json",
                "checksum": "sha256:diagnostics",
            }
        },
        "checkpoints": {},
        "context": {
            "run_id": "run-001",
            "metadata": {},
        },
    }
