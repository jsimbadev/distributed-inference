"""Artifact writers for engine-neutral and engine-specific inference results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from distributed_inference.engines.base import InferenceResult
from distributed_inference.errors import ManifestError
from distributed_inference.persistence.local import LocalInferenceStore
from distributed_inference.persistence.manifests import ArtifactReference


def write_result_artifacts(
    store: LocalInferenceStore,
    result: InferenceResult[Any],
) -> dict[str, ArtifactReference]:
    """Write result artifacts using an engine-specific adapter when available."""
    if result.engine_name == "pyvbmc":
        return write_pyvbmc_artifacts(store, result)
    return store.write_result_artifacts(result)


def write_pyvbmc_artifacts(
    store: LocalInferenceStore,
    result: InferenceResult[Any],
) -> dict[str, ArtifactReference]:
    """Write JSON artifacts for a PyVBMC result.

    The variational posterior object itself is a live Python object and PyVBMC's
    native persistence uses pickle-like semantics. This adapter records a
    process-independent JSON summary and leaves exact posterior serialization as
    an explicit future engine artifact decision.
    """
    run_id = _require_run_id(result)
    return {
        "posterior_summary": store.write_invocation_json_artifact(
            "artifacts/posterior-summary.json",
            _pyvbmc_posterior_summary(result.posterior),
            name=result.run.name,
            run_id=run_id,
        ),
        "diagnostics": store.write_invocation_json_artifact(
            "artifacts/diagnostics.json",
            dict(result.diagnostics),
            name=result.run.name,
            run_id=run_id,
        ),
        **_evaluation_artifact(store, result, run_id),
    }


def _pyvbmc_posterior_summary(posterior: object) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": "pyvbmc-variational-posterior-summary",
        "python_type": f"{type(posterior).__module__}.{type(posterior).__qualname__}",
        "serialization": "summary-only",
    }
    for manifest_name, attribute in {
        "dimension": "D",
        "n_components": "K",
        "weights": "w",
        "component_means": "mu",
        "component_scales": "sigma",
        "dimension_scales": "lambd",
        "stats": "stats",
    }.items():
        if hasattr(posterior, attribute):
            summary[manifest_name] = getattr(posterior, attribute)
    moments = _posterior_moments(posterior)
    if moments:
        summary["moments_unconstrained"] = moments
    return summary


def _posterior_moments(posterior: object) -> dict[str, Any]:
    moments = getattr(posterior, "moments", None)
    if not callable(moments):
        return {}
    try:
        mean, covariance = moments(orig_flag=False, cov_flag=True)
    except Exception:
        return {}
    return {
        "mean": mean,
        "covariance": covariance,
    }


def _evaluation_artifact(
    store: LocalInferenceStore,
    result: InferenceResult[Any],
    run_id: str,
) -> Mapping[str, ArtifactReference]:
    if not result.evaluations:
        return {}
    return {
        "evaluations": store.write_invocation_json_artifact(
            "artifacts/evaluations.json",
            [
                {"x": evaluation.x, "value": evaluation.value}
                for evaluation in result.evaluations
            ],
            name=result.run.name,
            run_id=run_id,
        )
    }


def _require_run_id(result: InferenceResult[Any]) -> str:
    if result.run.context is None or not result.run.context.run_id:
        msg = "Cannot persist result artifacts for a run without a run_id."
        raise ManifestError(msg)
    return result.run.context.run_id
