from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distributed_inference import CallableModel, EvaluationContext, InferenceRun
from distributed_inference._validation import FloatArray
from distributed_inference.engines.base import InferenceResult
from distributed_inference.persistence.artifacts import write_pyvbmc_artifacts
from distributed_inference.persistence.local import LocalInferenceStore


@dataclass(frozen=True)
class FakePyVBMCResult(InferenceResult[object]):
    pass


class FakePosterior:
    def __init__(self) -> None:
        self.D = 2
        self.K = 1
        self.w = np.array([[1.0]])
        self.mu = np.array([[0.0], [0.0]])
        self.sigma = np.array([[1.0]])
        self.lambd = np.array([[1.0], [1.0]])
        self.stats = {"elbo": np.float64(-1.0)}

    def moments(
        self,
        *,
        orig_flag: bool,
        cov_flag: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([[0.0, 0.0]]), np.eye(2)


def test_pyvbmc_artifact_writer_records_posterior_summary(tmp_path: Path) -> None:
    store = LocalInferenceStore(tmp_path)
    result = FakePyVBMCResult(
        engine_name="pyvbmc",
        run=_run(),
        posterior=FakePosterior(),
        diagnostics={"converged": np.bool_(True)},
    )

    artifacts = write_pyvbmc_artifacts(store, result)

    assert "posterior_summary" in artifacts


def test_pyvbmc_posterior_summary_is_json_readable(tmp_path: Path) -> None:
    store = LocalInferenceStore(tmp_path)
    result = FakePyVBMCResult(
        engine_name="pyvbmc",
        run=_run(),
        posterior=FakePosterior(),
        diagnostics={"converged": np.bool_(True)},
    )

    write_pyvbmc_artifacts(store, result)
    payload = store.read_invocation_json(
        "pyvbmc-smoke",
        "run-001",
        "artifacts/posterior-summary.json",
    )

    assert payload["n_components"] == 1


def _run() -> InferenceRun:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return -0.5 * float(np.dot(x, x))

    return InferenceRun(
        name="pyvbmc-smoke",
        model=CallableModel(name="gaussian", dimension=2, fn=log_density),
        initial_point=np.array([0.0, 0.0]),
        context=EvaluationContext(run_id="run-001"),
    )
