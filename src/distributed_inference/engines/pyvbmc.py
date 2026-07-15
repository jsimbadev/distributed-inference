"""PyVBMC inference engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from distributed_inference._validation import (
    FloatArray,
    as_vector,
    require_dimension,
    require_less_equal,
)
from distributed_inference.bounds import Bounds
from distributed_inference.engines.base import (
    EvaluationRecorder,
    InferenceResult,
    InferenceRun,
    ModelEvaluation,
)
from distributed_inference.errors import ModelError
from distributed_inference.model import (
    EvaluationContext,
    Model,
    ParameterSpace,
)


class _PyVBMCInstance(Protocol):
    def optimize(self) -> tuple[Any, Mapping[str, Any]]: ...


type PyVBMCFactory = Callable[..., _PyVBMCInstance]


@dataclass(frozen=True)
class PyVBMCOptions:
    """Options passed to the PyVBMC engine."""

    raw_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PyVBMCResult(InferenceResult[Any]):
    """Result returned by the PyVBMC engine."""


@dataclass(frozen=True)
class PyVBMCEngine:
    """Run a bounded model with PyVBMC."""

    options: PyVBMCOptions = field(default_factory=PyVBMCOptions)
    vbmc_factory: PyVBMCFactory | None = None

    @property
    def name(self) -> str:
        """Return the engine name."""
        return "pyvbmc"

    def run(
        self,
        model: Model,
        initial_point: ArrayLike,
        *,
        context: EvaluationContext | None = None,
        record_evaluations: bool = False,
    ) -> PyVBMCResult:
        """Run PyVBMC on a Distributed Inference model."""
        x0 = as_vector(initial_point, name="initial_point")
        run = InferenceRun(
            model=model,
            initial_point=x0,
            context=context,
            record_evaluations=record_evaluations,
        )
        return self.run_inference(run)

    def run_inference(self, run: InferenceRun) -> PyVBMCResult:
        """Run PyVBMC from an engine-neutral inference run."""
        _require_unconstrained(run.model)
        require_dimension(
            run.initial_point,
            run.model.info.dimension,
            name="initial_point",
        )
        bounds = _require_bounds(run.model)
        _require_initial_point_in_bounds(run.initial_point, bounds)
        _require_plausible_bounds(bounds)

        recorder = EvaluationRecorder() if run.record_evaluations else None
        log_density = _to_pyvbmc_log_density(run.model, run.context, recorder)
        vbmc = self._vbmc_factory()(
            log_density,
            x0=run.initial_point,
            lower_bounds=bounds.lower,
            upper_bounds=bounds.upper,
            plausible_lower_bounds=bounds.plausible_lower,
            plausible_upper_bounds=bounds.plausible_upper,
            options=dict(self.options.raw_options),
        )
        posterior, diagnostics = vbmc.optimize()

        return PyVBMCResult(
            engine_name=self.name,
            run=run,
            posterior=posterior,
            diagnostics=diagnostics,
            evaluations=recorder.evaluations if recorder is not None else (),
        )

    def _vbmc_factory(self) -> PyVBMCFactory:
        if self.vbmc_factory is not None:
            return self.vbmc_factory

        from pyvbmc import VBMC

        return VBMC


def _to_pyvbmc_log_density(
    model: Model,
    context: EvaluationContext | None,
    recorder: EvaluationRecorder | None,
) -> Callable[[ArrayLike], float]:
    def log_density(x: ArrayLike) -> float:
        vector = np.asarray(x, dtype=np.float64)
        value = model(vector, context)
        if recorder is not None:
            recorder.record(ModelEvaluation(x=vector.copy(), value=value))
        return value

    return log_density


def _require_bounds(model: Model) -> Bounds:
    bounds_method = getattr(model, "bounds", None)
    if not callable(bounds_method):
        msg = f"Model {model.info.name!r} does not define PyVBMC bounds."
        raise ModelError(msg)

    bounds = bounds_method()
    if bounds is None:
        msg = f"Model {model.info.name!r} does not define PyVBMC bounds."
        raise ModelError(msg)
    return bounds


def _require_unconstrained(model: Model) -> None:
    if model.info.input_space != ParameterSpace.UNCONSTRAINED:
        msg = (
            f"Model {model.info.name!r} uses {model.info.input_space.value} input "
            "space; PyVBMC requires an unconstrained model."
        )
        raise ModelError(msg)


def _require_initial_point_in_bounds(x0: FloatArray, bounds: Bounds) -> None:
    require_dimension(bounds.lower, x0.shape[0], name="lower")
    require_dimension(bounds.upper, x0.shape[0], name="upper")
    require_less_equal(bounds.lower, x0, left_name="lower", right_name="initial_point")
    require_less_equal(x0, bounds.upper, left_name="initial_point", right_name="upper")


def _require_plausible_bounds(bounds: Bounds) -> None:
    if bounds.plausible_lower is None or bounds.plausible_upper is None:
        msg = "PyVBMC requires plausible lower and plausible upper bounds."
        raise ModelError(msg)
