from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike

from distributed_inference import (
    Bounds,
    CallableModel,
    ModelError,
    ParameterSpace,
    WithBounds,
)
from distributed_inference._validation import FloatArray
from distributed_inference.engines.pyvbmc import PyVBMCEngine, PyVBMCOptions


class FakeVBMC:
    def __init__(
        self,
        log_density: Callable[[ArrayLike], float],
        *,
        x0: FloatArray,
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
        plausible_lower_bounds: FloatArray,
        plausible_upper_bounds: FloatArray,
        options: Mapping[str, Any],
    ) -> None:
        self.log_density = log_density
        self.x0 = x0
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.plausible_lower_bounds = plausible_lower_bounds
        self.plausible_upper_bounds = plausible_upper_bounds
        self.options = options

    def optimize(self) -> tuple[str, dict[str, Any]]:
        value = self.log_density(self.x0)
        return "posterior", {"value_at_x0": value, "options": dict(self.options)}


@pytest.fixture
def fake_vbmc_instances() -> list[FakeVBMC]:
    return []


@pytest.fixture
def fake_vbmc_factory(
    fake_vbmc_instances: list[FakeVBMC],
) -> Callable[..., FakeVBMC]:
    def factory(*args: Any, **kwargs: Any) -> FakeVBMC:
        instance = FakeVBMC(*args, **kwargs)
        fake_vbmc_instances.append(instance)
        return instance

    return factory


def test_pyvbmc_engine_returns_project_result(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
    )

    assert result.posterior == "posterior"


def test_pyvbmc_engine_result_keeps_run_context(
    bounded_gaussian_model,
    evaluation_context,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
        context=evaluation_context,
    )

    assert result.run.context is evaluation_context


def test_pyvbmc_engine_result_keeps_run_model(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
    )

    assert result.run.model is bounded_gaussian_model


def test_pyvbmc_engine_returns_diagnostics(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
    )

    assert result.diagnostics["value_at_x0"] == -2.5


def test_pyvbmc_engine_passes_bounds(
    bounded_gaussian_model,
    gaussian_bounds: Bounds,
    fake_vbmc_factory,
    fake_vbmc_instances: list[FakeVBMC],
) -> None:
    PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(
        fake_vbmc_instances[0].lower_bounds, gaussian_bounds.lower
    )


def test_pyvbmc_engine_passes_options(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(
        options=PyVBMCOptions(raw_options={"max_fun_evals": 10}),
        vbmc_factory=fake_vbmc_factory,
    ).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
    )

    assert result.diagnostics["options"] == {"max_fun_evals": 10}


def test_pyvbmc_engine_records_evaluations(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    result = PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
        bounded_gaussian_model,
        initial_point=np.array([1.0, 2.0]),
        record_evaluations=True,
    )

    assert result.evaluations[0].value == -2.5


def test_pyvbmc_engine_rejects_unbounded_model(
    gaussian_model: CallableModel,
    fake_vbmc_factory,
) -> None:
    with pytest.raises(ModelError):
        PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
            gaussian_model,
            initial_point=np.array([1.0, 2.0]),
        )


def test_pyvbmc_engine_rejects_constrained_model(fake_vbmc_factory) -> None:
    def log_density(x: FloatArray, context) -> float:
        return -float(x[0])

    model = WithBounds(
        CallableModel(
            name="constrained",
            dimension=1,
            fn=log_density,
            input_space=ParameterSpace.CONSTRAINED,
        ),
        Bounds(
            lower=np.array([0.0]),
            upper=np.array([10.0]),
            plausible_lower=np.array([1.0]),
            plausible_upper=np.array([9.0]),
        ),
    )

    with pytest.raises(ModelError):
        PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
            model,
            initial_point=np.array([5.0]),
        )


def test_pyvbmc_engine_rejects_initial_point_outside_bounds(
    bounded_gaussian_model,
    fake_vbmc_factory,
) -> None:
    with pytest.raises(ModelError):
        PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
            bounded_gaussian_model,
            initial_point=np.array([6.0, 2.0]),
        )


def test_pyvbmc_engine_rejects_missing_plausible_bounds(
    gaussian_model: CallableModel,
    fake_vbmc_factory,
) -> None:
    model = WithBounds(
        gaussian_model,
        Bounds(lower=np.array([-5.0, -5.0]), upper=np.array([5.0, 5.0])),
    )

    with pytest.raises(ModelError):
        PyVBMCEngine(vbmc_factory=fake_vbmc_factory).run(
            model,
            initial_point=np.array([1.0, 2.0]),
        )
