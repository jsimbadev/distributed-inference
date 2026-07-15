import numpy as np
import pytest

from distributed_inference import Bounds, CallableModel, ModelError, ParameterSpace
from distributed_inference.engines.pyvbmc import as_pyvbmc_log_density, pyvbmc_bounds
from distributed_inference.model import FloatArray


def test_as_pyvbmc_log_density_returns_callable(
    gaussian_model: CallableModel,
) -> None:
    assert callable(as_pyvbmc_log_density(gaussian_model))


def test_as_pyvbmc_log_density_evaluates_model(
    gaussian_model: CallableModel,
) -> None:
    log_density = as_pyvbmc_log_density(gaussian_model)

    assert log_density(np.array([1.0, 2.0])) == -2.5


def test_pyvbmc_bounds_returns_bounds(
    gaussian_model: CallableModel,
    gaussian_bounds: Bounds,
) -> None:
    assert pyvbmc_bounds(gaussian_model) == gaussian_bounds


def test_pyvbmc_bounds_requires_bounds() -> None:
    def log_density(x: FloatArray, context) -> float:
        return -float(x[0] ** 2)

    model = CallableModel(name="unbounded", dimension=1, fn=log_density)

    with pytest.raises(ModelError):
        pyvbmc_bounds(model)


def test_pyvbmc_adapter_requires_unconstrained_model(
    constrained_model: CallableModel,
) -> None:
    with pytest.raises(ModelError):
        as_pyvbmc_log_density(constrained_model)


def test_pyvbmc_bounds_requires_unconstrained_model() -> None:
    def log_density(x: FloatArray, context) -> float:
        return -float(x[0])

    model = CallableModel(
        name="constrained",
        dimension=1,
        fn=log_density,
        input_space=ParameterSpace.CONSTRAINED,
    )

    with pytest.raises(ModelError):
        pyvbmc_bounds(model)
