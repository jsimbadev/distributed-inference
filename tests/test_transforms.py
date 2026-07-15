import numpy as np
import pytest

from distributed_inference import CallableModel, ParameterSpace, TransformedModel


def test_transformed_model_uses_unconstrained_space(
    constrained_model: CallableModel,
    exp_transform,
) -> None:
    model = TransformedModel(constrained_model, exp_transform)

    assert model.info.input_space == ParameterSpace.UNCONSTRAINED


def test_transformed_model_uses_unconstrained_dimension(
    constrained_model: CallableModel,
    exp_transform,
) -> None:
    model = TransformedModel(constrained_model, exp_transform)

    assert model.info.dimension == 1


def test_transformed_model_evaluates_base_on_constrained_value(
    constrained_model: CallableModel,
    exp_transform,
) -> None:
    model = TransformedModel(constrained_model, exp_transform)

    assert model(np.array([0.0])) == -1.0


def test_transformed_model_adds_jacobian(
    constrained_model: CallableModel,
    exp_transform,
) -> None:
    model = TransformedModel(constrained_model, exp_transform)

    assert model(np.array([1.0])) == pytest.approx(-np.e + 1.0)
