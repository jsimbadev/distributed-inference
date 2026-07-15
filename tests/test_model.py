import numpy as np
import pytest

from distributed_inference import (
    CallableModel,
    EvaluationContext,
    ModelCapabilityError,
    ModelError,
)
from distributed_inference._validation import FloatArray


def test_callable_model_evaluates_scalar(gaussian_model: CallableModel) -> None:
    assert gaussian_model(np.array([1.0, 2.0])) == -2.5


def test_callable_model_exposes_model_info(gaussian_model: CallableModel) -> None:
    assert gaussian_model.info.name == "gaussian"


def test_callable_model_rejects_wrong_dimension(
    gaussian_model: CallableModel,
) -> None:
    with pytest.raises(ModelError):
        gaussian_model(np.array([1.0]))


def test_callable_model_reports_missing_gradient(
    gaussian_model: CallableModel,
) -> None:
    assert gaussian_model.info.supports_gradient is False


def test_callable_model_rejects_missing_gradient(
    gaussian_model: CallableModel,
) -> None:
    with pytest.raises(ModelCapabilityError):
        gaussian_model.log_density_and_gradient(np.array([1.0, 2.0]))


def test_gradient_model_reports_gradient_support(
    gradient_model: CallableModel,
) -> None:
    assert gradient_model.info.supports_gradient is True


def test_gradient_model_returns_value(gradient_model: CallableModel) -> None:
    value, _ = gradient_model.log_density_and_gradient(np.array([1.0, 2.0]))

    assert value == -2.5


def test_gradient_model_returns_gradient(gradient_model: CallableModel) -> None:
    _, gradient = gradient_model.log_density_and_gradient(np.array([1.0, 2.0]))

    np.testing.assert_allclose(gradient, np.array([-1.0, -2.0]))


def test_evaluation_context_is_passed_to_callable(
    evaluation_context: EvaluationContext,
) -> None:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return float(context.metadata["scale"]) if context is not None else 0.0

    model = CallableModel(name="stateful", dimension=1, fn=log_density)

    assert model(np.array([0.0]), evaluation_context) == 2.0


def test_evaluation_context_rng_can_be_used_by_callable(
    evaluation_context: EvaluationContext,
) -> None:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        if context is None or context.rng is None:
            raise RuntimeError("rng is required")
        return float(context.rng.normal())

    model = CallableModel(name="rng-user", dimension=1, fn=log_density)

    assert isinstance(model(np.array([0.0]), evaluation_context), float)


def test_evaluation_context_rng_is_not_stored_on_model() -> None:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return -float(x[0])

    model = CallableModel(name="rng-user", dimension=1, fn=log_density)

    assert not hasattr(model, "rng")
