import numpy as np
import pytest

from distributed_inference import Bounds, CallableModel, ModelError, WithBounds
from distributed_inference._validation import FloatArray


def test_bounds_store_lower_vector(gaussian_bounds: Bounds) -> None:
    np.testing.assert_allclose(gaussian_bounds.lower, np.array([-5.0, -5.0]))


def test_bounds_reject_mismatched_shapes() -> None:
    with pytest.raises(ModelError):
        Bounds(lower=np.array([0.0]), upper=np.array([1.0, 2.0]))


def test_with_bounds_delegates_model_evaluation(
    gaussian_model: CallableModel,
    gaussian_bounds: Bounds,
) -> None:
    model = WithBounds(gaussian_model, gaussian_bounds)

    assert model(np.array([1.0, 2.0])) == -2.5


def test_with_bounds_exposes_model_info(
    gaussian_model: CallableModel,
    gaussian_bounds: Bounds,
) -> None:
    model = WithBounds(gaussian_model, gaussian_bounds)

    assert model.info == gaussian_model.info


def test_with_bounds_exposes_bounds(
    gaussian_model: CallableModel,
    gaussian_bounds: Bounds,
) -> None:
    model = WithBounds(gaussian_model, gaussian_bounds)

    assert model.bounds() == gaussian_bounds


def test_callable_model_does_not_require_bounds() -> None:
    def log_density(x: FloatArray, context) -> float:
        return -float(x[0])

    model = CallableModel(name="unbounded", dimension=1, fn=log_density)

    assert not hasattr(model, "bounds")
