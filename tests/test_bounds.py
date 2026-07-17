import numpy as np
import pytest

from distributed_inference import Bounds, CallableModel, ModelError, WithBounds
from distributed_inference._validation import FloatArray


def test_bounds_reject_mismatched_shapes() -> None:
    with pytest.raises(ModelError):
        Bounds(lower=np.array([0.0]), upper=np.array([1.0, 2.0]))


def test_bounds_reject_inverted_limits() -> None:
    with pytest.raises(ModelError):
        Bounds(lower=np.array([1.0]), upper=np.array([0.0]))


def test_bounds_reject_plausible_lower_outside_limits() -> None:
    with pytest.raises(ModelError):
        Bounds(
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            plausible_lower=np.array([-0.1]),
        )


def test_bounds_reject_plausible_upper_outside_limits() -> None:
    with pytest.raises(ModelError):
        Bounds(
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            plausible_upper=np.array([1.1]),
        )


def test_bounds_reject_inverted_plausible_limits() -> None:
    with pytest.raises(ModelError):
        Bounds(
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            plausible_lower=np.array([0.8]),
            plausible_upper=np.array([0.2]),
        )


def test_with_bounds_delegates_model_evaluation(
    gaussian_model: CallableModel,
    gaussian_bounds: Bounds,
) -> None:
    model = WithBounds(gaussian_model, gaussian_bounds)

    assert model(np.array([1.0, 2.0])) == -2.5


def test_callable_model_does_not_require_bounds() -> None:
    def log_density(x: FloatArray, context) -> float:
        return -float(x[0])

    model = CallableModel(name="unbounded", dimension=1, fn=log_density)

    assert not hasattr(model, "bounds")
