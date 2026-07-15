import numpy as np
import pytest
from numpy.typing import ArrayLike

from distributed_inference import (
    Bounds,
    CallableModel,
    EvaluationContext,
    ParameterSpace,
)
from distributed_inference.model import FloatArray


@pytest.fixture
def gaussian_bounds() -> Bounds:
    return Bounds(
        lower=np.array([-5.0, -5.0]),
        upper=np.array([5.0, 5.0]),
        plausible_lower=np.array([-2.0, -2.0]),
        plausible_upper=np.array([2.0, 2.0]),
    )


@pytest.fixture
def gaussian_model(gaussian_bounds: Bounds) -> CallableModel:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return -0.5 * float(np.dot(x, x))

    return CallableModel(
        name="gaussian",
        dimension=2,
        fn=log_density,
        model_bounds=gaussian_bounds,
    )


@pytest.fixture
def gradient_model(gaussian_bounds: Bounds) -> CallableModel:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return -0.5 * float(np.dot(x, x))

    def gradient(
        x: FloatArray,
        context: EvaluationContext | None,
    ) -> tuple[float, FloatArray]:
        return -0.5 * float(np.dot(x, x)), -x

    return CallableModel(
        name="gradient-gaussian",
        dimension=2,
        fn=log_density,
        model_bounds=gaussian_bounds,
        gradient_fn=gradient,
    )


@pytest.fixture
def constrained_model() -> CallableModel:
    def log_density(x: FloatArray, context: EvaluationContext | None) -> float:
        return -float(x[0])

    return CallableModel(
        name="positive-rate",
        dimension=1,
        fn=log_density,
        input_space=ParameterSpace.CONSTRAINED,
    )


@pytest.fixture
def evaluation_context() -> EvaluationContext:
    return EvaluationContext(
        run_id="run-1",
        rng=np.random.default_rng(123),
        metadata={"scale": 2.0},
    )


class ExpTransform:
    @property
    def constrained_dimension(self) -> int:
        return 1

    @property
    def unconstrained_dimension(self) -> int:
        return 1

    def to_unconstrained(self, x: ArrayLike) -> FloatArray:
        return np.log(x)

    def to_constrained(self, z: ArrayLike) -> FloatArray:
        return np.exp(z)

    def log_abs_det_jacobian(self, z: ArrayLike) -> float:
        vector = np.asarray(z, dtype=np.float64)
        return float(vector[0])


@pytest.fixture
def exp_transform() -> ExpTransform:
    return ExpTransform()
