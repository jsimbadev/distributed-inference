"""Internal validation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from distributed_inference.errors import ModelError

FloatArray = NDArray[np.float64]


def as_vector(value: ArrayLike, *, name: str) -> FloatArray:
    """Convert a value to a one-dimensional floating-point vector."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be a one-dimensional array."
        raise ModelError(msg)
    return array


def require_same_shape(
    left: FloatArray,
    right: FloatArray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Require two arrays to have matching shapes."""
    if left.shape != right.shape:
        msg = f"{left_name} and {right_name} must have matching shapes."
        raise ModelError(msg)


def require_less_equal(
    left: FloatArray,
    right: FloatArray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Require each element in one vector to be less than or equal to another."""
    if np.any(left > right):
        msg = f"{left_name} must be less than or equal to {right_name}."
        raise ModelError(msg)


def require_dimension(value: FloatArray, dimension: int, *, name: str) -> None:
    """Require a vector to have a specific dimension."""
    if value.shape != (dimension,):
        msg = f"{name} must have dimension {dimension}; got {value.shape[0]}."
        raise ModelError(msg)
