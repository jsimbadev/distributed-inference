"""PyVBMC adapter helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from distributed_inference.model import (
    BoundedModel,
    Bounds,
    EvaluationContext,
    Model,
    ModelError,
    ParameterSpace,
)


def as_pyvbmc_log_density(
    model: Model,
    context: EvaluationContext | None = None,
) -> Callable[[ArrayLike], float]:
    """Return a PyVBMC-compatible log-density callable."""
    _require_unconstrained(model)

    def log_density(x: ArrayLike) -> float:
        return model(np.asarray(x, dtype=np.float64), context)

    return log_density


def pyvbmc_bounds(model: BoundedModel) -> Bounds:
    """Return bounds required by PyVBMC for an unconstrained model."""
    _require_unconstrained(model)
    if not hasattr(model, "bounds"):
        msg = f"Model {model.info.name!r} does not define PyVBMC bounds."
        raise ModelError(msg)

    bounds = model.bounds()
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
