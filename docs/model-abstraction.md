# Model Abstraction

The model abstraction separates three concerns:

- the parameter vector being evaluated;
- the coordinate space that vector belongs to;
- the execution context for that specific evaluation.

## Callable Models

Models are callable:

```{code-block} python
from distributed_inference import EvaluationContext

value = model(x, context)
```

The model instance has one input space. It does not switch between constrained
and unconstrained interpretation at call time. If a different space is needed,
wrap the model in another model.

This keeps calls explicit and avoids hidden coordinate changes through mutable
state.

## Evaluation Context

Models do not own randomness.

Any randomness, cache, run identifier, or backend-specific context is passed
into the operation:

```{code-block} python
import numpy as np

from distributed_inference import EvaluationContext

context = EvaluationContext(rng=np.random.default_rng(123))
value = model(x, context)
```

This keeps model objects reusable across runs and makes distributed evaluation
easier to inspect.

## Parameter Spaces

`ParameterSpace.CONSTRAINED` means the model accepts parameters in their native
domain. `ParameterSpace.UNCONSTRAINED` means the model accepts vectors in
Euclidean coordinates.

Inference engines usually want one of these spaces explicitly. PyVBMC is
adapted through an unconstrained callable plus bounds.

## Optional Bounds

Bounds are an optional model capability, not part of the base model protocol.
Many inference engines only need a callable log density and a dimension.

When an engine requires bounds, attach them by composition:

```{code-block} python
from distributed_inference import WithBounds

bounded_model = WithBounds(model, bounds)
```

This keeps the base callable abstraction small while allowing engines such as
PyVBMC to request a `BoundedModel` capability explicitly.

## Transformed Models

Use `TransformedModel` to expose a constrained model in unconstrained
coordinates. The transformed model evaluates:

```{code-block} python
base_model(to_constrained(z)) + log_abs_det_jacobian(z)
```

Automatic bound transformation is not part of the first implementation. If an
engine needs bounds for a transformed model, attach explicit bounds in the
transformed model input space.

## Optional Gradients

Gradient support is a capability, not a requirement. A `CallableModel` without a
gradient function only supports scalar log-density evaluation. A model with a
gradient function supports `log_density_and_gradient`.

This keeps black-box targets usable while leaving a path for engines that can
exploit derivative information.

## PyVBMC Adapter

PyVBMC receives a plain callable:

```{code-block} python
from distributed_inference.engines.pyvbmc import as_pyvbmc_log_density, pyvbmc_bounds

log_density = as_pyvbmc_log_density(model, context)
bounds = pyvbmc_bounds(bounded_model)
```

The adapter requires unconstrained model input space. Bound extraction requires
the separate bounded-model capability.
