# Models

A model is the smallest inference object in Distributed Inference. It is a
callable log-density object:

```{code-block} python
value = model(x, context)
```

The model receives a parameter vector and an optional evaluation context, then
returns a scalar log density.

This page defines only the callable model semantics. The `context` argument is
explained in [Evaluation Contexts](contexts.md). Other model capabilities are
introduced on their own concept pages when they become necessary.

## Model Metadata

Every model exposes `ModelInfo`:

```{code-block} python
info = model.info
dimension = info.dimension
input_space = info.input_space
```

`input_space` describes how the model interprets incoming vectors. A model has
one input-space interpretation. It should not switch between constrained and
unconstrained coordinates using hidden mutable state.

## Callable Models

`CallableModel` wraps a Python callable:

```{code-block} python
import numpy as np

from distributed_inference import CallableModel

def log_density(x, context):
    return -0.5 * float(np.dot(x, x))

model = CallableModel(name="gaussian", dimension=2, fn=log_density)
```

This is the default user-facing path for simple models.

## Differentiable Models

Gradient support is a separate model capability, not an optional branch inside
`CallableModel`. Use `CallableDifferentiableModel` when the model can evaluate a
log density and first derivative together.

```{code-block} python
from distributed_inference import CallableDifferentiableModel

def log_density_and_gradient(x, context):
    return -0.5 * float(np.dot(x, x)), -x

model = CallableDifferentiableModel(
    name="gaussian",
    dimension=2,
    fn=log_density,
    gradient_fn=log_density_and_gradient,
)
```

Black-box engines can ignore gradients. Gradient-aware engines can check
`model.info.supports_gradient` and use the `DifferentiableModel` protocol.
