# Models

A model is the smallest inference object in Distributed Inference. It is a
callable log-density object:

```{code-block} python
value = model(x, context)
```

The model receives a parameter vector and an optional evaluation context, then
returns a scalar log density.

The base model concept does not include bounds, transformations, samplers, job
submission, or posterior combination. Those are separate concepts layered around
the model when needed.

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

## Optional Gradients

Gradient support is a capability on a model, not a requirement for all models.
If a gradient function is supplied, `CallableModel` exposes
`log_density_and_gradient`.

```{code-block} python
def log_density_and_gradient(x, context):
    return -0.5 * float(np.dot(x, x)), -x
```

Black-box engines can ignore gradients. Gradient-aware engines can check
`model.info.supports_gradient`.
