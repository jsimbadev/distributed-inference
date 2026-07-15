# Run A PyVBMC Model

Distributed Inference treats PyVBMC as an engine implementation. User code
passes Distributed Inference models, bounds, and run inputs to `PyVBMCEngine`.

## Define The Model

```{code-block} python
import numpy as np

from distributed_inference import CallableModel

def log_density(x, context):
    return -0.5 * float(np.dot(x, x))

model = CallableModel(name="gaussian", dimension=2, fn=log_density)
```

## Add Bounds

```{code-block} python
from distributed_inference import Bounds, WithBounds

bounds = Bounds(
    lower=np.array([-5.0, -5.0]),
    upper=np.array([5.0, 5.0]),
    plausible_lower=np.array([-2.0, -2.0]),
    plausible_upper=np.array([2.0, 2.0]),
)
bounded_model = WithBounds(model, bounds)
```

## Run The Engine

```{code-block} python
from distributed_inference.engines.pyvbmc import PyVBMCEngine, PyVBMCOptions

engine = PyVBMCEngine(
    options=PyVBMCOptions(raw_options={"max_fun_evals": 100})
)
result = engine.run(
    bounded_model,
    initial_point=np.array([0.0, 0.0]),
    record_evaluations=True,
)
```

The result keeps PyVBMC's posterior object behind the project-level result:

```{code-block} python
posterior = result.posterior
diagnostics = result.diagnostics
evaluations = result.evaluations
```

`result.evaluations` is an in-memory tuple of model evaluations recorded through
the Distributed Inference abstraction, not through user-managed PyVBMC callbacks.

## Use An Explicit Inference Run

For code that should be independent of a concrete engine's convenience API, build
an `InferenceRun` and pass it to `run_inference`:

```{code-block} python
from distributed_inference import InferenceRun

run = InferenceRun(
    model=bounded_model,
    initial_point=np.array([0.0, 0.0]),
    record_evaluations=True,
)
result = engine.run_inference(run)
```

This is the shape distributed runners can use later: the runner receives a
project-level run description and delegates it to an engine without knowing how
that engine constructs backend objects.
