# Inference Runs

An inference run is the project-level object that connects a model to an engine.
It describes what should be executed without exposing the implementation details
of a particular backend.

The core pieces are:

- `InferenceRun`: the input to an engine.
- `InferenceEngine`: the protocol implemented by concrete engines.
- `InferenceResult`: the engine-neutral result returned after execution.
- `ModelEvaluation`: one recorded model evaluation.
- `EvaluationRecorder`: an in-memory collector for model evaluations.

## Run Inputs

`InferenceRun` contains the model, the initial point, an optional evaluation
context, and a flag controlling whether model evaluations should be retained in
memory:

```{code-block} python
from distributed_inference import InferenceRun

run = InferenceRun(
    model=bounded_model,
    initial_point=x0,
    context=context,
    record_evaluations=True,
)
```

This is intentionally engine-neutral. The run says "evaluate this model from
this starting point"; the engine decides how to turn that into backend-specific
work.

## Engines

An `InferenceEngine` has a name and implements `run_inference`:

```{code-block} python
result = engine.run_inference(run)
```

Concrete engines may also provide convenience methods. For example,
`PyVBMCEngine.run(...)` accepts the same essential inputs directly and builds an
`InferenceRun` internally. The important boundary is that user code passes
Distributed Inference abstractions into Distributed Inference engines. Backend
objects remain implementation details.

## Results

`InferenceResult` captures the common output shape:

```{code-block} python
posterior = result.posterior
diagnostics = result.diagnostics
evaluations = result.evaluations
```

`posterior` is intentionally generic at this stage. Different engines represent
posterior approximations differently. The result object keeps that engine-owned
object behind a stable project-level wrapper while exposing common metadata:

- `model_info`: the model metadata used for the run.
- `engine_name`: the engine that produced the result.
- `diagnostics`: engine diagnostics as a mapping.
- `evaluations`: optional in-memory model evaluations.

## Evaluation Recording

Recording evaluations is optional because it can be expensive. When enabled, the
engine stores `ModelEvaluation` records:

```{code-block} python
first = result.evaluations[0]
x = first.x
value = first.value
```

This is useful for smoke runs, diagnostics, and later run manifests. It also
keeps evaluation collection independent of backend callback APIs.

`EvaluationRecorder` is the internal in-memory collector used by engines. User
code normally reads `result.evaluations` rather than constructing a recorder
directly.
