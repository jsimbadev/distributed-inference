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

`InferenceRun` contains a logical name, the model, the initial point, an
optional evaluation context, and a flag controlling whether model evaluations
should be retained in memory:

```{code-block} python
from distributed_inference import InferenceRun

run = InferenceRun(
    name="local-gaussian",
    model=bounded_model,
    initial_point=x0,
    context=context,
    record_evaluations=True,
)
```

This is intentionally engine-neutral. The run says "for this named inference
problem, evaluate this model from this starting point"; the engine decides how
to turn that into backend-specific work. The context may carry `run_id`, which
identifies one concrete invocation of the named run.

## Engines

An `InferenceEngine` has a name and implements `run_inference`:

```{code-block} python
result = engine.run_inference(run)
```

An engine is algorithmic machinery. It should not own cluster clients, process
pools, queues, or persistence stores. Those concerns belong to
`ExecutionBackend` and persistence components.

Concrete engines may also provide convenience methods. For example,
`PyVBMCEngine.run(...)` accepts the same essential inputs directly and builds an
`InferenceRun` internally. The important boundary is that user code passes
Distributed Inference abstractions into Distributed Inference engines. Backend
objects remain implementation details.

## Execution Backends

An `ExecutionBackend` runs an engine against a run and records execution
provenance:

```{code-block} python
executed = backend.execute(
    run,
    engine,
    attempt_number=1,
)

result = executed.result
execution = executed.execution
```

Local execution uses `LocalExecutionBackend`. Future cluster execution should
implement the same boundary instead of leaking scheduler objects into models or
engines.

## Results

`InferenceResult` captures the common output shape:

```{code-block} python
posterior = result.posterior
diagnostics = result.diagnostics
evaluations = result.evaluations
context = result.run.context
```

`posterior` is intentionally generic at this stage. Different engines represent
posterior approximations differently. The result object keeps that engine-owned
object behind a stable project-level wrapper while exposing common metadata:

- `engine_name`: the engine that produced the result.
- `run`: the exact project-level run object that was executed.
- `diagnostics`: engine diagnostics as a mapping.
- `evaluations`: optional in-memory model evaluations.

Keeping `run` on the result means local code can inspect the model metadata,
initial point, and execution context without a parallel metadata API:

```{code-block} python
model_info = result.run.model.info
initial_point = result.run.initial_point
context = result.run.context
```

The context is exposed as the original in-memory object. That is useful for
local execution, but it is not a persistence format. Future run manifests should
store explicit serializable metadata rather than blindly serializing the whole
context, because a context may contain random number generators, cache handles,
worker resources, or other process-local objects.

For persisted results, use `ResultManifest` and artifact references rather than
serializing the `InferenceResult` object directly.

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
