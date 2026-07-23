# Execution

Execution is the orchestration layer around an inference run. It is separate
from the model and separate from the inference algorithm.

The three relevant objects are:

- `InferenceRun`: the statistical work to perform.
- `InferenceEngine`: the algorithm that turns the run into an inference result.
- `ExecutionBackend`: the orchestration mechanism that runs the selected engine
  against the selected run.

This separation matters because local execution, process pools, cluster
schedulers, and cloud runners should not change the model interface.

## Local Execution

`LocalExecutionBackend` runs synchronously in the current Python process:

```{code-block} python
from distributed_inference import LocalExecutionBackend
from distributed_inference.engines.dummy import DummyInferenceEngine

backend = LocalExecutionBackend()
engine = DummyInferenceEngine()

executed = backend.execute(
    run,
    engine,
    attempt_number=1,
)
result = executed.result
execution = executed.execution
```

The backend does not persist files and does not own the model or engine. It only
executes the pair and returns execution provenance.

## Execution Records

`ExecutionRecord` is the serializable provenance for a backend execution. It
contains:

- backend name, version, and configuration;
- the named run and invocation identity being executed;
- the infrastructure attempt identity;
- timestamps;
- status;
- backend metadata.

The stable logical label is `name`. The concrete invocation identity is
`run_id`. `attempt_number` is a positive integer scoped to that invocation and
increments when the same invocation is retried.

Cluster-specific scheduler IDs, array indices, worker IDs, futures, and queue
handles are backend details. They should be represented later as execution
metadata or execution artifacts, not as generic result-manifest identity fields.

## Failure Records

If local execution fails, `LocalExecutionBackend` raises `ExecutionError`. The
error carries a failed `ExecutionRecord`:

```{code-block} python
from distributed_inference import ExecutionError

try:
    backend.execute(run, engine, attempt_number=1)
except ExecutionError as error:
    failed_execution = error.record
```

This keeps failure provenance available without pretending a completed
`InferenceResult` exists.
