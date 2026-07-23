# Persistence

Persistence records enough process-independent information to understand and
recreate completed inference work. It does not serialize live Python runtime
objects.

The core persisted objects are:

- `ModelSpec`: an importable model builder reference plus model configuration.
- `RandomStreamSpec`: a versioned random-stream restart specification.
- `InferenceRunSpec`: a serializable specification for rehydrating an
  `InferenceRun`.
- `ResultManifest`: process-independent metadata for a completed result.
- `ExecutionRecord`: process-independent provenance for a backend execution.

Each persisted representation has an explicit `schema_version`.

## Serializable Boundaries

Manifests should contain only ordinary serializable values such as strings,
numbers, booleans, lists, dictionaries, and artifact references.

They should not implicitly serialize:

- model callables;
- NumPy random generators;
- context caches;
- engine instances;
- backend instances;
- process handles;
- futures;
- open files.

Instead, runtime objects are represented by durable references. A model callable
is represented by `PythonCallableSpec`; a random generator is represented by
`RandomStreamSpec`; large or engine-specific outputs are represented by
`ArtifactReference`.

## Identity

The current manifest identity separates:

- `name`: the human-facing logical name of the inference run.
- `run_id`: the unique identifier for one invocation of that named run.
- `attempt_number`: the infrastructure attempt number for that invocation.

The name is stable across repeated invocations of the same inference problem.
The `run_id` changes for each invocation, which prevents repeated local or
cluster executions from overwriting each other. The `attempt_number` belongs to
execution provenance and increments if the same invocation is retried by a
backend. Backend-specific job identifiers belong in execution metadata.

Random-stream identity lives in `RandomStreamSpec`. A future cluster backend can
add scheduler-specific IDs under execution metadata or execution artifacts
without changing the generic result identity.

## Restart Versus Checkpoint

`RandomStreamSpec` is restart metadata. It records an algorithm, seed, stream
identifier, and schema version:

```{code-block} python
from distributed_inference.persistence import RandomStreamSpec

random_stream = RandomStreamSpec(
    algorithm="numpy.pcg64",
    seed=42,
    stream_id="stream-000",
    schema_version="1",
)
```

Rehydrating this spec creates a fresh process-local generator for a reproducible
restart of the same logical stream. It does not claim to resume from the exact
mid-run state of an engine.

Exact continuation is represented by checkpoint references:

```{code-block} python
from distributed_inference.persistence import ArtifactReference

checkpoint = ArtifactReference(
    uri="checkpoints/engine-state.json",
    media_type="application/json",
    checksum="sha256:...",
)
```

Checkpoints live under the `checkpoints` section of a result manifest.
Completed outputs such as posterior summaries, diagnostics, and recorded
evaluations live under `artifacts`. Keeping these separate prevents a completed
posterior artifact from being mistaken for engine state that can resume an
interrupted computation exactly.

## Local Store

`LocalInferenceStore` writes JSON manifests and JSON artifacts to a directory:

```{code-block} python
from pathlib import Path

from distributed_inference.persistence import LocalInferenceStore

store = LocalInferenceStore(Path("inference-records"))
store.write_run(run_spec)
store.write_execution(executed.execution)
store.write_result(result_manifest)
```

This writes invocation-specific records under
`runs/{name}/{run_id}/`. Re-running the same named inference with a new
`run_id` creates a separate directory instead of overwriting `result.json`.

The store validates JSON serializability when writing. If a posterior object is
engine-specific and cannot be serialized as JSON, the engine integration should
write an explicit artifact format and put an `ArtifactReference` in the
manifest.

## Facade

Application code should normally use `run_inference` rather than manually
assembling result-manifest metadata:

```{code-block} python
from distributed_inference import run_inference

persisted = run_inference(
    run=run_spec,
    engine=engine,
    store=store,
    target=target,
)
```

The facade rehydrates the run, executes the engine through an execution backend,
writes engine-appropriate artifacts, constructs the result manifest, and writes
the run/result/execution manifests. The lower-level objects remain public
because engine adapters and non-local backends need explicit control.
