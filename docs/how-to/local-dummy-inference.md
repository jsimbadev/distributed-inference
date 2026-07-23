# Run And Persist Local Dummy Inference

This guide shows the smallest complete path from a serializable run
specification to local execution and persisted manifests.

It uses `DummyInferenceEngine` instead of PyVBMC. The point is to validate the
project-level execution and persistence boundaries with a fast engine whose
behavior is controlled by Distributed Inference.

## Define An Importable Model Builder

Put the model builder at module scope so `ModelSpec` can reference it later.

```{code-block} python
import numpy as np

from distributed_inference import CallableModel


def build_gaussian_model(config):
    dimension = int(config["dimension"])

    def log_density(x, context):
        return -0.5 * float(np.dot(x, x))

    return CallableModel(
        name="gaussian",
        dimension=dimension,
        fn=log_density,
    )
```

## Build A Run Specification

```{code-block} python
from pathlib import Path

from distributed_inference.persistence import ModelSpec, RandomStreamSpec
from distributed_inference.persistence.runs import InferenceRunSpec

model = ModelSpec.from_callable(
    build_gaussian_model,
    config={"dimension": 2},
    project_root=Path.cwd(),
    version="1",
)

run_spec = InferenceRunSpec(
    schema_version="1",
    name="local-dummy-smoke",
    run_id="run-001",
    model=model,
    initial_point=[0.0, 0.0],
    random_stream=RandomStreamSpec(
        algorithm="numpy.pcg64",
        seed=42,
        stream_id="stream-000",
        schema_version="1",
    ),
    context_metadata={"target": "full-posterior"},
    record_evaluations=True,
)
```

`name` identifies the logical inference run. `run_id` identifies this concrete
invocation of that named run. The random-stream identity lives in
`RandomStreamSpec`.

## Execute Locally

```{code-block} python
from pathlib import Path

from distributed_inference import run_inference
from distributed_inference.engines.dummy import DummyInferenceEngine
from distributed_inference.persistence import LocalInferenceStore
from distributed_inference.persistence.manifests import TargetSpec

engine = DummyInferenceEngine()
store = LocalInferenceStore(Path("inference-records"))
target = TargetSpec(
    identifier="gaussian.full-posterior",
    semantics="full-posterior",
    dimension=2,
    coordinate_space="unconstrained",
)

persisted = run_inference(
    run=run_spec,
    engine=engine,
    store=store,
    target=target,
)
```

`persisted.executed.result` is the in-memory engine result.
`persisted.result_manifest` is the process-independent result record.
`persisted.files` contains the paths written for the run, result, and execution
manifests.

## Written Files

The facade writes:

- `runs/local-dummy-smoke/run-001/run.json`;
- `runs/local-dummy-smoke/run-001/result.json`;
- `runs/local-dummy-smoke/run-001/execution.json`;
- `runs/local-dummy-smoke/run-001/artifacts/posterior.json`;
- `runs/local-dummy-smoke/run-001/artifacts/diagnostics.json`;
- `runs/local-dummy-smoke/run-001/artifacts/evaluations.json` when evaluations
  are recorded.

Lower-level manifest objects are still available for engine integrations that
need to provide explicit checkpoint references or custom artifact formats.

## Repeated Local Invocations

The repository includes a runnable example:

```bash
uv run python examples/local_repeated_runs.py
```

It creates two invocations under the same logical run name and writes each one
under a different `run_id`.

Inspect the records without importing the model:

```bash
uv run di inspect example-records
```

## Restart And Checkpoint Semantics

The `RandomStreamSpec` in `run.json` and `result.json` is enough to recreate a
fresh generator for a reproducible restart of the logical run.

The `checkpoints` section of `result.json` is different. It points to exact
engine state for continuation when an engine supports checkpointing. A completed
posterior artifact is not a checkpoint.
