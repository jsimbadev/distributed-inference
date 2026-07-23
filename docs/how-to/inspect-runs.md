# Inspect Persisted Runs

Persisted inference records are ordinary JSON files. They can be inspected
without importing the original model, constructing an engine, or rehydrating a
runtime context.

After running an example or application that writes records to a store, use:

```bash
uv run di inspect example-records
```

The command reads `runs/{name}/{run_id}/result.json` files and prints a compact
summary:

```text
name                 run_id       status     engine  target                            artifacts  checkpoints
gaussian-local-demo  run-001-...  completed  dummy   examples.gaussian.full-posterior  3          0
```

This is deliberately read-only. The inspection path should not need model
source files, PyVBMC objects, scheduler clients, or random generators. That
property is the point of the manifest boundary: completed inference work can be
listed, audited, and compared even when the runtime environment that produced it
is unavailable.
