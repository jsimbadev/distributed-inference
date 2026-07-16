# Development Guide

This guide describes how to set up a local development environment and run the project hygiene tools.

## Prerequisites

Install:

- Python 3.11 or newer;
- `uv`;
- Git.

The supported Python range is exercised in continuous integration. Python 3.11 is the minimum supported version, and CI tests each stable minor version through the newest supported release.

## Create The Environment

From the repository root:

```{code-block} bash
uv sync --group dev
```

This installs the package in editable mode with runtime and developer dependencies.

## Install Pre-Commit Hooks

```{code-block} bash
uv run pre-commit install
```

For day-to-day use, hooks run automatically on staged files during `git commit`.
To run the hooks manually against staged files:

```{code-block} bash
uv run pre-commit run
```

To run the hooks against the whole repository:

```{code-block} bash
uv run pre-commit run --all-files
```

The hooks currently run:

- `ruff check --fix`;
- `ruff format`;
- `ty check`.

The pre-commit hooks use local commands through `uv run`. This keeps hook behavior aligned with the project environment instead of relying on separately managed hook environments. Continuous integration invokes the same pre-commit entry point, so local and remote quality checks share one configuration.

## Run Hygiene Checks Directly

Lint:

```{code-block} bash
uv run ruff check
```

Format:

```{code-block} bash
uv run ruff format
```

Type check:

```{code-block} bash
uv run ty check
```

Tests:

```{code-block} bash
uv run pytest
```

Docs:

```{code-block} bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Only the initial scaffold exists at this stage.

## Continuous Integration

GitHub Actions runs independent jobs for:

- the complete pre-commit quality stack;
- unit tests on every supported Python version;
- a warning-as-error Sphinx documentation build.

These jobs run in parallel where possible. Pull requests should be considered ready only when all required CI checks pass.

## Development Principles

Prefer small, inspectable changes:

- every inference run should leave a manifest;
- failed runs should be represented explicitly, not silently dropped;
- config files should be validated before job submission;
- local execution should work before cluster execution;
- diagnostics should report uncertainty and disagreement instead of hiding it behind a single combined posterior.

## Whole project validation

Run:

```{code-block} bash
uv run pre-commit run --all-files
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

If a check fails, fix the underlying issue rather than weakening the tool configuration.
