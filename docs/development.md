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
- `pydoclint .`;
- `ty check`.

The pre-commit hooks use local commands through `uv run`. This keeps hook behavior aligned with the project environment instead of relying on separately managed hook environments. Continuous integration invokes Ruff, pydoclint, and `ty` directly over the project using the same committed configuration.

## Run Hygiene Checks Directly

Lint:

```{code-block} bash
uv run ruff check
```

Format:

```{code-block} bash
uv run ruff format
```

Docstring consistency:

```{code-block} bash
uv run pydoclint .
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
uv run sphinx-build -W -b html docs docs/_build/html
```

Only the initial scaffold exists at this stage.

## Continuous Integration

Continuous integration protects the default branch by requiring each proposed change to satisfy three independent guarantees:

- the project conforms to the repository's Ruff, pydoclint, and `ty` configuration;
- the unit-test suite passes on every supported Python version;
- the documentation builds without warnings.

A pull request is ready only when all required guarantees hold. Keeping the checks independent makes failures attributable and allows unrelated validation to proceed in parallel.

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
uv run sphinx-build -W -b html docs docs/_build/html
```

If a check fails, fix the underlying issue rather than weakening the tool configuration.