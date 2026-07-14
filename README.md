# Cluster VBMC Inference

Cluster VBMC Inference is a Python project for running, collecting, and combining repeated PyVBMC inference runs on local and cluster-backed infrastructure.

The initial workflow is intentionally small:

1. define an inference problem in `inference.yaml`;
2. run multiple independent PyVBMC fits;
3. collect completed run outputs;
4. combine posterior approximations;
5. generate diagnostics and a report.

The first backend target is local development, followed by `mini-cluster` and then SLURM-style internal HPC usage.

## Development Status

This repository is in early scaffold stage. The package metadata, hygiene tooling, and documentation structure are being established before the inference runner and backends are implemented.

## Developer Setup

This project uses:

- `uv` for environment and dependency management;
- `ruff` for linting and formatting;
- `ty` for type checking;
- `pre-commit` for local hygiene checks;
- `pytest` for tests.

See [docs/development.md](docs/development.md) for setup and contribution workflow.

## License

This project is currently scaffolded with MIT license metadata. Add a `LICENSE` file before publishing or distributing the package.
