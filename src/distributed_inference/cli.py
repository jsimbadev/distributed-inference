"""Command-line entrypoint for Distributed Inference."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from distributed_inference.inspection import format_run_summaries, iter_run_summaries

app = typer.Typer(
    name="di",
    help="Run, collect, and combine repeated inference runs.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Distributed Inference command-line interface."""


@app.command("inspect")
def inspect_store(
    root: Annotated[
        Path,
        typer.Argument(help="Directory containing persisted run records."),
    ],
) -> None:
    """Summarize persisted inference runs without importing model code."""
    summaries = iter_run_summaries(root)
    if not summaries:
        typer.echo("No persisted inference runs found.")
        return
    typer.echo(format_run_summaries(summaries))


def main() -> None:
    """Run the CLI application."""
    app()
