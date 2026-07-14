"""Command-line entrypoint for Distributed Inference."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="di",
    help="Run, collect, and combine repeated inference runs.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Distributed Inference command-line interface."""


def main() -> None:
    """Run the CLI application."""
    app()
