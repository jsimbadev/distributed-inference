"""Command-line entrypoint for cluster VBMC inference."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="csvbmc",
    help="Run, collect, and combine repeated PyVBMC inference runs.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Cluster VBMC inference command-line interface."""


def main() -> None:
    """Run the CLI application."""
    app()
