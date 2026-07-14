from typer.testing import CliRunner

from distributed_inference.cli import app


def test_cli_shows_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run, collect, and combine repeated inference runs." in result.stdout
