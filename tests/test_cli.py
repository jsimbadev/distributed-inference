from typer.testing import CliRunner

from cluster_vbmc_inference.cli import app


def test_cli_shows_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run, collect, and combine repeated PyVBMC inference runs." in result.stdout
