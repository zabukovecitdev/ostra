from pathlib import Path

from ostra.cli.main import _apply_database_override
from ostra.cli.parser import build_parser
from ostra.models.app_config import AppConfig


def testbuild_parser__database_option__parsed_for_every_command():
    """Accept --database on the index and search commands."""
    parser = build_parser()

    for command in (["index"], ["search", "q"]):
        assert parser.parse_args([*command, "--database", "x.db"]).database == Path("x.db")


def test_apply_database_override__path_given__replaces_configured_path(tmp_path: Path):
    """Use the database of the option instead of the configured one."""
    config = AppConfig(database_path=tmp_path / "config.db")

    assert _apply_database_override(config, tmp_path / "other.db").database_path == tmp_path / "other.db"


def test_apply_database_override__tilde__expands_home_directory(tmp_path: Path):
    """Expand a leading tilde in the option."""
    config = AppConfig(database_path=tmp_path / "config.db")

    assert _apply_database_override(config, Path("~/other.db")).database_path == Path.home() / "other.db"


def test_apply_database_override__none__keeps_configured_path(tmp_path: Path):
    """Keep the configured database when the option is missing."""
    config = AppConfig(database_path=tmp_path / "config.db")

    assert _apply_database_override(config, None) is config
