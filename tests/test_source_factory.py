from pathlib import Path

from ostra.models.app_config import AppConfig
from ostra.services.source_factory import build_sources


def test_build_sources__no_directories_given__uses_configured_folders(tmp_path: Path):
    """Read the configured folders when no folder is given."""
    config = AppConfig(database_path=tmp_path / "db", directories=[tmp_path / "configured"])

    (source,) = build_sources(config)

    assert source.directories == [tmp_path / "configured"]


def test_build_sources__directories_given__ignores_configured_folders(tmp_path: Path):
    """Read the given folders instead of the configured folders."""
    config = AppConfig(database_path=tmp_path / "db", directories=[tmp_path / "configured"])

    (source,) = build_sources(config, directories=[tmp_path / "given"])

    assert source.directories == [tmp_path / "given"]
