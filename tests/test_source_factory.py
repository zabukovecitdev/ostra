from pathlib import Path

import pytest

from ostra.models.app_config import AppConfig
from ostra.services.ingestion import source_factory
from ostra.services.ingestion.source_factory import build_sources


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


def _record_recursive(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Replace the file system source with a recorder of its recursive argument."""
    calls: list[bool] = []
    monkeypatch.setattr(
        source_factory, "FileSystemSource", lambda directories, recursive: calls.append(recursive) or object()
    )
    return calls


def test_build_sources__recursive_not_given__uses_config_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Follow the recursive setting of the config when the caller does not set it."""
    calls = _record_recursive(monkeypatch)

    build_sources(AppConfig(database_path=tmp_path / "db", recursive=True))

    assert calls == [True]


def test_build_sources__recursive_given__overrides_config_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Let an explicit value win over the recursive setting of the config."""
    calls = _record_recursive(monkeypatch)

    build_sources(AppConfig(database_path=tmp_path / "db", recursive=True), recursive=False)

    assert calls == [False]
