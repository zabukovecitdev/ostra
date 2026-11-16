import logging
from pathlib import Path

import pytest

from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.models.embedding_model import EmbeddingModel
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.services import background_indexing_service, index_session, systemd_service

LOGGER_NAME = "ostra.services.background_indexing_service"


def test_index_once__config_fails__logs_error_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Log the error and do not raise it, so the loop goes on."""

    def fail_to_load():
        raise ValueError("bad config")

    monkeypatch.setattr(background_indexing_service, "load_config", fail_to_load)

    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        background_indexing_service.index_once()

    assert "Indexing run failed." in caplog.text
    assert "bad config" in caplog.text


def test_index_once__empty_directory__logs_zero_files_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Report zero processed files for an empty folder."""
    notes = tmp_path / "notes"
    notes.mkdir()
    config = AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    monkeypatch.setattr(background_indexing_service, "load_config", lambda: config)
    monkeypatch.setattr(index_session, "build_embedding_model", lambda config, model_id=None: _fake_model())

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        background_indexing_service.index_once()

    assert "FilesProcessed=0" in caplog.text


def test_unit_template__interval_given__puts_interval_in_exec_start():
    """Write the interval in the ExecStart line of the unit."""
    unit = systemd_service.UNIT_TEMPLATE.format(executable="/bin/ostra-daemon", interval_minutes=5)

    assert "ExecStart=/bin/ostra-daemon run --interval 5" in unit


def test_uninstall_service__unit_missing__reports_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a message and run no systemctl command when the unit file is missing."""
    monkeypatch.setattr(systemd_service, "_require_systemd", lambda: None)
    monkeypatch.setattr(systemd_service, "unit_path", lambda: tmp_path / "missing.service")

    assert systemd_service.uninstall_service() == "Service is not installed."


class _FakeProvider(EmbeddingProvider):
    def _embed(self, text: str) -> list[float]:
        return [0.0] * 1024


class _FakeTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_index_once__no_directories__logs_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Tell the user to add folders when the folder list is empty."""
    config = AppConfig(database_path=tmp_path / "ostra.db")
    monkeypatch.setattr(background_indexing_service, "load_config", lambda: config)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        background_indexing_service.index_once()

    assert NO_DIRECTORIES_MESSAGE in caplog.text


def _fake_model() -> EmbeddingModel:
    return EmbeddingModel(spec=MXBAI_EMBED_LARGE, provider=_FakeProvider(MXBAI_EMBED_LARGE), tokenizer=_FakeTokenizer())
