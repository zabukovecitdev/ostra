import json
from pathlib import Path

import pytest

from ostra import daemon, service_manager
from ostra.config import AppConfig


def test_index_once__config_fails__writes_error_to_status_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Save the error in the status file and do not raise it."""

    def fail_to_load():
        raise ValueError("bad config")

    monkeypatch.setattr(daemon, "load_config", fail_to_load)
    status_path = tmp_path / "status.json"

    daemon.index_once(status_path=status_path)

    status = json.loads(status_path.read_text())
    assert status["error"] == "bad config"
    assert status["files_processed"] == 0


def test_index_once__empty_directory__writes_status_without_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Report zero processed files and no error for an empty folder."""
    notes = tmp_path / "notes"
    notes.mkdir()
    config = AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    monkeypatch.setattr(daemon, "load_config", lambda: config)
    monkeypatch.setattr(daemon, "MxbaiEmbeddingModel", lambda settings: _FakeModel())
    status_path = tmp_path / "status.json"

    daemon.index_once(status_path=status_path)

    status = json.loads(status_path.read_text())
    assert status["error"] is None
    assert status["files_processed"] == 0


def test_unit_template__interval_given__puts_interval_in_exec_start():
    """Write the interval in the ExecStart line of the unit."""
    unit = service_manager.UNIT_TEMPLATE.format(executable="/bin/ostra-daemon", interval_minutes=5)

    assert "ExecStart=/bin/ostra-daemon run --interval 5" in unit


def test_uninstall_service__unit_missing__reports_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a message and run no systemctl command when the unit file is missing."""
    monkeypatch.setattr(service_manager, "_require_systemd", lambda: None)
    monkeypatch.setattr(service_manager, "unit_path", lambda: tmp_path / "missing.service")

    assert service_manager.uninstall_service() == "Service is not installed."


class _FakeModel:
    def get_tokenizer(self):
        return _FakeTokenizer()

    def get_embedding_provider(self):
        return None


class _FakeTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())
