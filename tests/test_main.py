import logging
from pathlib import Path

import pytest

from personal_rag import main


def test_main__database_ready__prints_database_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the database path after the database is ready."""
    database_path = tmp_path / "personal_rag.db"
    monkeypatch.setattr(main, "ensure_database", lambda: database_path)

    main.main()

    assert capsys.readouterr().out == f"Database ready at {database_path}\n"


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(main, "ensure_database", lambda: tmp_path / "personal_rag.db")

    main.main()

    assert basic_config_calls == [{"level": logging.INFO}]


def test_main__database_ready__calls_ensure_database_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Call ensure_database exactly once per run."""
    calls: list[None] = []

    def fake_ensure_database():
        calls.append(None)
        return tmp_path / "personal_rag.db"

    monkeypatch.setattr(main, "ensure_database", fake_ensure_database)

    main.main()

    assert len(calls) == 1
