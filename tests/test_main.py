import logging
from pathlib import Path

import pytest

from personal_rag import main


def test_main__database_ready__prints_database_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the database path after the database is ready."""
    database_path = tmp_path / "personal_rag.db"
    monkeypatch.setattr(main, "DATABASE_PATH", database_path)

    main.main()

    assert capsys.readouterr().out == f"Database ready at {database_path}\n"
    assert database_path.is_file()


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(main, "DATABASE_PATH", tmp_path / "personal_rag.db")

    main.main()

    assert basic_config_calls == [{"level": logging.INFO}]


def test_database_path__module_constant__points_to_home_personal_rag_folder():
    assert main.DATABASE_PATH == Path.home() / "personal-rag" / "personal_rag.db"
