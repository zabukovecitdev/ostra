import logging
from pathlib import Path

import pytest

from personal_rag import main
from personal_rag.config import AppConfig


def test_main__database_ready__prints_database_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the database path from the config after the database is ready."""
    database_path = tmp_path / "personal_rag.db"
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=database_path, directories=[]))

    main.main([])

    assert capsys.readouterr().out == f"Database ready at {database_path}\n"
    assert database_path.is_file()


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[])
    )

    main.main([])

    assert basic_config_calls == [{"level": logging.INFO}]


def test_main__directories_have_files__prints_each_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print each line of the files in the configured directories without the line ending."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\nbeta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )

    main.main([])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha", "beta"]


def test_main__reindex_flag__prints_lines_of_unchanged_file_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the lines of an unchanged file again when the reindex flag is given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )
    main.main([])
    capsys.readouterr()

    main.main(["--reindex"])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha"]


def test_main__no_reindex_flag__skips_lines_of_unchanged_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print no lines of an unchanged file when the reindex flag is not given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )
    main.main([])
    capsys.readouterr()

    main.main([])

    assert capsys.readouterr().out.splitlines()[1:] == []
