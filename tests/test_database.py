import logging
import sqlite3
from pathlib import Path

import pytest
import sqlite_vec

from personal_rag import database
from personal_rag.database import DATABASE_DIRECTORY, DATABASE_PATH, ensure_database


def test_ensure_database__database_missing__creates_file_in_missing_directory(tmp_path: Path):
    """Create the database file and its missing parent directories."""
    database_path = tmp_path / "outer" / "nested" / "personal_rag.db"

    result = ensure_database(database_path)

    assert result == database_path
    assert database_path.is_file()


def test_ensure_database__directory_exists__creates_file_without_error(tmp_path: Path):
    """Create the database file in an existing directory."""
    database_path = tmp_path / "personal_rag.db"

    result = ensure_database(database_path)

    assert result == database_path
    assert database_path.is_file()


def test_connect__connection_returned__has_extension_loading_disabled(tmp_path: Path):
    """Return a connection that cannot load further extensions."""
    connection = database._connect(tmp_path / "personal_rag.db")

    with pytest.raises(sqlite3.OperationalError, match="not authorized"):
        connection.load_extension(str(tmp_path / "missing_extension"))

    connection.close()


def test_ensure_database__database_exists__keeps_file_unchanged(tmp_path: Path):
    """Keep the content of an existing database file unchanged."""
    database_path = tmp_path / "personal_rag.db"
    database_path.write_bytes(b"existing content")

    result = ensure_database(database_path)

    assert result == database_path
    assert database_path.read_bytes() == b"existing content"


def test_ensure_database__database_exists__does_not_log_creation(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log nothing when the database file already exists."""
    database_path = tmp_path / "personal_rag.db"
    database_path.write_bytes(b"")

    with caplog.at_level(logging.INFO, logger="personal_rag.database"):
        ensure_database(database_path)

    assert caplog.records == []


def test_ensure_database__database_missing__loads_sqlite_vec_extension(tmp_path: Path):
    """Load the sqlite-vec extension into the new database."""
    database_path = tmp_path / "personal_rag.db"

    ensure_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    vec_version = connection.execute("SELECT vec_version()").fetchone()[0]
    connection.close()

    assert vec_version == f"v{sqlite_vec.__version__}"


def test_ensure_database__database_missing__logs_creation_with_path(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log the creation steps with the database path and the sqlite-vec version."""
    database_path = tmp_path / "personal_rag.db"

    with caplog.at_level(logging.INFO, logger="personal_rag.database"):
        ensure_database(database_path)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"Database not found. Creating it. Path={database_path}",
        f"Database created with sqlite-vec. VecVersion=v{sqlite_vec.__version__}",
    ]
    assert all(record.levelno == logging.INFO for record in caplog.records)


def test_ensure_database__no_path_given__uses_default_database_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create the database at DATABASE_PATH when no path is given."""
    default_path = tmp_path / "default" / "personal_rag.db"
    monkeypatch.setattr(database, "DATABASE_PATH", default_path)

    result = ensure_database()

    assert result == default_path
    assert default_path.is_file()


def test_ensure_database__database_missing__creates_owner_only_directory_and_file(tmp_path: Path):
    """Create the directory with mode 700 and the database file with mode 600."""
    database_path = tmp_path / "outer" / "personal_rag.db"

    ensure_database(database_path)

    assert database_path.parent.stat().st_mode & 0o777 == 0o700
    assert database_path.stat().st_mode & 0o777 == 0o600


def test_ensure_database__database_exists_with_open_permissions__restricts_to_owner(tmp_path: Path):
    """Set mode 600 on an existing database file with open permissions."""
    database_path = tmp_path / "personal_rag.db"
    database_path.write_bytes(b"")
    database_path.chmod(0o644)

    ensure_database(database_path)

    assert database_path.stat().st_mode & 0o777 == 0o600


def test_database_path__module_constants__point_to_home_personal_rag_folder():
    assert DATABASE_DIRECTORY == Path.home() / "personal-rag"
    assert DATABASE_PATH == Path.home() / "personal-rag" / "personal_rag.db"
