import logging
import sqlite3
import sys
from pathlib import Path

import pytest
import sqlite_vec

from ostra.repositories.sqlite.sqlite_database import SqliteDatabase

LOGGER_NAME = "ostra.repositories.sqlite.sqlite_database"
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission modes do not apply on Windows")


def test_ensure_database__database_missing__creates_file_in_missing_directory(tmp_path: Path):
    """Create the database file and its missing parent directories."""
    database_path = tmp_path / "outer" / "nested" / "ostra.db"
    store = SqliteDatabase(database_path)

    store.ensure_database()

    assert database_path.is_file()


def test_ensure_database__directory_exists__creates_file_without_error(tmp_path: Path):
    """Create the database file in an existing directory."""
    database_path = tmp_path / "ostra.db"
    store = SqliteDatabase(database_path)

    store.ensure_database()

    assert database_path.is_file()


def test_ensure_database__database_exists__keeps_file_unchanged(tmp_path: Path):
    """Keep the content of an existing database file unchanged."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"existing content")
    store = SqliteDatabase(database_path)

    store.ensure_database()

    assert database_path.read_bytes() == b"existing content"


def test_ensure_database__database_exists__does_not_log_creation(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log nothing when the database file already exists."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"")
    store = SqliteDatabase(database_path)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        store.ensure_database()

    assert caplog.records == []


def test_ensure_database__database_missing__loads_sqlite_vec_extension(tmp_path: Path):
    """Load the sqlite-vec extension into the new database."""
    store = SqliteDatabase(tmp_path / "ostra.db")
    store.ensure_database()

    vec_version = store.connect().execute("SELECT vec_version()").fetchone()[0]
    store.close()

    assert vec_version == f"v{sqlite_vec.__version__}"


def test_ensure_database__database_missing__logs_creation_with_path(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log the creation steps with the database path and the sqlite-vec version."""
    database_path = tmp_path / "ostra.db"
    store = SqliteDatabase(database_path)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        store.ensure_database()

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"Database not found. Creating it. Path={database_path}",
        f"Database created with sqlite-vec. VecVersion=v{sqlite_vec.__version__}",
    ]
    assert all(record.levelno == logging.INFO for record in caplog.records)


@POSIX_ONLY
def test_ensure_database__database_missing__creates_owner_only_directory_and_file(tmp_path: Path):
    """Create the directory with mode 700 and the database file with mode 600."""
    database_path = tmp_path / "outer" / "ostra.db"
    store = SqliteDatabase(database_path)

    store.ensure_database()

    assert database_path.parent.stat().st_mode & 0o777 == 0o700
    assert database_path.stat().st_mode & 0o777 == 0o600


@POSIX_ONLY
def test_ensure_database__database_exists_with_open_permissions__restricts_to_owner(tmp_path: Path):
    """Set mode 600 on an existing database file with open permissions."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"")
    database_path.chmod(0o644)
    store = SqliteDatabase(database_path)

    store.ensure_database()

    assert database_path.stat().st_mode & 0o777 == 0o600


def test_connect__connection_returned__has_extension_loading_disabled(tmp_path: Path):
    """Return a connection that cannot load further extensions."""
    store = SqliteDatabase(tmp_path / "ostra.db")
    connection = store.connect()

    with pytest.raises(sqlite3.OperationalError, match="not authorized"):
        connection.load_extension(str(tmp_path / "missing_extension"))

    store.close()


def test_connect__connection_open__returns_same_connection(tmp_path: Path):
    """Return the open connection on repeated calls to connect."""
    store = SqliteDatabase(tmp_path / "ostra.db")

    first = store.connect()
    second = store.connect()

    assert first is second
    store.close()


def test_close__connection_open__closes_connection(tmp_path: Path):
    """Close the open connection so that later queries fail."""
    store = SqliteDatabase(tmp_path / "ostra.db")
    connection = store.connect()

    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_close__no_connection_open__does_not_raise(tmp_path: Path):
    """Do nothing when close is called without an open connection."""
    store = SqliteDatabase(tmp_path / "ostra.db")

    store.close()

    assert store.connect() is not None
    store.close()


def test_connect__database_opened__uses_write_ahead_logging(tmp_path: Path):
    """Turn on WAL mode so a reader does not block on a writer."""
    store = SqliteDatabase(tmp_path / "ostra.db")

    mode = store.connect().execute("PRAGMA journal_mode").fetchone()[0]
    store.close()

    assert mode == "wal"
