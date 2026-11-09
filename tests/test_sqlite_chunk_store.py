import logging
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import sqlite_vec
from conftest import make_spec

from ostra.embeddings.vector import Vector
from ostra.models.registry import MXBAI_EMBED_LARGE
from ostra.store.model_index_mismatch_error import ModelIndexMismatchError
from ostra.store.sqlite.sqlite_chunk_store import SQLITE_MIGRATIONS_DIRECTORY, SqliteChunkStore

LOGGER_NAME = "ostra.store.sqlite.sqlite_chunk_store"
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission modes do not apply on Windows")


def test_ensure_database__database_missing__creates_file_in_missing_directory(tmp_path: Path):
    """Create the database file and its missing parent directories."""
    database_path = tmp_path / "outer" / "nested" / "ostra.db"
    store = SqliteChunkStore(database_path)

    store.ensure_database()

    assert database_path.is_file()


def test_ensure_database__directory_exists__creates_file_without_error(tmp_path: Path):
    """Create the database file in an existing directory."""
    database_path = tmp_path / "ostra.db"
    store = SqliteChunkStore(database_path)

    store.ensure_database()

    assert database_path.is_file()


def test_ensure_database__database_exists__keeps_file_unchanged(tmp_path: Path):
    """Keep the content of an existing database file unchanged."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"existing content")
    store = SqliteChunkStore(database_path)

    store.ensure_database()

    assert database_path.read_bytes() == b"existing content"


def test_ensure_database__database_exists__does_not_log_creation(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log nothing when the database file already exists."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"")
    store = SqliteChunkStore(database_path)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        store.ensure_database()

    assert caplog.records == []


def test_ensure_database__database_missing__loads_sqlite_vec_extension(tmp_path: Path):
    """Load the sqlite-vec extension into the new database."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    store.ensure_database()

    vec_version = store.connect().execute("SELECT vec_version()").fetchone()[0]
    store.close()

    assert vec_version == f"v{sqlite_vec.__version__}"


def test_ensure_database__database_missing__logs_creation_with_path(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log the creation steps with the database path and the sqlite-vec version."""
    database_path = tmp_path / "ostra.db"
    store = SqliteChunkStore(database_path)

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
    store = SqliteChunkStore(database_path)

    store.ensure_database()

    assert database_path.parent.stat().st_mode & 0o777 == 0o700
    assert database_path.stat().st_mode & 0o777 == 0o600


@POSIX_ONLY
def test_ensure_database__database_exists_with_open_permissions__restricts_to_owner(tmp_path: Path):
    """Set mode 600 on an existing database file with open permissions."""
    database_path = tmp_path / "ostra.db"
    database_path.write_bytes(b"")
    database_path.chmod(0o644)
    store = SqliteChunkStore(database_path)

    store.ensure_database()

    assert database_path.stat().st_mode & 0o777 == 0o600


def test_connect__connection_returned__has_extension_loading_disabled(tmp_path: Path):
    """Return a connection that cannot load further extensions."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    connection = store.connect()

    with pytest.raises(sqlite3.OperationalError, match="not authorized"):
        connection.load_extension(str(tmp_path / "missing_extension"))

    store.close()


def test_connect__connection_open__returns_same_connection(tmp_path: Path):
    """Return the open connection on repeated calls to connect."""
    store = SqliteChunkStore(tmp_path / "ostra.db")

    first = store.connect()
    second = store.connect()

    assert first is second
    store.close()


def test_close__connection_open__closes_connection(tmp_path: Path):
    """Close the open connection so that later queries fail."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    connection = store.connect()

    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_close__no_connection_open__does_not_raise(tmp_path: Path):
    """Do nothing when close is called without an open connection."""
    store = SqliteChunkStore(tmp_path / "ostra.db")

    store.close()

    assert store.connect() is not None
    store.close()


def test_migrate__new_database__creates_documents_table_with_expected_columns(tmp_path: Path):
    """Create the documents table with the document_id, source, title, and processed_at columns."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    store.ensure_database()

    store.migrate()
    columns = store.connect().execute("PRAGMA table_info(documents)").fetchall()
    store.close()

    assert [(column[1], column[2], column[3]) for column in columns] == [
        ("document_id", "TEXT", 0),
        ("source", "TEXT", 1),
        ("title", "TEXT", 1),
        ("processed_at", "TEXT", 1),
    ]


def test_migrate__run_twice__records_each_migration_once(tmp_path: Path):
    """Record each migration once when migrate runs twice."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    store.ensure_database()

    store.migrate()
    store.migrate()
    versions = store.connect().execute("SELECT version FROM schema_migrations").fetchall()
    store.close()

    assert versions == [
        ("001_create_files_table",),
        ("002_create_file_chunks_table",),
        ("003_key_file_chunks_by_digest",),
        ("004_add_chunk_embedding",),
        ("005_create_chunk_vectors",),
        ("006_create_model_indexes",),
    ]


def test_migrate__migration_logged__logs_each_applied_version(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Log the version of each migration that it applies."""
    store = SqliteChunkStore(tmp_path / "ostra.db")
    store.ensure_database()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        store.migrate()

    assert [record.getMessage() for record in caplog.records] == [
        "Applying migration. Version=001_create_files_table",
        "Applying migration. Version=002_create_file_chunks_table",
        "Applying migration. Version=003_key_file_chunks_by_digest",
        "Applying migration. Version=004_add_chunk_embedding",
        "Applying migration. Version=005_create_chunk_vectors",
        "Applying migration. Version=006_create_model_indexes",
    ]


def test_connect__database_opened__uses_write_ahead_logging(tmp_path: Path):
    """Turn on WAL mode so a reader does not block on a writer."""
    store = SqliteChunkStore(tmp_path / "ostra.db")

    mode = store.connect().execute("PRAGMA journal_mode").fetchone()[0]
    store.close()

    assert mode == "wal"


def _build_legacy_database(database_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a database at migration 005 with one file, one chunk and its vector."""
    old_migrations = tmp_path / "old_migrations"
    old_migrations.mkdir()
    for migration in sorted(SQLITE_MIGRATIONS_DIRECTORY.glob("00[1-5]_*.sql")):
        (old_migrations / migration.name).write_text(migration.read_text())

    monkeypatch.setattr("ostra.store.sqlite.sqlite_chunk_store.SQLITE_MIGRATIONS_DIRECTORY", old_migrations)
    store = SqliteChunkStore(database_path)
    store.ensure_database()
    store.migrate()
    connection = store.connect()
    connection.execute("INSERT INTO files (file_path, digest) VALUES ('/notes/a.txt', 'file-digest')")
    connection.execute(
        "INSERT INTO file_chunks (file_path, digest, chunk_text) VALUES ('/notes/a.txt', 'c1', 'legacy')"
    )
    rowid = connection.execute("SELECT rowid FROM file_chunks").fetchone()[0]
    vector = [1.0] + [0.0] * 1023
    connection.execute(
        "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)", (rowid, sqlite_vec.serialize_float32(vector))
    )
    connection.commit()
    store.close()
    monkeypatch.undo()


def test_migrate__legacy_database__keeps_chunks_and_vectors_as_mxbai_index(tmp_path: Path, monkeypatch):
    """Move the old files and chunks into the mxbai index, so that the existing vectors stay searchable."""
    database_path = tmp_path / "ostra.db"
    _build_legacy_database(database_path, tmp_path, monkeypatch)

    with SqliteChunkStore.open(database_path) as store:
        index = store.model_index(MXBAI_EMBED_LARGE, create=False)
        matches = index.search(Vector(MXBAI_EMBED_LARGE.model_id, tuple([1.0] + [0.0] * 1023)), limit=5)

    assert [(m.document_id, m.source, m.title, m.digest, m.text) for m in matches] == [
        ("/notes/a.txt", "filesystem", "/notes/a.txt", "c1", "legacy")
    ]
    assert matches[0].distance == pytest.approx(0.0, abs=1e-6)


def test_migrate__legacy_database__saves_backup_copy_before_changing_tables(tmp_path: Path, monkeypatch):
    """Save a copy of the database that still has the old tables."""
    database_path = tmp_path / "ostra.db"
    _build_legacy_database(database_path, tmp_path, monkeypatch)

    SqliteChunkStore.open(database_path).close()

    backup = sqlite3.connect(tmp_path / "ostra.db.before-006_create_model_indexes")
    assert backup.execute("SELECT file_path FROM files").fetchall() == [("/notes/a.txt",)]
    backup.close()


def test_migrate__new_database__saves_no_backup(tmp_path: Path):
    """Save no copy for a database that has no data yet."""
    SqliteChunkStore.open(tmp_path / "ostra.db").close()

    assert [path.name for path in tmp_path.iterdir() if "before" in path.name] == []


def test_model_index__legacy_index_opened_for_writing__stores_fingerprint(tmp_path: Path, monkeypatch):
    """Stamp the legacy index with the fingerprint of mxbai on the first write open, and refuse other settings later."""
    database_path = tmp_path / "ostra.db"
    _build_legacy_database(database_path, tmp_path, monkeypatch)
    other = replace(MXBAI_EMBED_LARGE, ollama_model="bge-m3:latest")

    with SqliteChunkStore.open(database_path) as store:
        store.model_index(MXBAI_EMBED_LARGE, create=True)
        with pytest.raises(ModelIndexMismatchError):
            store.model_index(other, create=True)


def test_model_index__index_created__has_own_vector_table_with_model_dimensions(tmp_path: Path):
    """Create a vector table that has the dimensions of the new model."""
    with SqliteChunkStore.open(tmp_path / "ostra.db") as store:
        store.model_index(make_spec("small", 3), create=True)
        row = store.connect().execute("SELECT vectors_table FROM model_indexes WHERE model_id = 'small'").fetchone()

        connection = store.connect()
        connection.execute(
            f"INSERT INTO {row[0]} (rowid, embedding) VALUES (1, ?)", (sqlite_vec.serialize_float32([1, 0, 0]),)
        )
        count = connection.execute(f"SELECT count(*) FROM {row[0]}").fetchone()[0]

    assert count == 1
