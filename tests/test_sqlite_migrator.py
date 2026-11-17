import logging
import sqlite3
from pathlib import Path

import pytest
from conftest import build_legacy_database

from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.models.vector import Vector
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.repositories.sqlite.sqlite_database import SqliteDatabase
from ostra.repositories.sqlite.sqlite_migrator import SqliteMigrator

LOGGER_NAME = "ostra.repositories.sqlite.sqlite_migrator"


def test_migrate__new_database__creates_documents_table_with_expected_columns(tmp_path: Path):
    """Create the documents table with the document_id, source, title, and processed_at columns."""
    database = SqliteDatabase(tmp_path / "ostra.db")
    database.ensure_database()

    SqliteMigrator(database).migrate()
    columns = database.connect().execute("PRAGMA table_info(documents)").fetchall()
    database.close()

    assert [(column[1], column[2], column[3]) for column in columns] == [
        ("document_id", "TEXT", 0),
        ("source", "TEXT", 1),
        ("title", "TEXT", 1),
        ("processed_at", "TEXT", 1),
    ]


def test_migrate__run_twice__records_each_migration_once(tmp_path: Path):
    """Record each migration once when migrate runs twice."""
    database = SqliteDatabase(tmp_path / "ostra.db")
    database.ensure_database()

    SqliteMigrator(database).migrate()
    SqliteMigrator(database).migrate()
    versions = database.connect().execute("SELECT version FROM schema_migrations").fetchall()
    database.close()

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
    database = SqliteDatabase(tmp_path / "ostra.db")
    database.ensure_database()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        SqliteMigrator(database).migrate()

    assert [record.getMessage() for record in caplog.records] == [
        "Applying migration. Version=001_create_files_table",
        "Applying migration. Version=002_create_file_chunks_table",
        "Applying migration. Version=003_key_file_chunks_by_digest",
        "Applying migration. Version=004_add_chunk_embedding",
        "Applying migration. Version=005_create_chunk_vectors",
        "Applying migration. Version=006_create_model_indexes",
    ]


def test_migrate__legacy_database__keeps_chunks_and_vectors_as_mxbai_index(tmp_path: Path, monkeypatch):
    """Move the old files and chunks into the mxbai index, so that the existing vectors stay searchable."""
    database_path = tmp_path / "ostra.db"
    build_legacy_database(database_path, tmp_path, monkeypatch)

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
    build_legacy_database(database_path, tmp_path, monkeypatch)

    SqliteChunkStore.open(database_path).close()

    backup = sqlite3.connect(tmp_path / "ostra.db.before-006_create_model_indexes")
    assert backup.execute("SELECT file_path FROM files").fetchall() == [("/notes/a.txt",)]
    backup.close()


def test_migrate__new_database__saves_no_backup(tmp_path: Path):
    """Save no copy for a database that has no data yet."""
    SqliteChunkStore.open(tmp_path / "ostra.db").close()

    assert [path.name for path in tmp_path.iterdir() if "before" in path.name] == []
