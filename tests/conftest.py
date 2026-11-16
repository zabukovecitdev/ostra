from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlite_vec

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.repositories.sqlite.sqlite_database import SqliteDatabase
from ostra.repositories.sqlite.sqlite_migrator import SQLITE_MIGRATIONS_DIRECTORY, SqliteMigrator
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex


def make_spec(model_id: str, dimensions: int, **overrides) -> EmbeddingModelSpec:
    """Build a model spec for a test. A small vector keeps the tests light."""
    values = {
        "model_id": model_id,
        "ollama_model": f"{model_id}:latest",
        "dimensions": dimensions,
        "max_tokens": 512,
        "special_tokens": 2,
        "default_chunk_size_tokens": 510,
        "tokenizer_file": "mxbai-embed-large-v1/tokenizer.json",
    }
    return EmbeddingModelSpec(**(values | overrides))


TEST_SPEC = make_spec("test-model", 4)


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide the location of the SQLite database file for a test."""
    return tmp_path / "ostra.db"


@pytest.fixture
def store(database_path: Path) -> Iterator[SqliteChunkStore]:
    """Provide an open store and close it after the test."""
    with SqliteChunkStore.open(database_path) as opened:
        yield opened


@pytest.fixture
def index(store: SqliteChunkStore) -> SqliteModelIndex:
    """Provide the index of the small test model."""
    return store.model_index(TEST_SPEC, create=True)


def build_legacy_database(database_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a database at migration 005 with one file, one chunk and its vector."""
    old_migrations = tmp_path / "old_migrations"
    old_migrations.mkdir()
    for migration in sorted(SQLITE_MIGRATIONS_DIRECTORY.glob("00[1-5]_*.sql")):
        (old_migrations / migration.name).write_text(migration.read_text())

    monkeypatch.setattr("ostra.repositories.sqlite.sqlite_migrator.SQLITE_MIGRATIONS_DIRECTORY", old_migrations)
    database = SqliteDatabase(database_path)
    database.ensure_database()
    SqliteMigrator(database).migrate()
    connection = database.connect()
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
    database.close()
    monkeypatch.undo()
