from dataclasses import replace
from pathlib import Path

import pytest
import sqlite_vec
from conftest import build_legacy_database, make_spec

from ostra.errors.model_index_mismatch_error import ModelIndexMismatchError
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore


def test_model_index__legacy_index_opened_for_writing__stores_fingerprint(tmp_path: Path, monkeypatch):
    """Stamp the legacy index with the fingerprint of mxbai on the first write open, and refuse other settings later."""
    database_path = tmp_path / "ostra.db"
    build_legacy_database(database_path, tmp_path, monkeypatch)
    other = replace(MXBAI_EMBED_LARGE, provider_model="bge-m3:latest")

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
