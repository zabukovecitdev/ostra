import sqlite3
from pathlib import Path

from conftest import make_spec

from ostra.models.document_record import DocumentRecord
from ostra.models.embedded_chunk import EmbeddedChunk
from ostra.models.vector import Vector
from ostra.repositories.sqlite.index_summary import indexed_document_counts
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore


def save_document(index, model_id: str, document_id: str, digests: list[str]) -> None:
    """Save one document with one chunk for each digest, with a vector of the model."""
    chunks = [
        EmbeddedChunk(digest=digest, text="text", embedding=Vector(model_id=model_id, values=(1.0, 0.0, 0.0, 0.0)))
        for digest in digests
    ]
    index.save_document(
        DocumentRecord(document_id=document_id, source="filesystem", title=document_id),
        chunks,
        current_digests={chunk.digest for chunk in chunks},
    )


def test_indexed_document_counts__missing_file__returns_empty_and_creates_nothing(tmp_path: Path):
    """Return no counts and leave the missing file missing, so that a look at the settings changes nothing."""
    database_path = tmp_path / "ostra.db"

    assert indexed_document_counts(database_path) == {}
    assert not database_path.exists()


def test_indexed_document_counts__two_models__counts_documents_not_chunks(tmp_path: Path):
    """Count the distinct documents of each model index, so that several chunks of one document count once."""
    database_path = tmp_path / "ostra.db"
    small = make_spec("small", 4)
    with SqliteChunkStore.open(database_path) as store:
        save_document(store.model_index(small, create=True), "small", "/notes/a.txt", ["a1", "a2", "a3"])
        save_document(store.model_index(small, create=True), "small", "/notes/b.txt", ["b1"])
        large = make_spec("large", 4)
        save_document(store.model_index(large, create=True), "large", "/notes/a.txt", ["a1"])

    # The migration also creates the index of the legacy mxbai-embed-large model, which has no chunks.
    assert indexed_document_counts(database_path) == {"mxbai-embed-large": 0, "small": 2, "large": 1}


def test_indexed_document_counts__model_index_without_chunks__counts_zero(tmp_path: Path):
    """Report a model index that exists but holds no chunks with zero documents."""
    database_path = tmp_path / "ostra.db"
    with SqliteChunkStore.open(database_path) as store:
        store.model_index(make_spec("empty", 4), create=True)

    assert indexed_document_counts(database_path)["empty"] == 0


def test_indexed_document_counts__database_without_model_indexes__returns_empty(tmp_path: Path):
    """Return no counts for an old database that has no model_indexes table, and leave its tables alone."""
    database_path = tmp_path / "ostra.db"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE file_chunks (file_path TEXT)")
    connection.commit()
    connection.close()

    assert indexed_document_counts(database_path) == {}
    connection = sqlite3.connect(database_path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    connection.close()
    assert tables == {"file_chunks"}
