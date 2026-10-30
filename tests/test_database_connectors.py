from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from ostra.connectors.database_facade import DatabaseBackend, DatabaseFacade

EMBEDDING = [0.1] * 1024


def one_hot(index: int) -> list[float]:
    """Return a 1024-value vector with a single 1.0 at the index, so that vectors differ in direction."""
    vector = [0.0] * 1024
    vector[index] = 1.0
    return vector


def test_get_file_digest__file_not_saved__returns_none(connector: DatabaseFacade):
    """Return None for a file that no run has saved."""
    assert connector.get_file_digest("/notes/a.txt") is None


def test_save_file_digest__new_file__returns_saved_digest(connector: DatabaseFacade):
    """Return the digest that was saved for a file."""
    connector.save_file_digest("/notes/a.txt", "digest-1")

    assert connector.get_file_digest("/notes/a.txt") == "digest-1"


def test_save_file_digest__saved_again__returns_latest_digest(connector: DatabaseFacade):
    """Replace the stored digest when the same file is saved again."""
    connector.save_file_digest("/notes/a.txt", "digest-1")

    connector.save_file_digest("/notes/a.txt", "digest-2")

    assert connector.get_file_digest("/notes/a.txt") == "digest-2"


def test_get_chunk_digests__no_chunks__returns_empty_set(connector: DatabaseFacade):
    """Return an empty set for a file without saved chunks."""
    assert connector.get_chunk_digests("/notes/a.txt") == set()


def test_save_chunk_digest__two_digests__returns_both(connector: DatabaseFacade):
    """Return every chunk digest that was saved for the file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)
    connector.save_chunk_digest("/notes/a.txt", "chunk-2", "chunk text", EMBEDDING)

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1", "chunk-2"}


def test_save_chunk_digest__same_digest_twice__returns_one(connector: DatabaseFacade):
    """Store a repeated chunk digest only once for the file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_get_chunk_digests__other_file__not_returned(connector: DatabaseFacade):
    """Return only the chunk digests of the requested file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)
    connector.save_chunk_digest("/notes/b.txt", "chunk-2", "chunk text", EMBEDDING)

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_delete_chunk_digests__some_digests__removes_only_those(connector: DatabaseFacade):
    """Delete the given chunk digests and keep the others."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)
    connector.save_chunk_digest("/notes/a.txt", "chunk-2", "chunk text", EMBEDDING)

    connector.delete_chunk_digests("/notes/a.txt", {"chunk-1"})

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-2"}


def test_delete_chunk_digests__empty_set__keeps_chunks(connector: DatabaseFacade):
    """Keep every chunk when no digest is given to delete."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)

    connector.delete_chunk_digests("/notes/a.txt", set())

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_delete_chunk_digests__digest_not_stored__raises_no_error(connector: DatabaseFacade):
    """Ignore a digest that is not stored."""
    connector.delete_chunk_digests("/notes/a.txt", {"missing"})

    assert connector.get_chunk_digests("/notes/a.txt") == set()


def test_save_then_reopen__data_persisted(backend: DatabaseBackend, database_path: Path, connector: DatabaseFacade):
    """Read back the saved data from a new facade that opens the same store."""
    connector.save_file_digest("/notes/a.txt", "digest-1")
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "chunk text", EMBEDDING)
    connector.close()
    # Chroma keeps one shared system per path inside the process. Clear it so the reopened client reads the disk.
    SharedSystemClient.clear_system_cache()

    reopened = DatabaseFacade(backend=backend, database_path=database_path)
    try:
        assert reopened.get_file_digest("/notes/a.txt") == "digest-1"
        assert reopened.get_chunk_digests("/notes/a.txt") == {"chunk-1"}
    finally:
        reopened.close()
        SharedSystemClient.clear_system_cache()


@pytest.mark.parametrize("digest", ["chunk-1", "chunk-2"])
def test_save_chunk_digest__multiple_files__keeps_files_apart(connector: DatabaseFacade, digest: str):
    """Keep the chunk digests of different files apart even when the digest is the same."""
    connector.save_chunk_digest("/notes/a.txt", digest, "chunk text", EMBEDDING)
    connector.save_chunk_digest("/notes/b.txt", digest, "chunk text", EMBEDDING)

    assert connector.get_chunk_digests("/notes/a.txt") == {digest}
    assert connector.get_chunk_digests("/notes/b.txt") == {digest}


def test_search_chunks__query_matches_one_chunk__returns_nearest_first(connector: DatabaseFacade):
    """Return the chunk with the same direction first, with its text and a distance of 0."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "first text", one_hot(1))
    connector.save_chunk_digest("/notes/a.txt", "chunk-2", "second text", one_hot(2))

    matches = connector.search_chunks(query_embedding=one_hot(1), limit=2)

    assert [(match.digest, match.text) for match in matches] == [("chunk-1", "first text"), ("chunk-2", "second text")]
    assert matches[0].distance == pytest.approx(0.0, abs=1e-6)
    assert matches[1].distance == pytest.approx(1.0, abs=1e-6)


def test_search_chunks__limit_reached__returns_only_limit_matches(connector: DatabaseFacade):
    """Return no more matches than the limit."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "first text", one_hot(1))
    connector.save_chunk_digest("/notes/a.txt", "chunk-2", "second text", one_hot(2))

    matches = connector.search_chunks(query_embedding=one_hot(1), limit=1)

    assert [match.digest for match in matches] == ["chunk-1"]


def test_search_chunks__no_chunks_stored__returns_empty_list(connector: DatabaseFacade):
    """Return an empty list when no chunk is stored."""
    assert connector.search_chunks(query_embedding=one_hot(1), limit=5) == []


def test_search_chunks__chunk_deleted__is_not_returned(connector: DatabaseFacade):
    """Stop returning a chunk after it is deleted, so its vector is removed too."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "first text", one_hot(1))
    connector.save_chunk_digest("/notes/a.txt", "chunk-2", "second text", one_hot(2))

    connector.delete_chunk_digests("/notes/a.txt", {"chunk-1"})
    matches = connector.search_chunks(query_embedding=one_hot(1), limit=5)

    assert [match.digest for match in matches] == ["chunk-2"]


def test_save_chunk_digest__digest_saved_again__replaces_text_and_vector(connector: DatabaseFacade):
    """Replace the text and the vector of a stored chunk when the same digest is saved again."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "old text", one_hot(1))

    connector.save_chunk_digest("/notes/a.txt", "chunk-1", "new text", one_hot(2))
    matches = connector.search_chunks(query_embedding=one_hot(2), limit=5)

    assert [(match.text, match.distance) for match in matches] == [("new text", pytest.approx(0.0, abs=1e-6))]
