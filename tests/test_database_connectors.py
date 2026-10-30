from pathlib import Path

import pytest

from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings


def test_get_file_digest__file_not_saved__returns_none(connector: SqliteConnector):
    """Return None for a file that no run has saved."""
    assert connector.get_file_digest("/notes/a.txt") is None


def test_save_file_digest__new_file__returns_saved_digest(connector: SqliteConnector):
    """Return the digest that was saved for a file."""
    connector.save_file_digest("/notes/a.txt", "digest-1")

    assert connector.get_file_digest("/notes/a.txt") == "digest-1"


def test_save_file_digest__saved_again__returns_latest_digest(connector: SqliteConnector):
    """Replace the stored digest when the same file is saved again."""
    connector.save_file_digest("/notes/a.txt", "digest-1")

    connector.save_file_digest("/notes/a.txt", "digest-2")

    assert connector.get_file_digest("/notes/a.txt") == "digest-2"


def test_get_chunk_digests__no_chunks__returns_empty_set(connector: SqliteConnector):
    """Return an empty set for a file without saved chunks."""
    assert connector.get_chunk_digests("/notes/a.txt") == set()


def test_save_chunk_digest__two_digests__returns_both(connector: SqliteConnector):
    """Return every chunk digest that was saved for the file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")
    connector.save_chunk_digest("/notes/a.txt", "chunk-2")

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1", "chunk-2"}


def test_save_chunk_digest__same_digest_twice__returns_one(connector: SqliteConnector):
    """Store a repeated chunk digest only once for the file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_get_chunk_digests__other_file__not_returned(connector: SqliteConnector):
    """Return only the chunk digests of the requested file."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")
    connector.save_chunk_digest("/notes/b.txt", "chunk-2")

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_delete_chunk_digests__some_digests__removes_only_those(connector: SqliteConnector):
    """Delete the given chunk digests and keep the others."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")
    connector.save_chunk_digest("/notes/a.txt", "chunk-2")

    connector.delete_chunk_digests("/notes/a.txt", {"chunk-1"})

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-2"}


def test_delete_chunk_digests__empty_set__keeps_chunks(connector: SqliteConnector):
    """Keep every chunk when no digest is given to delete."""
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")

    connector.delete_chunk_digests("/notes/a.txt", set())

    assert connector.get_chunk_digests("/notes/a.txt") == {"chunk-1"}


def test_delete_chunk_digests__digest_not_stored__raises_no_error(connector: SqliteConnector):
    """Ignore a digest that is not stored."""
    connector.delete_chunk_digests("/notes/a.txt", {"missing"})

    assert connector.get_chunk_digests("/notes/a.txt") == set()


def test_save_then_reopen__data_persisted(database_path: Path, connector: SqliteConnector):
    """Read back the saved digests from a new connector that opens the same store."""
    connector.save_file_digest("/notes/a.txt", "digest-1")
    connector.save_chunk_digest("/notes/a.txt", "chunk-1")
    connector.close()

    reopened = SqliteConnector(SqliteSettings(path=database_path))
    try:
        assert reopened.get_file_digest("/notes/a.txt") == "digest-1"
        assert reopened.get_chunk_digests("/notes/a.txt") == {"chunk-1"}
    finally:
        reopened.close()


@pytest.mark.parametrize("digest", ["chunk-1", "chunk-2"])
def test_save_chunk_digest__multiple_files__keeps_files_apart(connector: SqliteConnector, digest: str):
    """Keep the chunk digests of different files apart even when the digest is the same."""
    connector.save_chunk_digest("/notes/a.txt", digest)
    connector.save_chunk_digest("/notes/b.txt", digest)

    assert connector.get_chunk_digests("/notes/a.txt") == {digest}
    assert connector.get_chunk_digests("/notes/b.txt") == {digest}
