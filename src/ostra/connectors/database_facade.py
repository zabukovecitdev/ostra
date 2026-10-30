from enum import StrEnum
from pathlib import Path
from typing import Any

from ostra.connectors.base import ChunkMatch, VectorDatabaseConnector
from ostra.connectors.sqlite_connector import SqliteConnector, SqliteSettings

CHROMA_DIRECTORY_NAME = "chroma"


class DatabaseBackend(StrEnum):
    """Name the database backends that the application can use."""

    SQLITE = "sqlite"
    CHROMA = "chroma"


class DatabaseFacade(VectorDatabaseConnector):
    """Give the services one connector and choose the backend behind it."""

    def __init__(self, backend: DatabaseBackend, database_path: Path):
        self._connector = _create_connector(backend=backend, database_path=database_path)

    def ensure_database(self) -> None:
        """Create the database or collection when it is missing."""
        self._connector.ensure_database()

    def connect(self) -> Any:
        """Open a connection with the vector support loaded.

        Returns:
            An open connection of the selected backend.
        """
        return self._connector.connect()

    def close(self) -> None:
        """Release the open connection."""
        self._connector.close()

    def migrate(self) -> None:
        """Apply the schema migrations that have not run yet."""
        self._connector.migrate()

    def save_file_digest(self, file_path: str, digest: str) -> None:
        """Insert a file or update its digest and processed time.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the file content.
        """
        self._connector.save_file_digest(file_path=file_path, digest=digest)

    def get_file_digest(self, file_path: str) -> str | None:
        """Return the stored digest of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digest of the file content, or None when the file is not stored.
        """
        return self._connector.get_file_digest(file_path=file_path)

    def get_chunk_digests(self, file_path: str) -> set[str]:
        """Return the stored chunk digests of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digests of the chunks that were processed.
        """
        return self._connector.get_chunk_digests(file_path=file_path)

    def save_chunk_digest(self, file_path: str, digest: str, text: str, embedding: list[float]) -> None:
        """Insert a chunk and its embedding, or update both when the digest is already stored.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.
            text: The text of the chunk.
            embedding: The embedding vector of the chunk.
        """
        self._connector.save_chunk_digest(file_path=file_path, digest=digest, text=text, embedding=embedding)

    def get_chunk_embedding(self, file_path: str, digest: str) -> list[float] | None:
        """Return the stored embedding of one chunk.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.

        Returns:
            The embedding vector of the chunk, or None when the chunk is not stored.
        """
        return self._connector.get_chunk_embedding(file_path=file_path, digest=digest)

    def search_chunks(self, query_embedding: list[float], limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query_embedding: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first.
        """
        return self._connector.search_chunks(query_embedding=query_embedding, limit=limit)

    def delete_chunk_digests(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored chunks that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        self._connector.delete_chunk_digests(file_path=file_path, digests=digests)


def _create_connector(backend: DatabaseBackend, database_path: Path) -> VectorDatabaseConnector:
    """Build the connector of the selected backend.

    The Chroma connector loads its package only when that backend is selected. The Chroma store is a directory
    named chroma beside the SQLite database file.

    Args:
        backend: The backend to build.
        database_path: Location of the SQLite database file. The Chroma store uses its parent directory.

    Returns:
        A connector of the selected backend.

    Raises:
        ValueError: When the backend has no connector.
    """
    if backend is DatabaseBackend.SQLITE:
        return SqliteConnector(SqliteSettings(path=database_path))

    if backend is DatabaseBackend.CHROMA:
        from ostra.connectors.chroma_connector import ChromaConnector, ChromaSettings

        return ChromaConnector(ChromaSettings(path=database_path.parent / CHROMA_DIRECTORY_NAME))

    raise ValueError(f"Database backend has no connector. Backend={backend}")
