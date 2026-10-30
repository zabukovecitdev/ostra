from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkMatch:
    """Hold one stored chunk returned by a vector search."""

    file_path: str
    digest: str
    text: str
    distance: float


class VectorDatabaseConnector(ABC):
    """Define the interface that every vector database connector implements."""

    @abstractmethod
    def ensure_database(self) -> None:
        """Create the database or collection when it is missing."""

    @abstractmethod
    def connect(self) -> Any:
        """Open a connection with the vector support loaded.

        Returns:
            An open connection.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the open connection."""

    @abstractmethod
    def migrate(self) -> None:
        """Apply the schema migrations that have not run yet."""

    @abstractmethod
    def save_file_digest(self, file_path: str, digest: str) -> None:
        """Insert a file or update its digest and processed time.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the file content.
        """

    @abstractmethod
    def get_file_digest(self, file_path: str) -> str | None:
        """Return the stored digest of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digest of the file content, or None when the file is not stored.
        """

    @abstractmethod
    def get_chunk_digests(self, file_path: str) -> set[str]:
        """Return the stored chunk digests of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digests of the chunks that were processed.
        """

    @abstractmethod
    def save_chunk_digest(self, file_path: str, digest: str, text: str, embedding: list[float]) -> None:
        """Insert a chunk and its embedding, or update both when the digest is already stored.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.
            text: The text of the chunk.
            embedding: The embedding vector of the chunk.
        """

    @abstractmethod
    def get_chunk_embedding(self, file_path: str, digest: str) -> list[float] | None:
        """Return the stored embedding of one chunk.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.

        Returns:
            The embedding vector of the chunk, or None when the chunk is not stored.
        """

    @abstractmethod
    def search_chunks(self, query_embedding: list[float], limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query_embedding: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first.
        """

    @abstractmethod
    def delete_chunk_digests(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored chunk digests that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
