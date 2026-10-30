from personal_rag.connectors.base import VectorDatabaseConnector


class FileChunkRepository:
    """Store the digest of each processed line through a vector database connector."""

    def __init__(self, connector: VectorDatabaseConnector):
        self._connector = connector

    def get_digests(self, file_path: str) -> set[str]:
        """Return the stored digests of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digests of the lines that were processed.
        """
        return self._connector.get_chunk_digests(file_path=file_path)

    def save_chunk(self, file_path: str, digest: str) -> None:
        """Insert a line digest or update the processed time of a stored digest.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the line.
        """
        self._connector.save_chunk_digest(file_path=file_path, digest=digest)

    def delete_chunks(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored digests that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        self._connector.delete_chunk_digests(file_path=file_path, digests=digests)
