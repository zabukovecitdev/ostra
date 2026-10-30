from personal_rag.connectors.base import VectorDatabaseConnector


class FileRepository:
    """Store the digest of each processed file through a vector database connector."""

    def __init__(self, connector: VectorDatabaseConnector):
        self._connector = connector

    def save_file(self, file_path: str, digest: str) -> None:
        """Insert a file or update its digest and processed time.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the file content.
        """
        self._connector.save_file_digest(file_path=file_path, digest=digest)

    def get_digest(self, file_path: str) -> str | None:
        """Return the stored digest of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digest of the file content, or None when the file is not stored.
        """
        return self._connector.get_file_digest(file_path=file_path)
