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
        connection = self._connector.connect()
        connection.execute(
            "INSERT INTO files (file_path, digest) VALUES (?, ?) "
            "ON CONFLICT (file_path) DO UPDATE SET digest = excluded.digest, processed_at = CURRENT_TIMESTAMP",
            (file_path, digest),
        )
        connection.commit()
