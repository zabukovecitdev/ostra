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
        connection = self._connector.connect()
        rows = connection.execute("SELECT digest FROM file_chunks WHERE file_path = ?", (file_path,)).fetchall()
        return {row[0] for row in rows}

    def save_chunk(self, file_path: str, digest: str) -> None:
        """Insert a line digest or update the processed time of a stored digest.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the line.
        """
        connection = self._connector.connect()
        connection.execute(
            "INSERT INTO file_chunks (file_path, digest) VALUES (?, ?) "
            "ON CONFLICT (file_path, digest) DO UPDATE SET processed_at = CURRENT_TIMESTAMP",
            (file_path, digest),
        )
        connection.commit()

    def delete_chunks(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored digests that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        connection = self._connector.connect()
        connection.executemany(
            "DELETE FROM file_chunks WHERE file_path = ? AND digest = ?",
            [(file_path, digest) for digest in digests],
        )
        connection.commit()
