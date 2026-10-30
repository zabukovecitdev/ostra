from ostra.connectors.base import ChunkMatch, VectorDatabaseConnector


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

    def save_chunk(self, file_path: str, digest: str, text: str, embedding: list[float]) -> None:
        """Insert a chunk and its embedding, or update both when the digest is already stored.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.
            text: The text of the chunk.
            embedding: The embedding vector of the chunk.
        """
        self._connector.save_chunk_digest(file_path=file_path, digest=digest, text=text, embedding=embedding)

    def search(self, query_embedding: list[float], limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query_embedding: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first.
        """
        return self._connector.search_chunks(query_embedding=query_embedding, limit=limit)

    def delete_chunks(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored digests that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        self._connector.delete_chunk_digests(file_path=file_path, digests=digests)
