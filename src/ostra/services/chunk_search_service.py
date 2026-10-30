from ostra.connectors.base import ChunkMatch
from ostra.embeddings.base import EmbeddingProvider
from ostra.repositories.file_chunk_repository import FileChunkRepository


class ChunkSearchService:
    """Find the stored chunks that are closest in meaning to a query text."""

    def __init__(self, embedding_provider: EmbeddingProvider, file_chunk_repository: FileChunkRepository):
        self._embedding_provider = embedding_provider
        self._file_chunk_repository = file_chunk_repository

    def search(self, query: str, limit: int) -> list[ChunkMatch]:
        """Embed the query and return the closest stored chunks.

        Args:
            query: The text to search for.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first.
        """
        query_embedding = self._embedding_provider.get_embedding(query)
        return self._file_chunk_repository.search(query_embedding=query_embedding, limit=limit)
