from ostra.embeddings.base import EmbeddingProvider
from ostra.store.chunk_match import ChunkMatch
from ostra.store.model_index import ModelIndex


class ChunkSearchService:
    """Find the stored chunks that are closest in meaning to a query text."""

    def __init__(self, embedding_provider: EmbeddingProvider, model_index: ModelIndex):
        self._embedding_provider = embedding_provider
        self._model_index = model_index

    def search(self, query: str, limit: int, max_distance: float | None = None) -> list[ChunkMatch]:
        """Embed the query and return the closest stored chunks.

        Args:
            query: The text to search for.
            limit: The number of chunks to return.
            max_distance: Drop the chunks with a larger cosine distance. None keeps every chunk.

        Returns:
            The closest chunks, nearest first.
        """
        matches = self._model_index.search(query=self._embedding_provider.embed_query(query), limit=limit)
        if max_distance is None:
            return matches

        return [match for match in matches if match.distance <= max_distance]
