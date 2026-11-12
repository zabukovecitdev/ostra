from ostra.models.app_config import AppConfig
from ostra.models.chunk_match import ChunkMatch
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex
from ostra.services.index_session import open_index


class ChunkSearchService:
    """Find the stored chunks that are closest in meaning to a query text."""

    def __init__(self, embedding_provider: EmbeddingProvider, model_index: SqliteModelIndex):
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


def search_chunks(config: AppConfig, query: str, limit: int, max_distance: float | None) -> list[ChunkMatch]:
    """Search the stored chunks for a query. Open and close the database for this one call.

    Args:
        config: The application settings.
        query: The text to search for.
        limit: The number of chunks to return.
        max_distance: Drop the chunks with a larger cosine distance. None keeps every chunk.

    Returns:
        The closest chunks, nearest first.
    """
    with open_index(config) as session:
        service = ChunkSearchService(
            embedding_provider=session.embedding_model.provider, model_index=session.model_index
        )
        return service.search(query=query, limit=limit, max_distance=max_distance)
