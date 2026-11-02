from ostra.config import AppConfig
from ostra.models.factory import build_embedding_model
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.store.chunk_match import ChunkMatch
from ostra.store.open_chunk_store import open_chunk_store


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
    embedding_model = build_embedding_model(config)
    with open_chunk_store(config.database_backend, config.database_path) as store:
        service = ChunkSearchService(
            embedding_provider=embedding_model.provider,
            model_index=store.model_index(embedding_model.spec, create=False),
        )
        return service.search(query=query, limit=limit, max_distance=max_distance)
