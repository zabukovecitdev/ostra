from ostra.config import AppConfig
from ostra.connectors.base import ChunkMatch
from ostra.connectors.database_facade import DatabaseFacade
from ostra.models.factory import build_embedding_model
from ostra.repositories.file_chunk_repository import FileChunkRepository
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.database_service import DatabaseService


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
    connector = DatabaseFacade(backend=config.database_backend, database_path=config.database_path)
    try:
        DatabaseService(connector=connector).initialize()
        service = ChunkSearchService(
            embedding_provider=build_embedding_model(config).get_embedding_provider(),
            file_chunk_repository=FileChunkRepository(connector=connector),
        )
        matches = service.search(query=query, limit=limit)
    finally:
        connector.close()

    if max_distance is None:
        return matches

    return [match for match in matches if match.distance <= max_distance]
