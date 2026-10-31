from collections.abc import Callable
from pathlib import Path

from ostra.config import AppConfig
from ostra.connectors.base import VectorDatabaseConnector
from ostra.models.base import EmbeddingModel
from ostra.repositories.file_chunk_repository import FileChunkRepository
from ostra.repositories.file_repository import FileRepository
from ostra.services.file_ingestion_service import FileIngestionService
from ostra.services.file_service import FileService
from ostra.services.text_chunker import TextChunker


def build_ingestion_service(
    config: AppConfig,
    connector: VectorDatabaseConnector,
    embedding_model: EmbeddingModel,
    directories: list[Path] | None = None,
    reindex: bool = False,
    on_file_processed: Callable[[Path], None] | None = None,
) -> FileIngestionService:
    """Build the service that embeds and stores the chunks of the text files.

    Args:
        config: The application settings.
        connector: The database connector that stores the chunks.
        embedding_model: The embedding model and its tokenizer.
        directories: Folders to index. Use the configured folders when None.
        reindex: Embed every chunk again, including chunks that are already stored.
        on_file_processed: Callback that receives the path of each processed file.

    Returns:
        The ingestion service.
    """
    tokenizer = embedding_model.get_tokenizer()
    return FileIngestionService(
        file_service=FileService(directories=directories or config.directories),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
        chunker=TextChunker(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            count_tokens=tokenizer.count_tokens,
        ),
        embedding_provider=embedding_model.get_embedding_provider(),
        reindex=reindex,
        on_file_processed=on_file_processed,
    )
