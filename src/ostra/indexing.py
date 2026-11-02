from collections.abc import Callable
from pathlib import Path

from ostra.config import AppConfig
from ostra.models.embedding_model import EmbeddingModel
from ostra.pipeline.ingestion_pipeline import IngestionPipeline
from ostra.pipeline.ingestion_runner import run_ingestion
from ostra.services.text_chunker import TextChunker
from ostra.sources.base import DocumentSource
from ostra.sources.file_system_source import FileSystemSource
from ostra.store.model_index import ModelIndex


def build_sources(config: AppConfig, directories: list[Path] | None = None) -> list[DocumentSource]:
    """Build the sources that the settings name.

    Args:
        config: The application settings.
        directories: Folders to index. Use the configured folders when None.

    Returns:
        The sources. There is one file system source now.
    """
    return [FileSystemSource(directories=directories or config.directories)]


def build_pipeline(
    config: AppConfig,
    model_index: ModelIndex,
    embedding_model: EmbeddingModel,
    reindex: bool = False,
    on_document_processed: Callable[[str], None] | None = None,
    on_document_started: Callable[[str], None] | None = None,
) -> IngestionPipeline:
    """Build the pipeline that chunks, embeds and stores documents.

    Args:
        config: The application settings.
        model_index: The index that stores the chunks of the embedding model.
        embedding_model: The embedding model and its tokenizer.
        reindex: Embed every chunk again, including chunks that are already stored.
        on_document_processed: Callback that receives the ID of each processed document.
        on_document_started: Callback that receives the ID of each document before its processing starts.

    Returns:
        The ingestion pipeline.
    """
    return IngestionPipeline(
        model_index=model_index,
        chunker=TextChunker(
            chunk_size_tokens=embedding_model.chunk_size_tokens(config.chunk_size_tokens),
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            count_tokens=embedding_model.tokenizer.count_tokens,
        ),
        embedding_provider=embedding_model.provider,
        reindex=reindex,
        on_document_processed=on_document_processed,
        on_document_started=on_document_started,
    )


def run_indexing(
    config: AppConfig,
    model_index: ModelIndex,
    embedding_model: EmbeddingModel,
    directories: list[Path] | None = None,
    reindex: bool = False,
    on_document_processed: Callable[[str], None] | None = None,
    on_document_started: Callable[[str], None] | None = None,
) -> None:
    """Read the documents of every source and embed and store their chunks.

    Args:
        config: The application settings.
        model_index: The index that stores the chunks of the embedding model.
        embedding_model: The embedding model and its tokenizer.
        directories: Folders to index. Use the configured folders when None.
        reindex: Embed every chunk again, including chunks that are already stored.
        on_document_processed: Callback that receives the ID of each processed document.
        on_document_started: Callback that receives the ID of each document before its processing starts.
    """
    pipeline = build_pipeline(
        config=config,
        model_index=model_index,
        embedding_model=embedding_model,
        reindex=reindex,
        on_document_processed=on_document_processed,
        on_document_started=on_document_started,
    )
    for source in build_sources(config, directories):
        run_ingestion(source, pipeline)
