from collections.abc import Callable

from ostra.models.app_config import AppConfig
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.model_index import ChunkWriteIndex
from ostra.services.embedding.embedding_model import EmbeddingModel
from ostra.services.ingestion.ingestion_pipeline import IngestionPipeline
from ostra.services.ingestion.ingestion_runner import run_ingestion
from ostra.services.ingestion.text_chunker import TextChunker


def build_pipeline(
    config: AppConfig,
    model_index: ChunkWriteIndex,
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
        on_document_started: Callback that receives the ID of each document before it is processed.

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
    model_index: ChunkWriteIndex,
    embedding_model: EmbeddingModel,
    sources: list[DocumentSource],
    reindex: bool = False,
    on_document_processed: Callable[[str], None] | None = None,
    on_document_started: Callable[[str], None] | None = None,
) -> None:
    """Read the documents of every source and embed and store their chunks.

    Args:
        config: The application settings.
        model_index: The index that stores the chunks of the embedding model.
        embedding_model: The embedding model and its tokenizer.
        sources: The sources whose documents are indexed, in order.
        reindex: Embed every chunk again, including chunks that are already stored.
        on_document_processed: Callback that receives the ID of each processed document.
        on_document_started: Callback that receives the ID of each document before it is processed.
    """
    pipeline = build_pipeline(
        config=config,
        model_index=model_index,
        embedding_model=embedding_model,
        reindex=reindex,
        on_document_processed=on_document_processed,
        on_document_started=on_document_started,
    )
    for source in sources:
        run_ingestion(source, pipeline)
