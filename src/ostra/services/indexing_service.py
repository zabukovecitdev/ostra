from collections.abc import Callable
from functools import partial
from pathlib import Path

from tqdm import tqdm

from ostra.models.app_config import AppConfig
from ostra.models.embedding_model import EmbeddingModel
from ostra.providers.sources.document_source import DocumentSource
from ostra.providers.sources.file_system_source import FileSystemSource
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex
from ostra.services.ingestion_pipeline import IngestionPipeline
from ostra.services.ingestion_runner import run_ingestion
from ostra.services.text_chunker import TextChunker


def build_pipeline(
    config: AppConfig,
    model_index: SqliteModelIndex,
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
    model_index: SqliteModelIndex,
    embedding_model: EmbeddingModel,
    directories: list[Path] | None = None,
    reindex: bool = False,
    on_document_processed: Callable[[str], None] | None = None,
    recursive: bool = False,
    show_progress: bool = False,
) -> None:
    """Read the documents of every source and embed and store their chunks.

    Args:
        config: The application settings.
        model_index: The index that stores the chunks of the embedding model.
        embedding_model: The embedding model and its tokenizer.
        directories: Folders to index. Use the configured folders when None.
        reindex: Embed every chunk again, including chunks that are already stored.
        on_document_processed: Callback that receives the ID of each processed document.
        recursive: Index the files in the subfolders of the folders too.
        show_progress: Show a progress bar on the terminal, with the count of documents and the current document.
    """
    sources: list[DocumentSource] = [
        FileSystemSource(directories=directories or config.directories, recursive=recursive)
    ]
    progress = _start_progress(sources) if show_progress else None
    pipeline = build_pipeline(
        config=config,
        model_index=model_index,
        embedding_model=embedding_model,
        reindex=reindex,
        on_document_processed=_chain(on_document_processed, partial(_advance, progress) if progress else None),
        on_document_started=partial(_show_document, progress) if progress else None,
    )
    try:
        for source in sources:
            run_ingestion(source, pipeline)
    finally:
        if progress is not None:
            progress.close()


def _start_progress(sources: list[DocumentSource]) -> tqdm:
    """Start the progress bar with the total count of documents, when every source knows it.

    tqdm hides the bar when the output is not a terminal.

    Args:
        sources: The sources that will be indexed.

    Returns:
        The progress bar, which writes to stderr.
    """
    counts = [source.count_documents() for source in sources]
    known = [count for count in counts if count is not None]
    total = sum(known) if len(known) == len(counts) else None
    return tqdm(total=total, unit="doc", desc="Indexing", disable=None)


def _advance(progress: tqdm, document_id: str) -> None:
    """Count one processed document on the progress bar.

    Args:
        progress: The progress bar.
        document_id: The ID of the processed document. Not shown, because the next document replaces it.
    """
    progress.update(1)


def _show_document(progress: tqdm, document_id: str) -> None:
    """Show the ID of the document that is being processed after the bar.

    Args:
        progress: The progress bar.
        document_id: The ID of the document.
    """
    progress.set_postfix_str(document_id, refresh=True)


def _chain(first: Callable[[str], None] | None, second: Callable[[str], None] | None) -> Callable[[str], None] | None:
    """Return one callback that calls both callbacks, the one that is set, or None.

    Args:
        first: The first callback, or None.
        second: The second callback, or None.

    Returns:
        A callback that calls the callbacks that are set, in order.
    """
    callbacks = [callback for callback in (first, second) if callback is not None]
    if not callbacks:
        return None

    def call_all(document_id: str) -> None:
        for callback in callbacks:
            callback(document_id)

    return call_all
