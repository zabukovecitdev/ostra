import argparse

from ostra.cli.progress import ProgressReporter
from ostra.cli.search_output import SearchOutputOptions, print_search_results
from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.index_session import IndexSession
from ostra.services.ingestion.indexing_service import run_indexing
from ostra.services.ingestion.source_factory import build_sources


def run_index(arguments: argparse.Namespace, config: AppConfig, session: IndexSession) -> None:
    """Embed and store the new or changed chunks of the folders.

    Args:
        arguments: The parsed index arguments.
        config: The application settings.
        session: The open embedding model and its index.
    """
    print("Database ready.")
    given = [*arguments.folders, *(arguments.directory or [])]
    directories = [path.expanduser() for path in given] or None
    if not (directories or config.directories):
        print(NO_DIRECTORIES_MESSAGE)
        return

    sources = build_sources(config, directories=directories, recursive=arguments.recursive)
    with ProgressReporter(sources) as progress:
        run_indexing(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            model_index=session.model_index,
            embedding_model=session.embedding_model,
            sources=sources,
            reindex=arguments.reindex,
            on_document_started=progress.document_started,
            on_document_processed=progress.document_processed,
        )


def run_search(arguments: argparse.Namespace, config: AppConfig, session: IndexSession) -> None:
    """Print the stored chunks that are closest to the query, nearest first.

    Args:
        arguments: The parsed search arguments.
        config: The application settings.
        session: The open embedding model and its index.
    """
    service = ChunkSearchService(embedding_provider=session.embedding_model.provider, model_index=session.model_index)
    matches = service.search(query=arguments.query, limit=arguments.limit, max_distance=arguments.max_distance)
    print_search_results(matches, SearchOutputOptions(full_text=arguments.full, as_json=arguments.json))
