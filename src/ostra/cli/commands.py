import argparse
import sys

from ostra.cli.init.prompter import DefaultsPrompter, Prompter
from ostra.cli.init.questionary_prompter import QuestionaryPrompter
from ostra.cli.init.wizard import run_setup
from ostra.cli.progress import ProgressReporter
from ostra.cli.search_output import SearchOutputOptions, print_search_results
from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.config_repository import default_config_path
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.index_session import IndexSession, open_index
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
    _index_with_progress(config, session, sources, reindex=arguments.reindex)


def run_init(arguments: argparse.Namespace) -> None:
    """Ask the setup questions, save the settings and offer to index the folders right away.

    Args:
        arguments: The parsed init arguments.

    Raises:
        SystemExit: When the questions cannot be asked, because the input is not a terminal.
    """
    if not (arguments.defaults or sys.stdin.isatty()):
        raise SystemExit("ostra init needs a terminal. Use ostra init --defaults, or edit the config file.")

    prompter: Prompter = DefaultsPrompter() if arguments.defaults else QuestionaryPrompter()
    config = run_setup(prompter, default_config_path(), directories=arguments.directory or [])
    if config is None or arguments.defaults or not config.directories:
        return

    if prompter.confirm("Index the folders now?", default=True):
        with open_index(config, create=True) as session:
            _index_with_progress(config, session, build_sources(config))


def _index_with_progress(
    config: AppConfig, session: IndexSession, sources: list[DocumentSource], reindex: bool = False
) -> None:
    """Embed and store the chunks of the sources and show a progress bar.

    Args:
        config: The application settings.
        session: The open embedding model and its index.
        sources: The sources of the documents.
        reindex: Embed every chunk again, including the stored ones.
    """
    with ProgressReporter(sources) as progress:
        run_indexing(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            model_index=session.model_index,
            embedding_model=session.embedding_model,
            sources=sources,
            reindex=reindex,
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
