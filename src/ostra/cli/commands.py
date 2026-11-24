import argparse
import sqlite3
import sys
from pathlib import Path

import httpx
from ollama import ResponseError

from ostra.cli.config.prompter import DefaultsPrompter, Prompter
from ostra.cli.config.questionary_prompter import QuestionaryPrompter
from ostra.cli.config.wizard import SetupResult, run_setup
from ostra.cli.progress import ProgressReporter
from ostra.cli.search_output import SearchOutputOptions, print_search_results
from ostra.cli.search_picker import pick_and_open
from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.config_repository import default_config_path, load_config, save_config, with_environment
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.index_session import IndexSession, open_index
from ostra.services.ingestion.indexing_service import run_indexing
from ostra.services.ingestion.source_factory import build_sources

# The errors that stop an indexing run with a message instead of a traceback.
_INDEX_ERRORS = (ConnectionError, httpx.HTTPError, ResponseError, ValueError, sqlite3.Error)


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


def run_config(arguments: argparse.Namespace) -> None:
    """Ask the setup questions, save the settings and index the folders when the user asks for it.

    Args:
        arguments: The parsed config arguments.

    Raises:
        SystemExit: When the questions cannot be asked, because the input is not a terminal; with status 2 when a
            --directory path is not a folder; with status 1 when --defaults cannot save the settings.
    """
    if not (arguments.defaults or sys.stdin.isatty()):
        raise SystemExit("ostra config needs a terminal. Use ostra config --defaults, or edit the config file.")

    given = arguments.directory or []
    _require_folders(given)
    prompter: Prompter = DefaultsPrompter() if arguments.defaults else QuestionaryPrompter()
    config_path = default_config_path()
    result = run_setup(prompter, config_path, directories=given, shortcut=arguments.setting)
    if result is None:
        if arguments.defaults:
            raise SystemExit(1)
        return

    _apply_result(result, config_path)


def _require_folders(folders: list[Path]) -> None:
    """Stop before any question when a --directory path is not an existing folder.

    Raises:
        SystemExit: With status 2, as for any other invalid option, naming the paths.
    """
    missing = [str(folder) for folder in folders if not folder.expanduser().is_dir()]
    if missing:
        print(f"ostra config: not an existing folder: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)


def _apply_result(result: SetupResult, config_path: Path) -> None:
    """Index now when the user chose to, or remind the user to run ostra index when a change needs it."""
    if result.index_now:
        _index_after_save(result, config_path)
    elif result.index_changes and result.config.directories:
        _print_index_reminder(result)


def _index_after_save(result: SetupResult, config_path: Path) -> None:
    """Index with the chosen settings. The settings of a model change are saved only after indexing."""
    if result.model_changed:
        _index_then_save(result, config_path)
    else:
        _index_saved_settings(config_path)


def _print_index_reminder(result: SetupResult) -> None:
    """Tell the user to run ostra index, and why searches may find nothing until then."""
    print(f"Run ostra index to apply the changes ({', '.join(result.index_changes)}).")
    if result.model_changed:
        print("Searches with the new model find nothing until then.")


def _index_then_save(result: SetupResult, config_path: Path) -> None:
    """Index with the new model, and save the settings only when indexing finished.

    A failure or Ctrl-C keeps the old settings in the file, so that the old model still matches the index.

    Args:
        result: The answers of the setup, with the new settings.
        config_path: The config file to write.
    """
    failure = _index_new_settings(result.config)
    if failure is None:
        save_config(result.config, config_path)
        print(f"Settings saved to {config_path}.")
        return

    _print_not_saved(result, failure)


def _print_not_saved(result: SetupResult, failure: BaseException) -> None:
    """Say why indexing stopped, that the settings were not saved, and what the user can do next."""
    keep = "Chunks embedded so far are kept; run ostra config again to save the settings and continue."
    # Without a config file, ostra used no settings before, so the message names no previous model.
    still_used = f", so ostra still uses {result.previous_model}" if result.had_config_file else ""
    print(_stopped_text(failure))
    print(f"The settings were not saved{still_used}. {keep}")


def _index_new_settings(config: AppConfig) -> BaseException | None:
    """Index with the settings that are not saved yet, with the OSTRA_ environment variables applied.

    Returns:
        The error or Ctrl-C that stopped the indexing, or None when indexing finished.
    """
    try:
        settings = with_environment(config)
    except ValueError as error:
        return error

    return _index_settings(settings)


def _index_saved_settings(config_path: Path) -> None:
    """Index the configured folders with the saved settings, and report a failure without a traceback.

    The settings are read with load_config, so that the OSTRA_ environment variables apply as they do for
    ostra index and ostra search.

    Args:
        config_path: The saved config file.
    """
    try:
        config = load_config(config_path)
    except ValueError as error:
        print(_failure_message(error))
        return

    failure = _index_settings(config)
    if isinstance(failure, KeyboardInterrupt):
        print("Indexing stopped. The settings are saved; run ostra index to continue.")
    elif failure is not None:
        print(_failure_message(failure))


def _index_settings(config: AppConfig) -> BaseException | None:
    """Index the configured folders with these settings. Both index paths use it.

    Returns:
        The error or Ctrl-C that stopped the indexing, or None when indexing finished.
    """
    try:
        with open_index(config, create=True) as session:
            _index_with_progress(config, session, build_sources(config))
    except (KeyboardInterrupt, *_INDEX_ERRORS) as error:
        return error

    return None


def _stopped_text(failure: BaseException) -> str:
    """Name the error that stopped the indexing, or say that the user stopped it."""
    if isinstance(failure, KeyboardInterrupt):
        return "Indexing stopped."

    return f"Indexing stopped: {failure}"


def _failure_message(error: BaseException) -> str:
    """Say why indexing stopped and what the user can do about it."""
    if isinstance(error, (ConnectionError, httpx.HTTPError, ResponseError)):
        return f"Indexing stopped: {error}\nCheck the Ollama server, then run ostra index."

    return f"Indexing stopped: {error}\nFix the problem, then run ostra index."


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
    if _should_pick(arguments):
        files = service.search_files(query=arguments.query, limit=arguments.limit, max_distance=arguments.max_distance)
        pick_and_open(files, QuestionaryPrompter())
        return

    matches = service.search(query=arguments.query, limit=arguments.limit, max_distance=arguments.max_distance)
    print_search_results(matches, SearchOutputOptions(full_text=arguments.full, as_json=arguments.json))


def _should_pick(arguments: argparse.Namespace) -> bool:
    """Decide whether the search lists the files to open, or prints the chunks.

    --json always prints. --interactive and --no-interactive decide when given. Otherwise the list is shown when a
    person can answer it in a terminal and --full does not ask for the chunk texts.
    """
    if arguments.json:
        return False

    if arguments.interactive is not None:
        return bool(arguments.interactive)

    return not arguments.full and _is_terminal()


def _is_terminal() -> bool:
    """Check that a person can answer questions: both the input and the output are a terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()
