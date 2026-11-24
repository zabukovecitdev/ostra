import argparse
from pathlib import Path

from ostra.cli.commands import run_config, run_index, run_search
from ostra.cli.config.menu import SETTING_SHORTCUTS
from ostra.models.embedding_model_registry import EMBEDDING_MODELS

DEFAULT_SEARCH_LIMIT = 5


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the config, index and search commands.

    Returns:
        The parser. Each command has its own options, and the database option works with both commands.
    """
    parser = argparse.ArgumentParser(
        prog="ostra",
        description="Index text files into a local vector database and search them.",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--model",
        choices=list(EMBEDDING_MODELS),
        help="Embedding model to use. Overrides embedding_model in the config file.",
    )

    shared.add_argument(
        "--database",
        type=Path,
        metavar="PATH",
        help="SQLite database file to use. Overrides database_path in the config file.",
    )

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    config = commands.add_parser(
        "config",
        aliases=["init"],
        help="Set up ostra, or change its settings.",
        description="Walk through the setup the first time, and change single settings later.",
    )
    config.set_defaults(handler=run_config, needs_index=False)
    config.add_argument(
        "setting",
        nargs="?",
        choices=list(SETTING_SHORTCUTS),
        help="Change only this setting: model, folders, chunks or database. Ignored when there is no config file "
        "with folders yet, because the setup then asks every question.",
    )
    config.add_argument(
        "--defaults",
        action="store_true",
        help="Ask no questions. Keep the current settings, or save the defaults when there is no config file.",
    )
    config.add_argument(
        "--directory",
        action="append",
        type=Path,
        metavar="PATH",
        help="Folder to add to the configured folders. Repeat the option for more folders.",
    )

    index = commands.add_parser(
        "index",
        parents=[shared],
        help="Embed and store the new or changed chunks of the configured folders.",
        description="Embed and store the new or changed chunks of the configured folders.",
    )
    index.set_defaults(handler=run_index, needs_index=True, create_index=True)
    index.add_argument(
        "folders",
        nargs="*",
        type=Path,
        metavar="PATH",
        help="Folders to index instead of the configured folders.",
    )
    index.add_argument(
        "--directory",
        action="append",
        type=Path,
        metavar="PATH",
        help="Folder to index instead of the configured folders. Repeat the option for more folders.",
    )
    index.add_argument(
        "--reindex",
        action="store_true",
        help="Embed every chunk again, including chunks that a previous run already stored.",
    )
    index.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        help="Index the files in the subfolders too, or not. Skip hidden subfolders, such as .git. "
        "Overrides recursive in the config file.",
    )

    search = commands.add_parser(
        "search",
        parents=[shared],
        help="Print the stored chunks closest to a query.",
        description="Print the stored chunks closest to a query, nearest first.",
    )
    search.set_defaults(handler=run_search, needs_index=True, create_index=False)
    search.add_argument("query", help="The text to search for.")
    search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of chunks to print, or of files to list in the interactive mode. Default is "
        f"{DEFAULT_SEARCH_LIMIT}.",
    )
    search.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        help="List the matching files and open the one you pick, or print the chunks. By default the list is shown "
        "when the command runs in a terminal without --json or --full.",
    )
    search.add_argument(
        "--max-distance",
        type=float,
        metavar="DISTANCE",
        help="Hide chunks with a larger cosine distance. Distance runs from 0 (same direction) to 2 (opposite).",
    )
    search.add_argument(
        "--full",
        action="store_true",
        help="Print the whole text of each chunk. By default the text is shortened.",
    )
    search.add_argument(
        "--json",
        action="store_true",
        help="Print the chunks as a JSON list, with the full text.",
    )

    return parser
