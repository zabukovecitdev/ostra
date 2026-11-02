import argparse
import json
import logging
from pathlib import Path

from ostra.config import NO_DIRECTORIES_MESSAGE, load_config
from ostra.indexing import run_indexing
from ostra.models.factory import build_embedding_model
from ostra.models.registry import EMBEDDING_MODELS
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.store.database_backend import DatabaseBackend
from ostra.store.model_index import ModelIndex
from ostra.store.open_chunk_store import open_chunk_store

DEFAULT_SEARCH_LIMIT = 5
MAX_SHORT_TEXT_CHARACTERS = 300


def main(argv: list[str] | None = None):
    """Run the command given on the command line, or print the help when no command is given.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    logging.basicConfig(level=logging.INFO)
    # The HTTP client logs every request at INFO level. Keep those lines out of the command output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return

    config = load_config()
    backend = arguments.database or config.database_backend
    embedding_model = build_embedding_model(config, model_id=arguments.model)
    spec = embedding_model.spec
    with open_chunk_store(backend, config.database_path) as store:
        if arguments.command == "search":
            _print_search_results(
                query=arguments.query,
                limit=arguments.limit,
                max_distance=arguments.max_distance,
                full_text=arguments.full,
                as_json=arguments.json,
                embedding_provider=embedding_model.provider,
                model_index=store.model_index(spec, create=False),
            )
        else:
            print(f"Database ready. Backend={backend.value}")
            _index_files(
                arguments=arguments,
                config=config,
                model_index=store.model_index(spec, create=True),
                embedding_model=embedding_model,
            )


def _index_files(arguments, config, model_index: ModelIndex, embedding_model) -> None:
    """Embed and store the new or changed chunks of the folders.

    Args:
        arguments: The parsed index arguments.
        config: The application settings.
        model_index: The index that stores the chunks of the embedding model.
        embedding_model: The embedding model and its tokenizer.
    """
    directories = [path.expanduser() for path in arguments.directory] if arguments.directory else None
    if not (directories or config.directories):
        print(NO_DIRECTORIES_MESSAGE)
        return

    run_indexing(
        config=config,
        model_index=model_index,
        embedding_model=embedding_model,
        directories=directories,
        reindex=arguments.reindex,
    )


def _print_search_results(query, limit, max_distance, full_text, as_json, embedding_provider, model_index) -> None:
    """Print the stored chunks that are closest to the query, nearest first.

    Args:
        query: The text to search for.
        limit: The number of chunks to print.
        max_distance: Hide the chunks with a larger cosine distance. None keeps every chunk.
        full_text: Print the whole text of each chunk instead of a shortened text.
        as_json: Print the chunks as a JSON list instead of plain text.
        embedding_provider: Provider that embeds the query.
        model_index: The index that searches the stored chunks.
    """
    search_service = ChunkSearchService(embedding_provider=embedding_provider, model_index=model_index)
    matches = search_service.search(query=query, limit=limit, max_distance=max_distance)

    if as_json:
        print(json.dumps([_match_to_dict(match) for match in matches], ensure_ascii=False, indent=2))
        return

    if not matches:
        print("No matching chunks.")
        return

    for match in matches:
        print(f"match {round(_similarity(match.distance) * 100)}%  file={match.document_id}")
        print(_shorten_text(match.text, full_text))
        print()


def _similarity(distance: float) -> float:
    """Convert a cosine distance to a similarity.

    Args:
        distance: The cosine distance of a chunk. 0 means the same direction and 1 means unrelated.

    Returns:
        The similarity from -1 to 1, where 1 means the same direction.
    """
    return 1 - distance


def _shorten_text(text: str, full_text: bool) -> str:
    """Cut a chunk text to a short preview, unless the full text is requested.

    Args:
        text: The text of the chunk.
        full_text: Return the text unchanged when True.

    Returns:
        The text, or its first characters followed by an ellipsis when it is long.
    """
    if full_text or len(text) <= MAX_SHORT_TEXT_CHARACTERS:
        return text

    return text[:MAX_SHORT_TEXT_CHARACTERS].rstrip() + "..."


def _match_to_dict(match) -> dict:
    """Convert a search match to a dictionary that holds the full text.

    Args:
        match: The chunk match from the search.

    Returns:
        The document ID, source, title, distance, similarity and full text of the chunk.
    """
    return {
        "document_id": match.document_id,
        "source": match.source,
        "title": match.title,
        "file_path": match.document_id,
        "distance": match.distance,
        "similarity": round(_similarity(match.distance), 4),
        "text": match.text,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the index and search commands.

    Returns:
        The parser. Each command has its own options, and the database option works with both commands.
    """
    parser = argparse.ArgumentParser(
        prog="ostra",
        description="Index text files into a local vector database and search them.",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--database",
        type=DatabaseBackend,
        choices=list(DatabaseBackend),
        help="Database backend to use. Overrides database_backend in the config file.",
    )
    shared.add_argument(
        "--model",
        choices=list(EMBEDDING_MODELS),
        help="Embedding model to use. Overrides embedding_model in the config file.",
    )

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    index = commands.add_parser(
        "index",
        parents=[shared],
        help="Embed and store the new or changed chunks of the configured folders.",
        description="Embed and store the new or changed chunks of the configured folders.",
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

    search = commands.add_parser(
        "search",
        parents=[shared],
        help="Print the stored chunks closest to a query.",
        description="Print the stored chunks closest to a query, nearest first.",
    )
    search.add_argument("query", help="The text to search for.")
    search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of chunks to print. Default is {DEFAULT_SEARCH_LIMIT}.",
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


if __name__ == "__main__":
    main()
