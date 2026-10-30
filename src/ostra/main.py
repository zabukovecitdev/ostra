import argparse
import logging
from pathlib import Path

from ostra.config import load_config
from ostra.connectors.database_facade import DatabaseBackend, DatabaseFacade
from ostra.embeddings.ollama_embedding_provider import OllamaSettings
from ostra.models.mxbai_embedding_model import MxbaiEmbeddingModel
from ostra.repositories.file_chunk_repository import FileChunkRepository
from ostra.repositories.file_repository import FileRepository
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.database_service import DatabaseService
from ostra.services.file_ingestion_service import FileIngestionService
from ostra.services.file_service import FileService
from ostra.services.text_chunker import TextChunker

DEFAULT_SEARCH_LIMIT = 5


def main(argv: list[str] | None = None):
    """Run the command given on the command line, or print the help when no command is given.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return

    config = load_config()
    backend = arguments.database or config.database_backend
    connector = DatabaseFacade(backend=backend, database_path=config.database_path)
    embedding_model = MxbaiEmbeddingModel(settings=OllamaSettings())
    DatabaseService(connector=connector).initialize()
    print(f"Database ready. Backend={backend.value}")

    try:
        if arguments.command == "search":
            _print_search_results(
                query=arguments.query,
                limit=arguments.limit,
                embedding_provider=embedding_model.get_embedding_provider(),
                file_chunk_repository=FileChunkRepository(connector=connector),
            )
        else:
            _index_files(
                arguments=arguments,
                config=config,
                connector=connector,
                embedding_model=embedding_model,
            )
    finally:
        connector.close()


def _index_files(arguments, config, connector, embedding_model) -> None:
    """Embed and store the new or changed chunks of the folders.

    Args:
        arguments: The parsed index arguments.
        config: The application settings.
        connector: The database connector that stores the chunks.
        embedding_model: The embedding model and its tokenizer.
    """
    tokenizer = embedding_model.get_tokenizer()
    directories = [path.expanduser() for path in arguments.directory] if arguments.directory else config.directories
    ingestion_service = FileIngestionService(
        file_service=FileService(directories=directories),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
        chunker=TextChunker(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            count_tokens=tokenizer.count_tokens,
        ),
        embedding_provider=embedding_model.get_embedding_provider(),
        reindex=arguments.reindex,
    )
    ingestion_service.process_files()


def _print_search_results(query, limit, embedding_provider, file_chunk_repository) -> None:
    """Print the stored chunks that are closest to the query, nearest first.

    Args:
        query: The text to search for.
        limit: The number of chunks to print.
        embedding_provider: Provider that embeds the query.
        file_chunk_repository: Repository that searches the stored chunks.
    """
    search_service = ChunkSearchService(
        embedding_provider=embedding_provider, file_chunk_repository=file_chunk_repository
    )
    for match in search_service.search(query=query, limit=limit):
        print(f"distance={match.distance:.4f} file={match.file_path}")
        print(match.text)
        print()


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

    return parser


if __name__ == "__main__":
    main()
