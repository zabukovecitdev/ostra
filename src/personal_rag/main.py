import argparse
import logging

from personal_rag.config import load_config
from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.embeddings.ollama_embedding_provider import OllamaSettings
from personal_rag.models.mxbai_embedding_model import MxbaiEmbeddingModel
from personal_rag.repositories.file_chunk_repository import FileChunkRepository
from personal_rag.repositories.file_repository import FileRepository
from personal_rag.services.chunk_search_service import ChunkSearchService
from personal_rag.services.database_service import DatabaseService
from personal_rag.services.file_ingestion_service import FileIngestionService
from personal_rag.services.file_service import FileService
from personal_rag.services.text_chunker import TextChunker

DEFAULT_SEARCH_LIMIT = 5


def main(argv: list[str] | None = None):
    """Prepare the local database, then process the configured files or search the stored chunks.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments(argv)
    config = load_config()
    connector = SqliteConnector(SqliteSettings(path=config.database_path))
    embedding_model = MxbaiEmbeddingModel(settings=OllamaSettings())
    DatabaseService(connector=connector).initialize()
    print(f"Database ready at {config.database_path}")

    try:
        if arguments.search is not None:
            _print_search_results(
                query=arguments.search,
                limit=arguments.limit,
                embedding_provider=embedding_model.get_embedding_provider(),
                file_chunk_repository=FileChunkRepository(connector=connector),
            )
            return

        tokenizer = embedding_model.get_tokenizer()
        ingestion_service = FileIngestionService(
            file_service=FileService(directories=config.directories),
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
    finally:
        connector.close()


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


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line arguments.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.

    Returns:
        The parsed arguments. The search attribute is None when no query is given.
    """
    parser = argparse.ArgumentParser(description="Process the configured text files into the local database.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Process every line again, including lines that a previous run already processed.",
    )
    parser.add_argument("--search", metavar="QUERY", help="Print the stored chunks closest to QUERY and exit.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of chunks to print with --search. Default is {DEFAULT_SEARCH_LIMIT}.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
