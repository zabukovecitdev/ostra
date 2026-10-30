import argparse
import logging

from personal_rag.config import load_config
from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.embeddings.ollama_embedding_provider import OllamaSettings
from personal_rag.models.mxbai_embedding_model import MxbaiEmbeddingModel
from personal_rag.repositories.file_chunk_repository import FileChunkRepository
from personal_rag.repositories.file_repository import FileRepository
from personal_rag.services.database_service import DatabaseService
from personal_rag.services.file_ingestion_service import FileIngestionService
from personal_rag.services.file_service import FileService
from personal_rag.services.text_chunker import TextChunker


def main(argv: list[str] | None = None):
    """Prepare the local database, then process the configured files.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.
    """
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments(argv)
    config = load_config()
    connector = SqliteConnector(SqliteSettings(path=config.database_path))
    tokenizer = MxbaiEmbeddingModel(settings=OllamaSettings()).get_tokenizer()
    DatabaseService(connector=connector).initialize()
    print(f"Database ready at {config.database_path}")

    ingestion_service = FileIngestionService(
        file_service=FileService(directories=config.directories),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
        chunker=TextChunker(
            chunk_size_tokens=config.chunk_size_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
            count_tokens=tokenizer.count_tokens,
        ),
        reindex=arguments.reindex,
    )
    try:
        ingestion_service.process_files()
    finally:
        connector.close()


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line arguments.

    Args:
        argv: Command line arguments without the program name. Read sys.argv when None.

    Returns:
        The parsed arguments. The reindex attribute is True when the reindex flag is given.
    """
    parser = argparse.ArgumentParser(description="Process the configured text files into the local database.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Process every line again, including lines that a previous run already processed.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
