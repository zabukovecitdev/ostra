import logging

from personal_rag.config import load_config
from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.repositories.file_chunk_repository import FileChunkRepository
from personal_rag.repositories.file_repository import FileRepository
from personal_rag.services.database_service import DatabaseService
from personal_rag.services.file_ingestion_service import FileIngestionService
from personal_rag.services.file_service import FileService


def main():
    """Prepare the local database, then process the configured files."""
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    connector = SqliteConnector(SqliteSettings(path=config.database_path))
    DatabaseService(connector=connector).initialize()
    print(f"Database ready at {config.database_path}")

    ingestion_service = FileIngestionService(
        file_service=FileService(directories=config.directories),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
    )
    try:
        ingestion_service.process_files()
    finally:
        connector.close()


if __name__ == "__main__":
    main()
