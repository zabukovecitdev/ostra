import logging
from pathlib import Path

from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.services.database_service import DatabaseService

DATABASE_PATH = Path.home() / "personal-rag" / "personal_rag.db"


def main():
    """Prepare the local database and print its path."""
    logging.basicConfig(level=logging.INFO)
    connector = SqliteConnector(SqliteSettings(path=DATABASE_PATH))
    service = DatabaseService(connector=connector)
    service.initialize()
    print(f"Database ready at {DATABASE_PATH}")


if __name__ == "__main__":
    main()
