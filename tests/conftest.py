from collections.abc import Iterator
from pathlib import Path

import pytest

from personal_rag.connectors.sqlite_connector import SqliteConnector, SqliteSettings
from personal_rag.services.database_service import DatabaseService


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide the location of the SQLite database file for a test."""
    return tmp_path / "personal_rag.db"


@pytest.fixture
def connector(database_path: Path) -> Iterator[SqliteConnector]:
    """Provide an initialized SQLite connector and close it after the test."""
    sqlite_connector = SqliteConnector(SqliteSettings(path=database_path))
    DatabaseService(connector=sqlite_connector).initialize()
    yield sqlite_connector
    sqlite_connector.close()
