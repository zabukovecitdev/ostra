from collections.abc import Iterator
from pathlib import Path

import pytest

from ostra.connectors.database_facade import DatabaseBackend, DatabaseFacade
from ostra.services.database_service import DatabaseService


@pytest.fixture(params=[DatabaseBackend.SQLITE, DatabaseBackend.CHROMA])
def backend(request: pytest.FixtureRequest) -> DatabaseBackend:
    """Provide each database backend in turn."""
    return request.param


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide the location of the SQLite database file for a test. The Chroma store is beside it."""
    return tmp_path / "ostra.db"


@pytest.fixture
def connector(backend: DatabaseBackend, database_path: Path) -> Iterator[DatabaseFacade]:
    """Provide an initialized facade for each backend and close it after the test."""
    facade = DatabaseFacade(backend=backend, database_path=database_path)
    DatabaseService(connector=facade).initialize()
    yield facade
    facade.close()
