from unittest.mock import call, create_autospec

from ostra.connectors.base import VectorDatabaseConnector
from ostra.services.database_service import DatabaseService


def test_initialize__connector_given__calls_ensure_database_once():
    """Call ensure_database on the connector exactly once."""
    connector = create_autospec(VectorDatabaseConnector, instance=True)
    service = DatabaseService(connector=connector)

    service.initialize()

    connector.ensure_database.assert_called_once_with()


def test_initialize__connector_given__applies_migrations_after_creating_database():
    """Apply the schema migrations on the connector after the database exists."""
    connector = create_autospec(VectorDatabaseConnector, instance=True)
    service = DatabaseService(connector=connector)

    service.initialize()

    assert connector.mock_calls == [call.ensure_database(), call.migrate()]
