from unittest.mock import create_autospec

from personal_rag.connectors.base import VectorDatabaseConnector
from personal_rag.services.database_service import DatabaseService


def test_initialize__connector_given__calls_ensure_database_once():
    """Call ensure_database on the connector exactly once."""
    connector = create_autospec(VectorDatabaseConnector, instance=True)
    service = DatabaseService(connector=connector)

    service.initialize()

    connector.ensure_database.assert_called_once_with()
