from unittest.mock import create_autospec

from personal_rag.connectors.base import VectorDatabaseConnector
from personal_rag.repositories.file_embedding_repository import FileEmbeddingRepository


def test_open_connection__connector_injected__returns_connection_from_connector():
    """Return the connection that the injected connector opens."""
    connector = create_autospec(VectorDatabaseConnector, instance=True)
    repository = FileEmbeddingRepository(connector=connector)

    connection = repository.open_connection()

    connector.connect.assert_called_once_with()
    assert connection is connector.connect.return_value
