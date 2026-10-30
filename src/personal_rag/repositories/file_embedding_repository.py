from typing import Any

from personal_rag.connectors.base import VectorDatabaseConnector


class FileEmbeddingRepository:
    """Store and read file embeddings through a vector database connector."""

    def __init__(self, connector: VectorDatabaseConnector):
        self._connector = connector

    def open_connection(self) -> Any:
        """Open a connection through the injected connector.

        Returns:
            An open connection from the connector.
        """
        return self._connector.connect()
