from personal_rag.connectors.base import VectorDatabaseConnector


class DatabaseService:
    """Prepare the database through a vector database connector."""

    def __init__(self, connector: VectorDatabaseConnector):
        self._connector = connector

    def initialize(self) -> None:
        """Create the database when it is missing, through the connector."""
        self._connector.ensure_database()
