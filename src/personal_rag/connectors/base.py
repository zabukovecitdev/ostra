from abc import ABC, abstractmethod
from typing import Any


class VectorDatabaseConnector(ABC):
    """Define the interface that every vector database connector implements."""

    @abstractmethod
    def ensure_database(self) -> None:
        """Create the database or collection when it is missing."""

    @abstractmethod
    def connect(self) -> Any:
        """Open a connection with the vector support loaded.

        Returns:
            An open connection.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the open connection."""
