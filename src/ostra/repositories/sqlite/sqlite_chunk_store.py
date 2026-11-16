import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.repositories.sqlite.sqlite_database import SqliteDatabase
from ostra.repositories.sqlite.sqlite_migrator import SqliteMigrator
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex
from ostra.repositories.sqlite.sqlite_model_index_registry import SqliteModelIndexRegistry


class SqliteChunkStore:
    """Keep the chunks of each embedding model in a local SQLite database with the sqlite-vec extension.

    The store puts the database file, the migrations and the model registry together. Each one is its own class.
    """

    def __init__(self, database: SqliteDatabase):
        self._database = database
        self._registry = SqliteModelIndexRegistry(database)

    @classmethod
    def open(cls, path: Path) -> Self:
        """Create the database when it is missing, apply the migrations and return the store.

        Args:
            path: Location of the database file.

        Returns:
            A store that is ready for use.
        """
        database = SqliteDatabase(path)
        database.ensure_database()
        SqliteMigrator(database).migrate()
        return cls(database)

    def connect(self) -> sqlite3.Connection:
        """Return the open connection of the database."""
        return self._database.connect()

    def close(self) -> None:
        """Close the connection of the database."""
        self._database.close()

    def model_index(self, spec: EmbeddingModelSpec, create: bool) -> SqliteModelIndex:
        """Return the index of an embedding model, and create it first when asked.

        Args:
            spec: The embedding model.
            create: Create the index when it does not exist.

        Returns:
            The index of the model.

        Raises:
            ModelNotIndexedError: When the index does not exist and `create` is False.
            ModelIndexMismatchError: When the stored dimensions or fingerprint differ from the spec.
        """
        return self._registry.model_index(spec, create)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
