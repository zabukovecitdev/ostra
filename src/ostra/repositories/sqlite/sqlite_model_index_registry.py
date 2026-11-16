import re
import sqlite3

from ostra.errors.model_index_mismatch_error import ModelIndexMismatchError
from ostra.errors.model_not_indexed_error import ModelNotIndexedError
from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.repositories.sqlite.sqlite_database import SqliteDatabase
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex


class SqliteModelIndexRegistry:
    """Find, create and open the index of each embedding model in the database."""

    def __init__(self, database: SqliteDatabase):
        self._database = database

    def model_index(self, spec: EmbeddingModelSpec, create: bool) -> SqliteModelIndex:
        """Return the index of an embedding model, and create it first when asked.

        A new index gets its own chunk table and its own vector table, sized for the model. Only a call with
        `create` set can change the schema. The index of the first model has no fingerprint until a call with
        `create` set stores it.

        Args:
            spec: The embedding model.
            create: Create the index when it does not exist.

        Returns:
            The index of the model.

        Raises:
            ModelNotIndexedError: When the index does not exist and `create` is False.
            ModelIndexMismatchError: When the stored dimensions or fingerprint differ from the spec.
        """
        connection = self._database.connect()
        row = self._find_index(connection, spec.model_id)
        if row is None:
            if not create:
                raise ModelNotIndexedError(spec.model_id, self._indexed_model_ids(connection))

            row = self._create_index(connection, spec)

        _, stored_dimensions, stored_fingerprint, chunks_table, vectors_table = row
        if stored_dimensions != spec.dimensions or stored_fingerprint not in (None, spec.fingerprint):
            raise ModelIndexMismatchError(spec.model_id, stored_dimensions, spec.dimensions)

        if stored_fingerprint is None and create:
            connection.execute(
                "UPDATE model_indexes SET fingerprint = ? WHERE model_id = ?", (spec.fingerprint, spec.model_id)
            )
            connection.commit()

        return SqliteModelIndex(
            connection,
            model_id=spec.model_id,
            dimensions=spec.dimensions,
            chunks_table=_table_name(chunks_table),
            vectors_table=_table_name(vectors_table),
        )

    def _find_index(self, connection: sqlite3.Connection, model_id: str) -> tuple | None:
        """Read the registry row of a model.

        Args:
            connection: An open connection.
            model_id: The ID of the model.

        Returns:
            The model index ID, the dimensions, the fingerprint and the two table names. None when the model has
            no index.
        """
        return connection.execute(
            "SELECT model_index_id, dimensions, fingerprint, chunks_table, vectors_table "
            "FROM model_indexes WHERE model_id = ?",
            (model_id,),
        ).fetchone()

    def _indexed_model_ids(self, connection: sqlite3.Connection) -> list[str]:
        """List the models that have an index with at least one chunk.

        Args:
            connection: An open connection.

        Returns:
            The model IDs in the order of their creation.
        """
        model_ids = []
        for model_id, chunks_table in connection.execute(
            "SELECT model_id, chunks_table FROM model_indexes ORDER BY model_index_id"
        ).fetchall():
            if connection.execute(f"SELECT 1 FROM {_table_name(chunks_table)} LIMIT 1").fetchone():
                model_ids.append(model_id)

        return model_ids

    def _create_index(self, connection: sqlite3.Connection, spec: EmbeddingModelSpec) -> tuple:
        """Create the chunk table and the vector table of a model in one write transaction.

        The registry is read again after the write lock is held, so two processes cannot create the same index.

        Args:
            connection: An open connection.
            spec: The embedding model.

        Returns:
            The registry row of the index.
        """
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._find_index(connection, spec.model_id)
            if row is not None:
                connection.rollback()
                return row

            cursor = connection.execute(
                "INSERT INTO model_indexes (model_id, dimensions, fingerprint, chunks_table, vectors_table) "
                "VALUES (?, ?, ?, '', '')",
                (spec.model_id, spec.dimensions, spec.fingerprint),
            )
            chunks_table = f"chunks_{cursor.lastrowid}"
            vectors_table = f"vectors_{cursor.lastrowid}"
            connection.execute(
                "UPDATE model_indexes SET chunks_table = ?, vectors_table = ? WHERE model_id = ?",
                (chunks_table, vectors_table, spec.model_id),
            )
            connection.execute(
                f"CREATE TABLE {chunks_table} ("
                "chunk_id INTEGER PRIMARY KEY, document_id TEXT NOT NULL, digest TEXT NOT NULL, chunk_text TEXT, "
                "processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE (document_id, digest))"
            )
            connection.execute(
                f"CREATE VIRTUAL TABLE {vectors_table} USING "
                f"vec0(embedding float[{int(spec.dimensions)}] distance_metric=cosine)"
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

        return self._find_index(connection, spec.model_id)  # type: ignore[return-value]


def _table_name(name: str) -> str:
    """Check that a table name from the registry is a plain identifier, so that it is safe in SQL text.

    Args:
        name: The table name read from the database.

    Returns:
        The same name.

    Raises:
        ValueError: When the name is not a plain identifier.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Table name is not a plain identifier. Name={name}")

    return name
