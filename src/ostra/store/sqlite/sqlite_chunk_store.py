import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Self

import sqlite_vec

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.store.chunk_store import ChunkStore
from ostra.store.model_index import ModelIndex
from ostra.store.model_index_mismatch_error import ModelIndexMismatchError
from ostra.store.model_not_indexed_error import ModelNotIndexedError
from ostra.store.sqlite.sqlite_model_index import SqliteModelIndex

logger = logging.getLogger(__name__)

OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600
# A reader waits this long for a writer lock before it fails.
BUSY_TIMEOUT_SECONDS = 30
SQLITE_MIGRATIONS_DIRECTORY = Path(__file__).parent / "migrations"
# This migration replaces tables. A copy of the database is saved before it runs on a database with data.
BACKUP_BEFORE_MIGRATION = "006_create_model_indexes"
CREATE_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class SqliteChunkStore(ChunkStore):
    """Keep the chunks of each embedding model in a local SQLite database with the sqlite-vec extension."""

    def __init__(self, path: Path):
        self._path = path
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def open(cls, path: Path) -> Self:
        """Create the database when it is missing, apply the migrations and return the store.

        Args:
            path: Location of the database file.

        Returns:
            A store that is ready for use.
        """
        store = cls(path)
        store.ensure_database()
        store.migrate()
        return store

    def ensure_database(self) -> None:
        """Create the database file when it is missing.

        Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.
        Load the sqlite-vec extension when the file is created.
        """
        database_path = self._path
        database_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
        database_path.parent.chmod(OWNER_ONLY_DIRECTORY_MODE)

        if database_path.exists():
            database_path.chmod(OWNER_ONLY_FILE_MODE)
            return

        logger.info("Database not found. Creating it. Path=%s", database_path)
        os.close(os.open(database_path, os.O_CREAT | os.O_WRONLY, OWNER_ONLY_FILE_MODE))
        try:
            vec_version = self.connect().execute("SELECT vec_version()").fetchone()[0]
        finally:
            self.close()

        logger.info("Database created with sqlite-vec. VecVersion=%s", vec_version)

    def connect(self) -> sqlite3.Connection:
        """Open a connection with the sqlite-vec extension loaded.

        Reuse the open connection when one exists. Disable extension loading again after the load.
        Turn on write-ahead logging, so one process can read while another process writes.

        Returns:
            An open connection with the sqlite-vec functions available.
        """
        if self._connection is None:
            connection = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            self._connection = connection

        return self._connection

    def close(self) -> None:
        """Close the open connection. Do nothing when no connection is open."""
        if self._connection is None:
            return

        self._connection.close()
        self._connection = None

    def migrate(self) -> None:
        """Apply each SQLite migration that has not run yet.

        Record each applied migration in the schema_migrations table. Run each migration only once.
        Apply the migrations in file name order.
        """
        connection = self.connect()
        try:
            connection.execute(CREATE_SCHEMA_MIGRATIONS_TABLE_SQL)
            applied_versions = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for migration_path in sorted(SQLITE_MIGRATIONS_DIRECTORY.glob("*.sql")):
                version = migration_path.stem
                if version in applied_versions:
                    continue

                if version == BACKUP_BEFORE_MIGRATION and applied_versions:
                    self._back_up(connection)

                logger.info("Applying migration. Version=%s", version)
                connection.executescript(migration_path.read_text())
                connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
                connection.commit()
        finally:
            self.close()

    def model_index(self, spec: EmbeddingModelSpec, create: bool) -> ModelIndex:
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
        connection = self.connect()
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

    def _back_up(self, connection: sqlite3.Connection) -> None:
        """Save a copy of the database next to it before a migration that replaces tables.

        Keep the first copy. Do nothing when a copy exists. Set owner-only permissions on the copy.

        Args:
            connection: An open connection to the database.
        """
        backup_path = self._path.with_name(f"{self._path.name}.before-{BACKUP_BEFORE_MIGRATION}")
        if backup_path.exists():
            return

        logger.info("Saving a copy of the database before the migration. Path=%s", backup_path)
        os.close(os.open(backup_path, os.O_CREAT | os.O_WRONLY, OWNER_ONLY_FILE_MODE))
        backup = sqlite3.connect(backup_path)
        try:
            connection.backup(backup)
        finally:
            backup.close()


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
