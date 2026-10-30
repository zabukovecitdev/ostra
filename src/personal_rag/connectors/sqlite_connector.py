import logging
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from personal_rag.connectors.base import VectorDatabaseConnector

logger = logging.getLogger(__name__)

OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600
SQLITE_MIGRATIONS_DIRECTORY = Path(__file__).parent / "sqlite_migrations"
CREATE_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass(frozen=True)
class SqliteSettings:
    """Hold the settings for a local SQLite database file."""

    path: Path


class SqliteConnector(VectorDatabaseConnector):
    """Connect to a local SQLite database with the sqlite-vec extension loaded."""

    def __init__(self, settings: SqliteSettings):
        self._settings = settings
        self._connection: sqlite3.Connection | None = None

    def ensure_database(self) -> None:
        """Create the database file when it is missing.

        Create the parent directory when it is missing. Set owner-only permissions on the directory and the file.
        Load the sqlite-vec extension when the file is created.
        """
        database_path = self._settings.path
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

        Returns:
            An open connection with the sqlite-vec functions available.
        """
        if self._connection is None:
            connection = sqlite3.connect(self._settings.path)
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

                logger.info("Applying migration. Version=%s", version)
                connection.executescript(migration_path.read_text())
                connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
                connection.commit()
        finally:
            self.close()

    def save_file_digest(self, file_path: str, digest: str) -> None:
        """Insert a file or update its digest and processed time.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the file content.
        """
        connection = self.connect()
        connection.execute(
            "INSERT INTO files (file_path, digest) VALUES (?, ?) "
            "ON CONFLICT (file_path) DO UPDATE SET digest = excluded.digest, processed_at = CURRENT_TIMESTAMP",
            (file_path, digest),
        )
        connection.commit()

    def get_file_digest(self, file_path: str) -> str | None:
        """Return the stored digest of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digest of the file content, or None when the file is not stored.
        """
        connection = self.connect()
        row = connection.execute("SELECT digest FROM files WHERE file_path = ?", (file_path,)).fetchone()
        return row[0] if row else None

    def get_chunk_embedding(self, file_path: str, digest: str) -> list[float] | None:
        """Return the stored embedding of one chunk.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.

        Returns:
            The embedding vector of the chunk, or None when the chunk is not stored.
        """
        connection = self.connect()
        row = connection.execute(
            "SELECT embedding FROM file_chunks WHERE file_path = ? AND digest = ?",
            (file_path, digest),
        ).fetchone()
        if row is None or row[0] is None:
            return None

        return list(struct.unpack(f"{len(row[0]) // 4}f", row[0]))

    def get_chunk_digests(self, file_path: str) -> set[str]:
        """Return the stored chunk digests of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digests of the chunks that were processed.
        """
        connection = self.connect()
        rows = connection.execute("SELECT digest FROM file_chunks WHERE file_path = ?", (file_path,)).fetchall()
        return {row[0] for row in rows}

    def save_chunk_digest(self, file_path: str, digest: str, embedding: list[float]) -> None:
        """Insert a chunk digest and its embedding, or update both when the digest is already stored.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.
            embedding: The embedding vector of the chunk, stored as float32 bytes.
        """
        connection = self.connect()
        connection.execute(
            "INSERT INTO file_chunks (file_path, digest, embedding) VALUES (?, ?, ?) "
            "ON CONFLICT (file_path, digest) DO UPDATE SET "
            "processed_at = CURRENT_TIMESTAMP, embedding = excluded.embedding",
            (file_path, digest, sqlite_vec.serialize_float32(embedding)),
        )
        connection.commit()

    def delete_chunk_digests(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored chunk digests that the file no longer has.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        connection = self.connect()
        connection.executemany(
            "DELETE FROM file_chunks WHERE file_path = ? AND digest = ?",
            [(file_path, digest) for digest in digests],
        )
        connection.commit()
