import logging
import os
import sqlite3
from pathlib import Path

from ostra.constants import OWNER_ONLY_FILE_MODE
from ostra.repositories.sqlite.sqlite_database import SqliteDatabase

logger = logging.getLogger(__name__)

SQLITE_MIGRATIONS_DIRECTORY = Path(__file__).parent / "migrations"
# This migration replaces tables. A copy of the database is saved before it runs on a database with data.
BACKUP_BEFORE_MIGRATION = "006_create_model_indexes"
CREATE_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class SqliteMigrator:
    """Apply the SQL migrations of the database schema, each one once."""

    def __init__(self, database: SqliteDatabase):
        self._database = database

    def migrate(self) -> None:
        """Apply each SQLite migration that has not run yet.

        Record each applied migration in the schema_migrations table. Run each migration only once.
        Apply the migrations in file name order.
        """
        connection = self._database.connect()
        try:
            connection.execute(CREATE_SCHEMA_MIGRATIONS_TABLE_SQL)
            applied_versions = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for migration_path in sorted(SQLITE_MIGRATIONS_DIRECTORY.glob("*.sql")):
                self._apply(connection, migration_path, applied_versions)
        finally:
            self._database.close()

    def _apply(self, connection: sqlite3.Connection, migration_path: Path, applied_versions: set[str]) -> None:
        """Apply one migration, unless it already ran, and record it.

        Args:
            connection: An open connection.
            migration_path: The SQL file of the migration.
            applied_versions: The versions that already ran.
        """
        version = migration_path.stem
        if version in applied_versions:
            return

        if version == BACKUP_BEFORE_MIGRATION and applied_versions:
            self._back_up(connection)

        logger.info("Applying migration. Version=%s", version)
        connection.executescript(migration_path.read_text())
        connection.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
        connection.commit()

    def _back_up(self, connection: sqlite3.Connection) -> None:
        """Save a copy of the database next to it before a migration that replaces tables.

        Keep the first copy. Do nothing when a copy exists. Set owner-only permissions on the copy.

        Args:
            connection: An open connection to the database.
        """
        path = self._database.path
        backup_path = path.with_name(f"{path.name}.before-{BACKUP_BEFORE_MIGRATION}")
        if backup_path.exists():
            return

        logger.info("Saving a copy of the database before the migration. Path=%s", backup_path)
        os.close(os.open(backup_path, os.O_CREAT | os.O_WRONLY, OWNER_ONLY_FILE_MODE))
        backup = sqlite3.connect(backup_path)
        try:
            connection.backup(backup)
        finally:
            backup.close()
