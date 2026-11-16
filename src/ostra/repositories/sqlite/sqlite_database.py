import logging
import os
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Self

import sqlite_vec

from ostra.constants import OWNER_ONLY_DIRECTORY_MODE, OWNER_ONLY_FILE_MODE

logger = logging.getLogger(__name__)

# A reader waits this long for a writer lock before it fails.
BUSY_TIMEOUT_SECONDS = 30


class SqliteDatabase:
    """Own the SQLite database file and its connection, with the sqlite-vec extension loaded."""

    def __init__(self, path: Path):
        self._path = path
        self._connection: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        """Return the location of the database file."""
        return self._path

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

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
