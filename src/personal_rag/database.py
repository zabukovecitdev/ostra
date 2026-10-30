import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import sqlite_vec

DATABASE_DIRECTORY = Path.home() / "personal-rag"
DATABASE_PATH = DATABASE_DIRECTORY / "personal_rag.db"
OWNER_ONLY_DIRECTORY_MODE = 0o700
OWNER_ONLY_FILE_MODE = 0o600

logger = logging.getLogger(__name__)


def ensure_database(database_path: Path | None = None) -> Path:
    """Create the SQLite database if it does not exist.

    Create the directory if it does not exist. Set owner-only permissions on the directory and the database file.
    Load the sqlite-vec extension when the database file is created.

    Args:
        database_path: Location of the SQLite database file. Defaults to DATABASE_PATH.

    Returns:
        The path to the database file.
    """
    if database_path is None:
        database_path = DATABASE_PATH

    database_path.parent.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    database_path.parent.chmod(OWNER_ONLY_DIRECTORY_MODE)

    if database_path.exists():
        database_path.chmod(OWNER_ONLY_FILE_MODE)
        return database_path

    logger.info("Database not found. Creating it. Path=%s", database_path)
    os.close(os.open(database_path, os.O_CREAT | os.O_WRONLY, OWNER_ONLY_FILE_MODE))
    with closing(_connect(database_path)) as connection:
        vec_version = connection.execute("SELECT vec_version()").fetchone()[0]
        logger.info("Database created with sqlite-vec. VecVersion=%s", vec_version)

    return database_path


def _connect(database_path: Path) -> sqlite3.Connection:
    """Open a connection to the database and load the sqlite-vec extension.

    Disable extension loading again after the load.

    Args:
        database_path: Location of the SQLite database file.

    Returns:
        An open connection with the sqlite-vec functions available.
    """
    connection = sqlite3.connect(database_path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)

    return connection
