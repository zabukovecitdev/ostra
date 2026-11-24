import sqlite3
from pathlib import Path

from ostra.repositories.sqlite.sqlite_model_index_registry import table_name


def indexed_document_counts(database_path: Path) -> dict[str, int]:
    """Count the documents that each model index holds, without changing the database.

    The file is opened read-only. A missing file is not created, and no migration runs.

    Args:
        database_path: Location of the database file.

    Returns:
        The number of distinct documents with chunks, by model ID. Empty when the file is missing, or when it
        is not a database with the model indexes.
    """
    if not database_path.exists():
        return {}

    try:
        connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    except sqlite3.Error:
        return {}

    try:
        return _count_documents(connection)
    except sqlite3.Error, ValueError:
        return {}
    finally:
        connection.close()


def _read_only_uri(database_path: Path) -> str:
    """Return the SQLite URI of the file opened read-only. as_uri quotes the characters of the path, on any system."""
    return f"{database_path.resolve().as_uri()}?mode=ro"


def _count_documents(connection: sqlite3.Connection) -> dict[str, int]:
    """Count the documents of each model index."""
    rows = connection.execute("SELECT model_id, chunks_table FROM model_indexes").fetchall()
    return {model_id: _count_documents_in(connection, chunks_table) for model_id, chunks_table in rows}


def _count_documents_in(connection: sqlite3.Connection, chunks_table: str) -> int:
    """Count the distinct documents that have chunks in one chunk table."""
    (count,) = connection.execute(f"SELECT COUNT(DISTINCT document_id) FROM {table_name(chunks_table)}").fetchone()
    return int(count)
