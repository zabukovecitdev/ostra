from pathlib import Path

from ostra.store.chunk_store import ChunkStore
from ostra.store.database_backend import DatabaseBackend
from ostra.store.sqlite.sqlite_chunk_store import SqliteChunkStore

CHROMA_DIRECTORY_NAME = "chroma"


def open_chunk_store(backend: DatabaseBackend, database_path: Path) -> ChunkStore:
    """Open the store of the selected backend. Create it and apply its migrations when it is new or old.

    The Chroma backend loads its package only when it is selected. The Chroma store is a directory named chroma
    beside the SQLite database file.

    Args:
        backend: The backend to open.
        database_path: Location of the SQLite database file. The Chroma store uses its parent directory.

    Returns:
        A store that is ready for use. Close it after use, or use it as a context manager.

    Raises:
        ImportError: When the Chroma backend is selected and the chroma extra is not installed.
        ValueError: When the backend has no store.
    """
    if backend is DatabaseBackend.SQLITE:
        return SqliteChunkStore.open(database_path)

    if backend is DatabaseBackend.CHROMA:
        try:
            from ostra.store.chroma.chroma_chunk_store import ChromaChunkStore
        except ImportError as error:
            raise ImportError(
                "The Chroma backend needs the chroma extra. Install it with: pip install 'ostra[chroma]'"
            ) from error

        return ChromaChunkStore.open(database_path.parent / CHROMA_DIRECTORY_NAME)

    raise ValueError(f"Database backend has no store. Backend={backend}")
