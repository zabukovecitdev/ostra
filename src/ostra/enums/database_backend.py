from enum import StrEnum


class DatabaseBackend(StrEnum):
    """Name the database backends that the application can use."""

    SQLITE = "sqlite"
    CHROMA = "chroma"
