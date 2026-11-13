from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ostra.models.app_config import AppConfig
from ostra.models.embedding_model import EmbeddingModel
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex
from ostra.services.embedding_model_factory import build_embedding_model


@dataclass(frozen=True)
class IndexSession:
    """The embedding model and the index of its chunks, while the database is open."""

    embedding_model: EmbeddingModel
    model_index: SqliteModelIndex


@contextmanager
def open_index(config: AppConfig, model_id: str | None = None, create: bool = False) -> Iterator[IndexSession]:
    """Open the database and the index of an embedding model, and close the database afterwards.

    Args:
        config: The application settings.
        model_id: The model to open. None means the model of the config file.
        create: Create the index when it does not exist. Only the code that writes chunks sets this.

    Yields:
        The embedding model and its index.
    """
    embedding_model = build_embedding_model(config, model_id=model_id)
    with SqliteChunkStore.open(config.database_path) as store:
        yield IndexSession(
            embedding_model=embedding_model,
            model_index=store.model_index(embedding_model.spec, create=create),
        )
