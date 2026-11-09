import logging
from pathlib import Path
from typing import Any, Self

import chromadb
from chromadb.config import Settings

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.store.chroma.chroma_model_index import (
    CHUNKS_COLLECTION_PREFIX,
    PLACEHOLDER_EMBEDDING,
    ChromaModelIndex,
)
from ostra.store.chunk_store import ChunkStore
from ostra.store.model_index import ModelIndex
from ostra.store.model_index_mismatch_error import ModelIndexMismatchError
from ostra.store.model_not_indexed_error import ModelNotIndexedError

logger = logging.getLogger(__name__)

OWNER_ONLY_DIRECTORY_MODE = 0o700
MODEL_INDEXES_COLLECTION_NAME = "model_indexes"


class ChromaChunkStore(ChunkStore):
    """Keep the chunks of each embedding model in a local Chroma store, with one collection for each model.

    Chroma does not support two processes on one store. Use the SQLite store when the background indexer runs.
    """

    def __init__(self, path: Path):
        self._path = path
        self._client: Any | None = None

    @classmethod
    def open(cls, path: Path) -> Self:
        """Create the store directory when it is missing and return the store.

        Args:
            path: Location of the store directory.

        Returns:
            A store that is ready for use.
        """
        store = cls(path)
        store.ensure_database()
        return store

    def ensure_database(self) -> None:
        """Create the store directory when it is missing and set owner-only permissions on it.

        Chroma writes its files inside this directory. The owner-only directory mode blocks other users from
        reading them.
        """
        if not self._path.exists():
            logger.info("Database not found. Creating it. Path=%s", self._path)

        self._path.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
        self._path.chmod(OWNER_ONLY_DIRECTORY_MODE)

    def connect(self) -> Any:
        """Open a client for the local store. Reuse the open client when one exists.

        Returns:
            An open Chroma client.
        """
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self._path), settings=Settings(anonymized_telemetry=False)
            )

        return self._client

    def close(self) -> None:
        """Release the open client. Do nothing when no client is open."""
        if self._client is None:
            return

        self._client.close()
        self._client = None

    def model_index(self, spec: EmbeddingModelSpec, create: bool) -> ModelIndex:
        """Return the index of an embedding model, and create it first when asked.

        Args:
            spec: The embedding model.
            create: Create the index when it does not exist.

        Returns:
            The index of the model.

        Raises:
            ModelNotIndexedError: When the index does not exist and `create` is False.
            ModelIndexMismatchError: When the stored dimensions or fingerprint differ from the spec.
        """
        client = self.connect()
        registry = client.get_or_create_collection(name=MODEL_INDEXES_COLLECTION_NAME, embedding_function=None)
        stored = registry.get(ids=[spec.model_id], include=["metadatas"])
        if not stored["ids"]:
            if not create:
                raise ModelNotIndexedError(spec.model_id, self._indexed_model_ids(registry))

            registry.upsert(
                ids=[spec.model_id],
                metadatas=[{"dimensions": spec.dimensions, "fingerprint": spec.fingerprint}],
                embeddings=[PLACEHOLDER_EMBEDDING],
            )
        else:
            metadata = stored["metadatas"][0]
            if metadata["dimensions"] != spec.dimensions or metadata["fingerprint"] != spec.fingerprint:
                raise ModelIndexMismatchError(spec.model_id, metadata["dimensions"], spec.dimensions)

        return ChromaModelIndex(
            client,
            collection_name=f"{CHUNKS_COLLECTION_PREFIX}{spec.model_id}",
            model_id=spec.model_id,
            dimensions=spec.dimensions,
        )

    def _indexed_model_ids(self, registry: Any) -> list[str]:
        """List the models that have an index with at least one chunk.

        Args:
            registry: The collection that lists the model indexes.

        Returns:
            The model IDs in the order of the registry.
        """
        client = self.connect()
        return [
            model_id
            for model_id in registry.get(include=[])["ids"]
            if client.get_or_create_collection(
                name=f"{CHUNKS_COLLECTION_PREFIX}{model_id}", embedding_function=None
            ).count()
        ]
