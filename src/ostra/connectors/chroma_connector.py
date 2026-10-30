import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ostra.connectors.base import ChunkMatch, VectorDatabaseConnector

logger = logging.getLogger(__name__)

OWNER_ONLY_DIRECTORY_MODE = 0o700
FILES_COLLECTION_NAME = "files"
FILE_CHUNKS_COLLECTION_NAME = "file_chunks"
# The files collection stores only metadata. Chroma still needs one vector per record,
# so each file gets this placeholder.
PLACEHOLDER_EMBEDDING = [0.0]


@dataclass(frozen=True)
class ChromaSettings:
    """Hold the settings for a local Chroma store."""

    path: Path


class ChromaConnector(VectorDatabaseConnector):
    """Connect to a local Chroma store that keeps the chunk vectors, text and digests."""

    def __init__(self, settings: ChromaSettings):
        self._settings = settings
        self._client: Any | None = None

    def ensure_database(self) -> None:
        """Create the store directory when it is missing and set owner-only permissions on it.

        Chroma writes its files inside this directory. The owner-only directory mode blocks other users from
        reading them.
        """
        store_path = self._settings.path
        if not store_path.exists():
            logger.info("Database not found. Creating it. Path=%s", store_path)

        store_path.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
        store_path.chmod(OWNER_ONLY_DIRECTORY_MODE)

    def connect(self) -> Any:
        """Open a client for the local store.

        Reuse the open client when one exists.

        Returns:
            An open Chroma client.
        """
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self._settings.path),
                settings=Settings(anonymized_telemetry=False),
            )

        return self._client

    def close(self) -> None:
        """Release the open client. Do nothing when no client is open."""
        if self._client is None:
            return

        self._client.close()
        self._client = None

    def migrate(self) -> None:
        """Create the collections when they are missing.

        Chroma has no schema migrations, so this method only makes the collections exist.
        """
        self._get_collection(FILES_COLLECTION_NAME)
        self._get_collection(FILE_CHUNKS_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    def save_file_digest(self, file_path: str, digest: str) -> None:
        """Insert a file or update its digest and processed time.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the file content.
        """
        self._get_collection(FILES_COLLECTION_NAME).upsert(
            ids=[file_path],
            metadatas=[{"digest": digest, "processed_at": _now()}],
            embeddings=[PLACEHOLDER_EMBEDDING],
        )

    def get_file_digest(self, file_path: str) -> str | None:
        """Return the stored digest of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digest of the file content, or None when the file is not stored.
        """
        result = self._get_collection(FILES_COLLECTION_NAME).get(ids=[file_path], include=["metadatas"])
        if not result["metadatas"]:
            return None

        return result["metadatas"][0]["digest"]

    def get_chunk_digests(self, file_path: str) -> set[str]:
        """Return the stored chunk digests of one file.

        Args:
            file_path: Location of the file.

        Returns:
            The SHA-256 hex digests of the chunks that were processed.
        """
        result = self._get_collection(FILE_CHUNKS_COLLECTION_NAME).get(
            where={"file_path": file_path},
            include=["metadatas"],
        )
        return {metadata["digest"] for metadata in result["metadatas"]}

    def save_chunk_digest(self, file_path: str, digest: str, text: str, embedding: list[float]) -> None:
        """Insert a chunk and its embedding, or update both when the digest is already stored.

        Chroma writes each chunk as one record, so the text, the vector and the digest are stored together.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.
            text: The text of the chunk.
            embedding: The embedding vector of the chunk.
        """
        self._get_collection(FILE_CHUNKS_COLLECTION_NAME).upsert(
            ids=[_chunk_id(file_path, digest)],
            documents=[text],
            metadatas=[{"file_path": file_path, "digest": digest, "processed_at": _now()}],
            embeddings=[embedding],
        )

    def get_chunk_embedding(self, file_path: str, digest: str) -> list[float] | None:
        """Return the stored embedding of one chunk.

        Args:
            file_path: Location of the file.
            digest: SHA-256 hex digest of the chunk.

        Returns:
            The embedding vector of the chunk, or None when the chunk is not stored.
        """
        result = self._get_collection(FILE_CHUNKS_COLLECTION_NAME).get(
            ids=[_chunk_id(file_path, digest)],
            include=["embeddings"],
        )
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return None

        return [float(value) for value in result["embeddings"][0]]

    def search_chunks(self, query_embedding: list[float], limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query_embedding: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first. Each match has a cosine distance, where 0 means the same direction.
        """
        collection = self._get_collection(FILE_CHUNKS_COLLECTION_NAME)
        if collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            ChunkMatch(
                file_path=metadata["file_path"],
                digest=metadata["digest"],
                text=text,
                distance=float(distance),
            )
            for metadata, text, distance in zip(
                result["metadatas"][0], result["documents"][0], result["distances"][0], strict=True
            )
        ]

    def delete_chunk_digests(self, file_path: str, digests: set[str]) -> None:
        """Delete the stored chunks that the file no longer has.

        Chroma rejects an empty list of ids, so this method returns without a call when no digest is given.

        Args:
            file_path: Location of the file.
            digests: SHA-256 hex digests to delete.
        """
        if not digests:
            return

        self._get_collection(FILE_CHUNKS_COLLECTION_NAME).delete(
            ids=[_chunk_id(file_path, digest) for digest in digests],
        )

    def _get_collection(self, name: str, metadata: dict | None = None) -> Any:
        """Return a collection and create it when it is missing.

        Args:
            name: Name of the collection.
            metadata: Settings for a new collection, such as the distance metric.

        Returns:
            The Chroma collection without an embedding function.
        """
        return self.connect().get_or_create_collection(name=name, embedding_function=None, metadata=metadata)


def _chunk_id(file_path: str, digest: str) -> str:
    """Build the record id of one chunk from its file path and digest.

    Args:
        file_path: Location of the file.
        digest: SHA-256 hex digest of the chunk.

    Returns:
        A record id that is unique for each file and digest pair.
    """
    return f"{file_path}\n{digest}"


def _now() -> str:
    """Return the current UTC time in ISO 8601 format.

    Returns:
        The current time as text.
    """
    return datetime.now(UTC).isoformat()
