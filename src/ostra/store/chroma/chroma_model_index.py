from collections.abc import Iterable, Set
from typing import Any

from ostra.embeddings.vector import Vector
from ostra.store.chunk_match import ChunkMatch
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk
from ostra.store.model_index import ModelIndex

CHUNKS_COLLECTION_PREFIX = "chunks."
DOCUMENTS_COLLECTION_NAME = "documents"
# The documents collection stores only metadata. Chroma still needs one vector per record,
# so each document gets this placeholder.
PLACEHOLDER_EMBEDDING = [0.0]


class ChromaModelIndex(ModelIndex):
    """Keep the chunks of one embedding model in one Chroma collection with cosine distance."""

    def __init__(self, client: Any, collection_name: str, model_id: str, dimensions: int):
        self._client = client
        self._model_id = model_id
        self._dimensions = dimensions
        self._chunks = client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._documents = client.get_or_create_collection(name=DOCUMENTS_COLLECTION_NAME, embedding_function=None)

    def stored_digests(self, document_id: str) -> set[str]:
        """Return the digests of the chunks that are stored for a document.

        Args:
            document_id: The ID of the document.

        Returns:
            The SHA-256 hex digests of the stored chunks.
        """
        result = self._chunks.get(where={"document_id": document_id}, include=["metadatas"])
        return {metadata["digest"] for metadata in result["metadatas"]}

    def save_document(
        self, document: DocumentRecord, new_chunks: Iterable[EmbeddedChunk], current_digests: Set[str]
    ) -> None:
        """Save a document and its new chunks, then delete the chunks that the document no longer has.

        Write each chunk as one record when the iterable gives it, so the text, the vector and the digest are
        stored together.

        Args:
            document: The ID, source and title of the document.
            new_chunks: The chunks to save, with their embeddings.
            current_digests: The digests of all chunks that the document has now.
        """
        self._documents.upsert(
            ids=[document.document_id],
            metadatas=[{"source": document.source, "title": document.title}],
            embeddings=[PLACEHOLDER_EMBEDDING],
        )
        for chunk in new_chunks:
            embedding = self._checked(chunk.embedding)
            self._chunks.upsert(
                ids=[_chunk_id(document.document_id, chunk.digest)],
                documents=[chunk.text],
                metadatas=[{"document_id": document.document_id, "digest": chunk.digest}],
                embeddings=[embedding],
            )

        stale = self.stored_digests(document.document_id) - set(current_digests)
        if stale:
            self._chunks.delete(ids=[_chunk_id(document.document_id, digest) for digest in stale])

    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first. Each match has a cosine distance, where 0 means the same direction.
        """
        query_values = self._checked(query)
        count = self._chunks.count()
        if count == 0:
            return []

        result = self._chunks.query(
            query_embeddings=[query_values],
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )
        metadatas = result["metadatas"][0]
        documents = self._documents.get(ids=sorted({m["document_id"] for m in metadatas}), include=["metadatas"])
        details = dict(zip(documents["ids"], documents["metadatas"], strict=True))
        return [
            ChunkMatch(
                document_id=metadata["document_id"],
                source=details.get(metadata["document_id"], {}).get("source", ""),
                title=details.get(metadata["document_id"], {}).get("title", metadata["document_id"]),
                digest=metadata["digest"],
                text=text,
                distance=float(distance),
            )
            for metadata, text, distance in zip(metadatas, result["documents"][0], result["distances"][0], strict=True)
        ]

    def _checked(self, vector: Vector) -> list[float]:
        """Check that a vector belongs to the model of this index and has its size.

        Args:
            vector: The vector to check.

        Returns:
            The values of the vector.

        Raises:
            ValueError: When the vector comes from another model or has the wrong size.
        """
        if vector.model_id != self._model_id or len(vector.values) != self._dimensions:
            raise ValueError(
                f"Vector does not fit the index. Index={self._model_id}/{self._dimensions} "
                f"Vector={vector.model_id}/{len(vector.values)}"
            )

        return list(vector.values)


def _chunk_id(document_id: str, digest: str) -> str:
    """Build the record ID of one chunk from its document ID and digest.

    Args:
        document_id: The ID of the document.
        digest: SHA-256 hex digest of the chunk.

    Returns:
        A record ID that is unique for each document and digest pair.
    """
    return f"{document_id}\n{digest}"
