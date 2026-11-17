from abc import ABC, abstractmethod
from collections.abc import Iterable, Set

from ostra.models.chunk_match import ChunkMatch
from ostra.models.document_record import DocumentRecord
from ostra.models.embedded_chunk import EmbeddedChunk
from ostra.models.vector import Vector


class ChunkSearchIndex(ABC):
    """Define what the search needs from the index of an embedding model."""

    @abstractmethod
    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first. Each match has a cosine distance, where 0 means the same direction.
        """


class ChunkWriteIndex(ABC):
    """Define what the ingestion needs from the index of an embedding model."""

    @abstractmethod
    def stored_digests(self, document_id: str) -> set[str]:
        """Return the digests of the chunks that are stored for a document.

        Args:
            document_id: The ID of the document.

        Returns:
            The SHA-256 hex digests of the stored chunks.
        """

    @abstractmethod
    def save_document(
        self, document: DocumentRecord, new_chunks: Iterable[EmbeddedChunk], current_digests: Set[str]
    ) -> None:
        """Save a document and its new chunks, then delete the chunks that the document no longer has.

        Args:
            document: The ID, source and title of the document.
            new_chunks: The chunks to save, with their embeddings.
            current_digests: The digests of all chunks that the document has now.
        """


class ModelIndex(ChunkSearchIndex, ChunkWriteIndex):
    """Define the full index of one embedding model, which both searches and stores chunks."""
