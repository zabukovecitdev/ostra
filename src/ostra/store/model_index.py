from abc import ABC, abstractmethod
from collections.abc import Iterable, Set

from ostra.embeddings.vector import Vector
from ostra.store.chunk_match import ChunkMatch
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk


class ModelIndex(ABC):
    """Hold the chunks of all documents for one embedding model.

    Vectors of one model never mix with vectors of another model, because each model has its own index.
    """

    @abstractmethod
    def stored_digests(self, document_id: str) -> set[str]:
        """Return the digests of the chunks that are stored for a document.

        Args:
            document_id: The ID of the document.

        Returns:
            The SHA-256 hex digests of the stored chunks. The set is empty for an unknown document.
        """

    @abstractmethod
    def save_document(
        self, document: DocumentRecord, new_chunks: Iterable[EmbeddedChunk], current_digests: Set[str]
    ) -> None:
        """Save a document and its new chunks, then delete the chunks that the document no longer has.

        Save each chunk when the iterable gives it, so that an interrupted run keeps the chunks it finished.
        A chunk whose digest is already stored is replaced. After the last chunk, delete every stored chunk of
        the document whose digest is not in `current_digests`.

        Args:
            document: The ID, source and title of the document.
            new_chunks: The chunks to save, with their embeddings.
            current_digests: The digests of all chunks that the document has now.

        Raises:
            ValueError: When a chunk vector comes from another model than the index has, or has the wrong size.
        """

    @abstractmethod
    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        """Return the stored chunks that are closest to a query vector.

        Args:
            query: The embedding vector of the query.
            limit: The number of chunks to return.

        Returns:
            The closest chunks, nearest first. Each match has a cosine distance, where 0 means the same direction.

        Raises:
            ValueError: When the vector comes from another model than the index has, or has the wrong size.
        """
