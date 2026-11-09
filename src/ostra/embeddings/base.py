from abc import ABC, abstractmethod

from ostra.embeddings.vector import Vector
from ostra.models.embedding_model_spec import EmbeddingModelSpec


class EmbeddingProvider(ABC):
    """Define the interface that every embedding provider implements.

    The base class puts the prefix of the model in front of the text, checks the size of the vector and adds
    the model ID. A provider only sends the text to its service.
    """

    def __init__(self, spec: EmbeddingModelSpec):
        self._spec = spec

    def embed_query(self, text: str) -> Vector:
        """Return the embedding of a search query.

        Args:
            text: The query text.

        Returns:
            The vector of the query, with the query prefix of the model in front of the text.

        Raises:
            ValueError: When the service returns a vector of the wrong size.
        """
        return self._embed_with_prefix(self._spec.query_prefix, text)

    def embed_document(self, text: str) -> Vector:
        """Return the embedding of a chunk of a document.

        Args:
            text: The chunk text.

        Returns:
            The vector of the chunk, with the document prefix of the model in front of the text.

        Raises:
            ValueError: When the service returns a vector of the wrong size.
        """
        return self._embed_with_prefix(self._spec.document_prefix, text)

    def _embed_with_prefix(self, prefix: str, text: str) -> Vector:
        """Embed the text with a prefix, check the size and add the model ID.

        Args:
            prefix: The text to put in front of the text.
            text: The text to embed.

        Returns:
            The vector of the model.
        """
        values = self._embed(prefix + text)
        if len(values) != self._spec.dimensions:
            raise ValueError(
                f"Embedding has the wrong size. Model={self._spec.model_id} "
                f"Expected={self._spec.dimensions} Actual={len(values)}"
            )

        return Vector(model_id=self._spec.model_id, values=tuple(values))

    @abstractmethod
    def _embed(self, text: str) -> list[float]:
        """Send the text to the embedding service.

        Args:
            text: The text to embed, with its prefix.

        Returns:
            The embedding vector as a list of floats.
        """
