from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Define the interface that every embedding provider implements."""

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Return the embedding vector of one text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a list of floats.
        """
