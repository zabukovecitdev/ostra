from abc import ABC, abstractmethod

from ostra.embeddings.base import EmbeddingProvider
from ostra.tokenization.base import Tokenizer


class EmbeddingModel(ABC):
    """Bundle an embedding model with the tokenizer that matches it, so that one can replace the other."""

    @abstractmethod
    def get_embedding_provider(self) -> EmbeddingProvider:
        """Return a provider that creates embeddings with this model.

        Returns:
            The embedding provider of the model.
        """

    @abstractmethod
    def get_tokenizer(self) -> Tokenizer:
        """Return the tokenizer that matches this model.

        Returns:
            The tokenizer of the model.
        """
