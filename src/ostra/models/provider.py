from enum import Enum


class EmbeddingProviderName(Enum):
    """The services that can create the embeddings."""

    OLLAMA = "ollama"
