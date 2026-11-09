from dataclasses import dataclass

from ostra.embeddings.base import EmbeddingProvider
from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.tokenization.base import Tokenizer


@dataclass(frozen=True)
class EmbeddingModel:
    """Bundle an embedding model with the provider that serves it and the tokenizer that matches it."""

    spec: EmbeddingModelSpec
    provider: EmbeddingProvider
    tokenizer: Tokenizer

    def chunk_size_tokens(self, configured: int | None) -> int:
        """Return the chunk size to use, in tokens.

        Args:
            configured: The chunk size from the config file. None means the default of the model.

        Returns:
            The configured chunk size, or the default chunk size of the model.

        Raises:
            ValueError: When the chunk size is too large for the model input.
        """
        chunk_size = self.spec.default_chunk_size_tokens if configured is None else configured
        largest = (
            self.spec.max_tokens - self.spec.special_tokens - self.tokenizer.count_tokens(self.spec.document_prefix)
        )
        if chunk_size > largest:
            raise ValueError(
                f"Chunk size is too large for the model. Model={self.spec.model_id} "
                f"ChunkSize={chunk_size} Largest={largest}"
            )

        return chunk_size
