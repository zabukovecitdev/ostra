import logging
from pathlib import Path

from fastembed import TextEmbedding

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingProvider(EmbeddingProvider):
    """Create embeddings in the ostra process with FastEmbed, on the CPU.

    The model files come from Hugging Face on first use and stay in the cache folder, so that later runs work
    offline. The model loads on the first text, so that commands that embed nothing do not download it.
    """

    def __init__(self, spec: EmbeddingModelSpec, cache_directory: Path):
        super().__init__(spec)
        self._cache_directory = cache_directory
        self._model: TextEmbedding | None = None

    def _embed(self, text: str) -> list[float]:
        """Run the model on the text and return its embedding vector.

        Args:
            text: The text to embed, with its prefix.

        Returns:
            The embedding vector that the model creates for the text.

        Raises:
            ValueError: When the model returns no embedding.
        """
        embedding = next(iter(self._loaded_model().embed([text])), None)
        if embedding is None:
            raise ValueError(f"FastEmbed returned no embedding. Model={self._spec.provider_model}")

        return [float(value) for value in embedding]

    def _loaded_model(self) -> TextEmbedding:
        """Return the model, and load it on the first call. The first load downloads the model files.

        Returns:
            The FastEmbed model.
        """
        if self._model is None:
            logger.info(
                "Loading the embedding model. It is downloaded on first use. Model=%s Cache=%s",
                self._spec.provider_model,
                self._cache_directory,
            )
            self._model = TextEmbedding(model_name=self._spec.provider_model, cache_dir=str(self._cache_directory))

        return self._model
