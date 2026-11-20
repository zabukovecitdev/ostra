from ollama import Client

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.models.ollama_settings import OllamaSettings
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Create embeddings with a model that runs on an Ollama server."""

    def __init__(self, spec: EmbeddingModelSpec, settings: OllamaSettings):
        super().__init__(spec)
        self._client = Client(host=settings.base_url)

    def _embed(self, text: str) -> list[float]:
        """Send the text to the Ollama server and return its embedding vector.

        The server must not cut the text. A text that is too long makes the request fail, so that no chunk is
        stored with a vector of a shortened text.

        Args:
            text: The text to embed, with its prefix.

        Returns:
            The embedding vector that the model returns for the text.

        Raises:
            ValueError: When the server response contains no embedding.
        """
        response = self._client.embed(model=self._spec.provider_model, input=text, truncate=False)
        if not response.embeddings:
            raise ValueError(f"Ollama response contains no embedding. Model={self._spec.provider_model}")

        return list(response.embeddings[0])
