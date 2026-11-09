from ollama import Client
from pydantic_settings import BaseSettings, SettingsConfigDict

from ostra.embeddings.base import EmbeddingProvider
from ostra.models.embedding_model_spec import EmbeddingModelSpec


class OllamaSettings(BaseSettings):
    """Hold the address of an Ollama server.

    The value can come from the OLLAMA_BASE_URL environment variable. The model is not a setting: the model
    spec names it.
    """

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"


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
        response = self._client.embed(model=self._spec.ollama_model, input=text, truncate=False)
        if not response.embeddings:
            raise ValueError(f"Ollama response contains no embedding. Model={self._spec.ollama_model}")

        return list(response.embeddings[0])
