from ollama import Client
from pydantic_settings import BaseSettings, SettingsConfigDict

from ostra.embeddings.base import EmbeddingProvider


class OllamaSettings(BaseSettings):
    """Hold the settings for a local Ollama server and its embedding model.

    Each value can come from an environment variable with the OLLAMA_ prefix, such as OLLAMA_MODEL.
    """

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    base_url: str = "http://localhost:11434"
    model: str = "mxbai-embed-large:latest"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Create embeddings with a model that runs on a local Ollama server."""

    def __init__(self, settings: OllamaSettings):
        self._settings = settings
        self._client = Client(host=settings.base_url)

    def get_embedding(self, text: str) -> list[float]:
        """Send the text to the Ollama server and return its embedding vector.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector that the model returns for the text.

        Raises:
            ValueError: When the server response contains no embedding.
        """
        response = self._client.embed(model=self._settings.model, input=text)
        if not response.embeddings:
            raise ValueError(f"Ollama response contains no embedding. Model={self._settings.model}")

        return list(response.embeddings[0])
