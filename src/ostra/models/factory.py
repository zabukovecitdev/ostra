from ostra.config import AppConfig
from ostra.embeddings.ollama_embedding_provider import OllamaSettings
from ostra.models.base import EmbeddingModel
from ostra.models.mxbai_embedding_model import MxbaiEmbeddingModel
from ostra.models.provider import EmbeddingProviderName


def ollama_settings(config: AppConfig) -> OllamaSettings:
    """Build the Ollama settings from the config file and the environment.

    The server address in the config file wins over the OLLAMA_BASE_URL environment variable.

    Args:
        config: The application settings.

    Returns:
        The Ollama server address and model name.
    """
    if config.ollama_base_url is None:
        return OllamaSettings()

    return OllamaSettings(base_url=config.ollama_base_url)


def build_embedding_model(config: AppConfig) -> EmbeddingModel:
    """Build the embedding model that the config file selects.

    Args:
        config: The application settings.

    Returns:
        The embedding model with its tokenizer.

    Raises:
        ValueError: When the provider is not supported.
    """
    if config.embedding_provider == EmbeddingProviderName.OLLAMA:
        return MxbaiEmbeddingModel(settings=ollama_settings(config))

    raise ValueError(f"Embedding provider is not supported. Provider={config.embedding_provider}")
