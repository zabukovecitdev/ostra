import os

from ostra.config import AppConfig
from ostra.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider, OllamaSettings
from ostra.models.embedding_model import EmbeddingModel
from ostra.models.provider import EmbeddingProviderName
from ostra.models.registry import get_model_spec
from ostra.tokenization.hugging_face_tokenizer import HuggingFaceTokenizer


def ollama_settings(config: AppConfig) -> OllamaSettings:
    """Build the Ollama settings from the config file and the environment.

    The server address in the config file wins over the OLLAMA_BASE_URL environment variable.

    Args:
        config: The application settings.

    Returns:
        The Ollama server address.

    Raises:
        ValueError: When the OLLAMA_MODEL environment variable is set. The model comes from the config file.
    """
    if "OLLAMA_MODEL" in os.environ:
        raise ValueError(
            "The OLLAMA_MODEL environment variable is not used any more. "
            "Set embedding_model in the config file, or use the --model option."
        )

    if config.ollama_base_url is None:
        return OllamaSettings()

    return OllamaSettings(base_url=config.ollama_base_url)


def build_embedding_model(config: AppConfig, model_id: str | None = None) -> EmbeddingModel:
    """Build the embedding model that the config file selects, or the model that is named.

    Args:
        config: The application settings.
        model_id: The ID of the model to build. None means the model of the config file.

    Returns:
        The embedding model with its provider and tokenizer.

    Raises:
        ValueError: When the provider or the model is not supported.
    """
    if config.embedding_provider != EmbeddingProviderName.OLLAMA:
        raise ValueError(f"Embedding provider is not supported. Provider={config.embedding_provider}")

    spec = get_model_spec(model_id or config.embedding_model)
    return EmbeddingModel(
        spec=spec,
        provider=OllamaEmbeddingProvider(spec, ollama_settings(config)),
        tokenizer=HuggingFaceTokenizer(spec.tokenizer_file),
    )
