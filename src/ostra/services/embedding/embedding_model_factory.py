import os
from collections.abc import Callable, Mapping
from types import MappingProxyType

from ostra.enums.embedding_provider_name import EmbeddingProviderName
from ostra.models.app_config import AppConfig
from ostra.models.embedding_model_registry import get_model_spec
from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.models.ollama_settings import OllamaSettings
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.providers.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
from ostra.providers.tokenizers.hugging_face_tokenizer import HuggingFaceTokenizer
from ostra.services.embedding.embedding_model import EmbeddingModel


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


def _build_ollama_provider(spec: EmbeddingModelSpec, config: AppConfig) -> EmbeddingProvider:
    """Build the provider that sends the texts to an Ollama server.

    Args:
        spec: The embedding model.
        config: The application settings.

    Returns:
        The Ollama embedding provider.
    """
    return OllamaEmbeddingProvider(spec, ollama_settings(config))


# A new provider is one new builder here and one new name in EmbeddingProviderName.
PROVIDER_BUILDERS: Mapping[EmbeddingProviderName, Callable[[EmbeddingModelSpec, AppConfig], EmbeddingProvider]] = (
    MappingProxyType({EmbeddingProviderName.OLLAMA: _build_ollama_provider})
)


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
    build_provider = PROVIDER_BUILDERS.get(config.embedding_provider)
    if build_provider is None:
        raise ValueError(f"Embedding provider is not supported. Provider={config.embedding_provider}")

    spec = get_model_spec(model_id or config.embedding_model)
    return EmbeddingModel(
        spec=spec,
        provider=build_provider(spec, config),
        tokenizer=HuggingFaceTokenizer(spec.tokenizer_file),
    )
