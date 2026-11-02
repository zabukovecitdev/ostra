from collections.abc import Mapping
from types import MappingProxyType

from ostra.models.embedding_model_spec import EmbeddingModelSpec

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

# The model accepts 512 tokens. Two of them are the special tokens [CLS] and [SEP].
MXBAI_EMBED_LARGE = EmbeddingModelSpec(
    model_id="mxbai-embed-large",
    ollama_model="mxbai-embed-large:latest",
    dimensions=1024,
    max_tokens=512,
    special_tokens=2,
    default_chunk_size_tokens=510,
    tokenizer_file="mxbai-embed-large-v1/tokenizer.json",
)

# The model accepts 8192 tokens, and it has the special tokens <s> and </s>. It supports more than 100 languages.
# The chunk size default is the same as for mxbai-embed-large, so that the results of both models compare fairly.
BGE_M3 = EmbeddingModelSpec(
    model_id="bge-m3",
    ollama_model="bge-m3:latest",
    dimensions=1024,
    max_tokens=8192,
    special_tokens=2,
    default_chunk_size_tokens=510,
    tokenizer_file="bge-m3/tokenizer.json",
)

EMBEDDING_MODELS: Mapping[str, EmbeddingModelSpec] = MappingProxyType(
    {spec.model_id: spec for spec in (MXBAI_EMBED_LARGE, BGE_M3)}
)


def get_model_spec(model_id: str) -> EmbeddingModelSpec:
    """Return the description of a supported embedding model.

    Args:
        model_id: The ID of the model.

    Returns:
        The spec of the model.

    Raises:
        ValueError: When the model is not supported. The message lists the supported IDs.
    """
    try:
        return EMBEDDING_MODELS[model_id]
    except KeyError as error:
        supported = ", ".join(EMBEDDING_MODELS)
        raise ValueError(
            f"Embedding model is not supported. Model={model_id}. Supported models: {supported}"
        ) from error
