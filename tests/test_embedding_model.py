import pytest

from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.ollama_embedding_provider import OllamaSettings
from ostra.models.base import EmbeddingModel
from ostra.models.mxbai_embedding_model import MxbaiEmbeddingModel
from ostra.tokenization.base import Tokenizer
from ostra.tokenization.mxbai_tokenizer import MxbaiTokenizer


def test_embedding_model__instantiated_directly__raises_type_error():
    """Raise TypeError when the abstract embedding model is instantiated."""
    with pytest.raises(TypeError):
        EmbeddingModel()


def test_tokenizer__instantiated_directly__raises_type_error():
    """Raise TypeError when the abstract tokenizer is instantiated."""
    with pytest.raises(TypeError):
        Tokenizer()


def test_mxbai_embedding_model__returns_provider_and_tokenizer_of_its_type():
    """Return an embedding provider and a tokenizer for the mxbai model."""
    model = MxbaiEmbeddingModel(settings=OllamaSettings(base_url="http://127.0.0.1:1", model="test-model"))

    provider = model.get_embedding_provider()
    tokenizer = model.get_tokenizer()

    assert isinstance(model, EmbeddingModel)
    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(tokenizer, MxbaiTokenizer)
    assert isinstance(tokenizer, Tokenizer)
