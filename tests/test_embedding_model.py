import pytest
from conftest import make_spec

from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.vector import Vector
from ostra.models.embedding_model import EmbeddingModel
from ostra.tokenization.base import Tokenizer


class FakeTokenizer(Tokenizer):
    """Count each word as one token."""

    def count_tokens(self, text: str) -> int:
        """Return the number of words in the text."""
        return len(text.split())


class FakeProvider(EmbeddingProvider):
    """Return a vector of zeros and remember the texts that it embeds."""

    def __init__(self, spec):
        super().__init__(spec)
        self.texts: list[str] = []

    def _embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.0] * self._spec.dimensions


def make_model(**overrides) -> EmbeddingModel:
    spec = make_spec("m", 2, **overrides)
    return EmbeddingModel(spec=spec, provider=FakeProvider(spec), tokenizer=FakeTokenizer())


def test_embedding_provider__instantiated_directly__raises_type_error():
    """Raise TypeError when the abstract embedding provider is instantiated."""
    with pytest.raises(TypeError):
        EmbeddingProvider(make_spec("m", 2))


def test_tokenizer__instantiated_directly__raises_type_error():
    """Raise TypeError when the abstract tokenizer is instantiated."""
    with pytest.raises(TypeError):
        Tokenizer()


def test_embed_query_and_document__prefixes_set__apply_each_prefix_and_stamp_model_id():
    """Put the query prefix in front of queries and the document prefix in front of chunks."""
    model = make_model(query_prefix="q: ", document_prefix="d: ")

    query = model.provider.embed_query("hello")
    document = model.provider.embed_document("hello")

    assert model.provider.texts == ["q: hello", "d: hello"]
    assert query == Vector("m", (0.0, 0.0))
    assert document == Vector("m", (0.0, 0.0))


def test_chunk_size_tokens__not_configured__returns_default_of_the_model():
    """Use the default chunk size of the model when the config does not set one."""
    assert make_model(default_chunk_size_tokens=300).chunk_size_tokens(None) == 300


def test_chunk_size_tokens__configured__returns_configured_value():
    """Use the configured chunk size when it fits the model."""
    assert make_model().chunk_size_tokens(200) == 200


def test_chunk_size_tokens__larger_than_model_input__raises_value_error():
    """Refuse a chunk size that does not fit the model input after the special tokens."""
    with pytest.raises(ValueError, match="too large.*ChunkSize=511 Largest=510"):
        make_model().chunk_size_tokens(511)


def test_chunk_size_tokens__document_prefix_uses_tokens__lowers_the_largest_size():
    """Count the tokens of the document prefix against the model input."""
    model = make_model(document_prefix="search document:")

    with pytest.raises(ValueError, match="Largest=508"):
        model.chunk_size_tokens(509)
