from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import make_spec

from ostra.enums.embedding_provider_name import EmbeddingProviderName
from ostra.models.vector import Vector
from ostra.providers.embeddings import fastembed_embedding_provider
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.providers.embeddings.fastembed_embedding_provider import FastEmbedEmbeddingProvider

SPEC = make_spec(
    "test-model",
    3,
    provider=EmbeddingProviderName.FASTEMBED,
    provider_model="owner/test-model",
    query_prefix="query: ",
    document_prefix="doc: ",
)


class FakeTextEmbedding:
    """Stand in for the FastEmbed model, so that the tests download nothing. Remember each load and text."""

    loads: list[dict] = []
    texts: list[str] = []
    vectors: list[list[float]] = [[0.5, 0.25, 0.125]]

    def __init__(self, model_name: str, cache_dir: str):
        type(self).loads.append({"model_name": model_name, "cache_dir": cache_dir})

    def embed(self, documents: Iterable[str]) -> Iterator[np.ndarray]:
        """Return the configured vectors and remember the texts."""
        type(self).texts.extend(documents)
        return iter(np.array(vector, dtype=np.float32) for vector in type(self).vectors)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeTextEmbedding]:
    """Replace the FastEmbed model with the fake and reset what the fake remembers."""
    FakeTextEmbedding.loads = []
    FakeTextEmbedding.texts = []
    FakeTextEmbedding.vectors = [[0.5, 0.25, 0.125]]
    monkeypatch.setattr(fastembed_embedding_provider, "TextEmbedding", FakeTextEmbedding)
    return FakeTextEmbedding


def test_fastembed_embedding_provider__is_an_embedding_provider():
    """Implement the provider interface, so that the services can use it."""
    assert issubclass(FastEmbedEmbeddingProvider, EmbeddingProvider)


def test_init__no_text_embedded__does_not_load_the_model(fake_model: type[FakeTextEmbedding], tmp_path: Path):
    """Load the model on the first text only, so that building the provider downloads nothing."""
    FastEmbedEmbeddingProvider(SPEC, tmp_path)

    assert fake_model.loads == []


def test_embed_document__first_text__loads_the_model_into_the_cache_folder(
    fake_model: type[FakeTextEmbedding], tmp_path: Path
):
    """Load the model of the spec into the given cache folder."""
    FastEmbedEmbeddingProvider(SPEC, tmp_path).embed_document("text")

    assert fake_model.loads == [{"model_name": "owner/test-model", "cache_dir": str(tmp_path)}]


def test_embed_document__two_texts__loads_the_model_once(fake_model: type[FakeTextEmbedding], tmp_path: Path):
    """Keep the loaded model for the next texts."""
    provider = FastEmbedEmbeddingProvider(SPEC, tmp_path)

    provider.embed_document("one")
    provider.embed_query("two")

    assert len(fake_model.loads) == 1


def test_embed_document__model_returns_vector__returns_vector_of_floats(
    fake_model: type[FakeTextEmbedding], tmp_path: Path
):
    """Return the model vector as plain floats with the model ID."""
    vector = FastEmbedEmbeddingProvider(SPEC, tmp_path).embed_document("text")

    assert vector == Vector(model_id="test-model", values=(0.5, 0.25, 0.125))
    assert all(type(value) is float for value in vector.values)


def test_embed_query_and_document__prefixes__are_sent_to_the_model(fake_model: type[FakeTextEmbedding], tmp_path: Path):
    """Put the query and document prefixes of the spec in front of the text."""
    provider = FastEmbedEmbeddingProvider(SPEC, tmp_path)

    provider.embed_query("question")
    provider.embed_document("chunk")

    assert fake_model.texts == ["query: question", "doc: chunk"]


def test_embed_document__model_returns_nothing__raises_value_error(fake_model: type[FakeTextEmbedding], tmp_path: Path):
    """Raise ValueError when the model returns no vector."""
    fake_model.vectors = []

    with pytest.raises(ValueError, match="no embedding"):
        FastEmbedEmbeddingProvider(SPEC, tmp_path).embed_document("text")


def test_embed_document__wrong_vector_size__raises_value_error(fake_model: type[FakeTextEmbedding], tmp_path: Path):
    """Refuse a vector that does not have the size of the spec."""
    fake_model.vectors = [[0.5, 0.25]]

    with pytest.raises(ValueError, match="wrong size"):
        FastEmbedEmbeddingProvider(SPEC, tmp_path).embed_document("text")
