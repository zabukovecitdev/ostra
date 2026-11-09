from collections.abc import Iterator
from pathlib import Path

import pytest

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.store.chunk_store import ChunkStore
from ostra.store.database_backend import DatabaseBackend
from ostra.store.model_index import ModelIndex
from ostra.store.open_chunk_store import open_chunk_store


def make_spec(model_id: str, dimensions: int, **overrides) -> EmbeddingModelSpec:
    """Build a model spec for a test. A small vector keeps the tests light."""
    values = {
        "model_id": model_id,
        "ollama_model": f"{model_id}:latest",
        "dimensions": dimensions,
        "max_tokens": 512,
        "special_tokens": 2,
        "default_chunk_size_tokens": 510,
        "tokenizer_file": "mxbai-embed-large-v1/tokenizer.json",
    }
    return EmbeddingModelSpec(**(values | overrides))


TEST_SPEC = make_spec("test-model", 4)


@pytest.fixture(params=[DatabaseBackend.SQLITE, DatabaseBackend.CHROMA])
def backend(request: pytest.FixtureRequest) -> DatabaseBackend:
    """Provide each database backend in turn."""
    return request.param


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide the location of the SQLite database file for a test. The Chroma store is beside it."""
    return tmp_path / "ostra.db"


@pytest.fixture
def store(backend: DatabaseBackend, database_path: Path) -> Iterator[ChunkStore]:
    """Provide an open store for each backend and close it after the test."""
    with open_chunk_store(backend, database_path) as opened:
        yield opened


@pytest.fixture
def index(store: ChunkStore) -> ModelIndex:
    """Provide the index of the small test model."""
    return store.model_index(TEST_SPEC, create=True)
