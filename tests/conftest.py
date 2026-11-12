from collections.abc import Iterator
from pathlib import Path

import pytest

from ostra.models.embedding_model_spec import EmbeddingModelSpec
from ostra.repositories.sqlite.sqlite_chunk_store import SqliteChunkStore
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex


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


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Provide the location of the SQLite database file for a test."""
    return tmp_path / "ostra.db"


@pytest.fixture
def store(database_path: Path) -> Iterator[SqliteChunkStore]:
    """Provide an open store and close it after the test."""
    with SqliteChunkStore.open(database_path) as opened:
        yield opened


@pytest.fixture
def index(store: SqliteChunkStore) -> SqliteModelIndex:
    """Provide the index of the small test model."""
    return store.model_index(TEST_SPEC, create=True)
