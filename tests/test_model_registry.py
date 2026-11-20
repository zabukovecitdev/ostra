from dataclasses import replace

import pytest

from ostra.enums.embedding_provider_name import EmbeddingProviderName
from ostra.models.embedding_model_registry import (
    BGE_M3,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    MXBAI_EMBED_LARGE,
    POTION_MULTILINGUAL_128M,
    get_model_spec,
)


def test_embedding_models__each_entry__is_keyed_by_its_model_id():
    """Key each spec by its own ID, so that the registry is the only list of models."""
    assert all(model_id == spec.model_id for model_id, spec in EMBEDDING_MODELS.items())


def test_default_embedding_model__is_in_the_registry():
    """Name a default model that exists."""
    assert DEFAULT_EMBEDDING_MODEL in EMBEDDING_MODELS


def test_mxbai_spec__legacy_index_literal__matches_migration_006():
    """Keep the model ID and size that migration 006 gives to the existing index."""
    assert (MXBAI_EMBED_LARGE.model_id, MXBAI_EMBED_LARGE.dimensions) == ("mxbai-embed-large", 1024)


def test_get_model_spec__unknown_model__raises_value_error_that_lists_supported_models():
    """Name the supported models in the error."""
    with pytest.raises(
        ValueError, match="Model=magic. Supported models: potion-multilingual-128m, mxbai-embed-large, bge-m3"
    ):
        get_model_spec("magic")


def test_bge_m3_spec__is_in_the_registry__with_its_own_tokenizer():
    """Register bge-m3 with its own tokenizer file, which the package ships."""
    assert get_model_spec("bge-m3") == BGE_M3
    assert BGE_M3.tokenizer_file == "bge-m3/tokenizer.json"


def test_fingerprint__mxbai_and_bge_m3__differ_although_dimensions_are_equal():
    """Keep the two 1024-dimension models apart, so that their vectors never share an index."""
    assert BGE_M3.dimensions == MXBAI_EMBED_LARGE.dimensions
    assert BGE_M3.fingerprint != MXBAI_EMBED_LARGE.fingerprint


def test_fingerprint__document_prefix_changed__changes():
    """Change the fingerprint when vectors of chunks mean something else."""
    assert replace(MXBAI_EMBED_LARGE, document_prefix="doc: ").fingerprint != MXBAI_EMBED_LARGE.fingerprint


def test_fingerprint__query_prefix_changed__stays_the_same():
    """Keep the fingerprint when only the query prefix changes, because query vectors are made fresh."""
    assert replace(MXBAI_EMBED_LARGE, query_prefix="query: ").fingerprint == MXBAI_EMBED_LARGE.fingerprint


def test_default_embedding_model__runs_in_process__so_no_server_is_needed():
    """Index right after installation: the default model runs in the ostra process, not on an Ollama server."""
    assert get_model_spec(DEFAULT_EMBEDDING_MODEL) == POTION_MULTILINGUAL_128M
    assert POTION_MULTILINGUAL_128M.provider is EmbeddingProviderName.FASTEMBED


def test_fingerprint__ollama_models__match_the_indexes_that_earlier_versions_wrote():
    """Keep the stored fingerprints of the Ollama models valid, so that existing indexes still open."""
    assert MXBAI_EMBED_LARGE.fingerprint == "8ade69b61a22d966f1ea5ee7e2ebba0ca8ef32a7ac8c9b2b87f8444f297b1825"
    assert BGE_M3.fingerprint == "7329bb8cd13716fe4bdde1c56820267c816768a8065ecdcd3674651c0a77ae04"


def test_fingerprint__same_model_on_another_provider__changes():
    """Keep the vectors of two providers apart, because they can differ for the same weights."""
    other = replace(MXBAI_EMBED_LARGE, provider=EmbeddingProviderName.FASTEMBED)

    assert other.fingerprint != MXBAI_EMBED_LARGE.fingerprint
