from dataclasses import replace

import pytest

from ostra.models.registry import BGE_M3, DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, MXBAI_EMBED_LARGE, get_model_spec


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
    with pytest.raises(ValueError, match="Model=magic. Supported models: mxbai-embed-large, bge-m3"):
        get_model_spec("magic")


def test_fingerprint__document_prefix_changed__changes():
    """Change the fingerprint when vectors of chunks mean something else."""
    assert replace(MXBAI_EMBED_LARGE, document_prefix="doc: ").fingerprint != MXBAI_EMBED_LARGE.fingerprint


def test_fingerprint__query_prefix_changed__stays_the_same():
    """Keep the fingerprint when only the query prefix changes, because query vectors are made fresh."""
    assert replace(MXBAI_EMBED_LARGE, query_prefix="query: ").fingerprint == MXBAI_EMBED_LARGE.fingerprint


def test_fingerprint__mxbai_and_bge_m3__differ_although_dimensions_are_equal():
    """Tell the two models apart, because both make 1024 dimensions."""
    assert BGE_M3.dimensions == MXBAI_EMBED_LARGE.dimensions
    assert BGE_M3.fingerprint != MXBAI_EMBED_LARGE.fingerprint


def test_chunk_size__every_model__fits_the_model_input_after_special_tokens():
    """Keep the default chunk size inside the input limit of each model."""
    assert all(
        spec.default_chunk_size_tokens + spec.special_tokens <= spec.max_tokens for spec in EMBEDDING_MODELS.values()
    )
