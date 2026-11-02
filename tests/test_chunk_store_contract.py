from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from conftest import TEST_SPEC, make_spec

from ostra.embeddings.vector import Vector
from ostra.store.chunk_store import ChunkStore
from ostra.store.database_backend import DatabaseBackend
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk
from ostra.store.model_index import ModelIndex
from ostra.store.model_index_mismatch_error import ModelIndexMismatchError
from ostra.store.model_not_indexed_error import ModelNotIndexedError
from ostra.store.open_chunk_store import open_chunk_store

OTHER_SPEC = make_spec("other-model", 3)


def one_hot(position: int, size: int = 4, model_id: str = TEST_SPEC.model_id) -> Vector:
    """Return a vector with a single 1.0 at the position, so that vectors differ in direction."""
    values = [0.0] * size
    values[position] = 1.0
    return Vector(model_id=model_id, values=tuple(values))


def document(document_id: str = "/notes/a.txt", source: str = "filesystem", title: str = "a.txt") -> DocumentRecord:
    return DocumentRecord(document_id=document_id, source=source, title=title)


def chunk(digest: str, text: str = "chunk text", position: int = 0) -> EmbeddedChunk:
    return EmbeddedChunk(digest=digest, text=text, embedding=one_hot(position))


def save(index: ModelIndex, chunks: Iterable[EmbeddedChunk], doc: DocumentRecord | None = None, keep=None) -> None:
    """Save the chunks of a document. Keep the digests of the given chunks, or of all saved chunks."""
    chunks = list(chunks)
    index.save_document(
        doc or document(), chunks, current_digests=keep if keep is not None else {c.digest for c in chunks}
    )


def test_stored_digests__unknown_document__returns_empty_set(index: ModelIndex):
    """Return an empty set for a document without saved chunks."""
    assert index.stored_digests("/notes/a.txt") == set()


def test_save_document__two_chunks__returns_both_digests(index: ModelIndex):
    """Return every chunk digest that was saved for the document."""
    save(index, [chunk("chunk-1"), chunk("chunk-2", position=1)])

    assert index.stored_digests("/notes/a.txt") == {"chunk-1", "chunk-2"}


def test_save_document__same_digest_twice__stores_one_chunk(index: ModelIndex):
    """Store a repeated chunk digest only once for the document."""
    save(index, [chunk("chunk-1")])
    save(index, [chunk("chunk-1")])

    assert index.stored_digests("/notes/a.txt") == {"chunk-1"}


def test_stored_digests__other_document__not_returned(index: ModelIndex):
    """Return only the chunk digests of the requested document."""
    save(index, [chunk("chunk-1")])
    save(index, [chunk("chunk-2", position=1)], document("/notes/b.txt", title="b.txt"))

    assert index.stored_digests("/notes/a.txt") == {"chunk-1"}


@pytest.mark.parametrize("digest", ["chunk-1", "chunk-2"])
def test_save_document__same_digest_in_two_documents__keeps_documents_apart(index: ModelIndex, digest: str):
    """Keep the chunks of different documents apart even when the digest is the same."""
    save(index, [chunk(digest)])
    save(index, [chunk(digest)], document("/notes/b.txt", title="b.txt"))

    assert index.stored_digests("/notes/a.txt") == {digest}
    assert index.stored_digests("/notes/b.txt") == {digest}


def test_save_document__digest_not_current__deletes_the_chunk(index: ModelIndex):
    """Delete a stored chunk whose digest the document no longer has, and keep the others."""
    save(index, [chunk("chunk-1"), chunk("chunk-2", position=1)])

    save(index, [], keep={"chunk-2"})

    assert index.stored_digests("/notes/a.txt") == {"chunk-2"}


def test_save_document__no_new_chunks_all_current__keeps_chunks(index: ModelIndex):
    """Keep every chunk when all digests are current and nothing is new."""
    save(index, [chunk("chunk-1")])

    save(index, [], keep={"chunk-1"})

    assert index.stored_digests("/notes/a.txt") == {"chunk-1"}


def test_save_document__other_documents_chunks__not_deleted(index: ModelIndex):
    """Delete stale chunks of the saved document only."""
    save(index, [chunk("chunk-1")], document("/notes/b.txt", title="b.txt"))

    save(index, [], keep=set())

    assert index.stored_digests("/notes/b.txt") == {"chunk-1"}


def test_save_document__chunks_given_lazily__saves_each_chunk_when_it_is_produced(index: ModelIndex):
    """Keep the chunks that were saved before an error, so that an interrupted run can continue."""

    def chunks():
        yield chunk("chunk-1")
        raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        index.save_document(document(), chunks(), current_digests={"chunk-1", "chunk-2"})

    assert index.stored_digests("/notes/a.txt") == {"chunk-1"}


def test_search__query_matches_one_chunk__returns_nearest_first_with_document_details(index: ModelIndex):
    """Return the chunk with the same direction first, with its text, document details and a distance of 0."""
    save(index, [chunk("chunk-1", "first text", 1), chunk("chunk-2", "second text", 2)], document(title="Notes A"))

    matches = index.search(query=one_hot(1), limit=2)

    assert [(m.document_id, m.source, m.title, m.digest, m.text) for m in matches] == [
        ("/notes/a.txt", "filesystem", "Notes A", "chunk-1", "first text"),
        ("/notes/a.txt", "filesystem", "Notes A", "chunk-2", "second text"),
    ]
    assert matches[0].distance == pytest.approx(0.0, abs=1e-6)
    assert matches[1].distance == pytest.approx(1.0, abs=1e-6)


def test_search__limit_reached__returns_only_limit_matches(index: ModelIndex):
    """Return no more matches than the limit."""
    save(index, [chunk("chunk-1", position=1), chunk("chunk-2", position=2)])

    assert [match.digest for match in index.search(query=one_hot(1), limit=1)] == ["chunk-1"]


def test_search__no_chunks_stored__returns_empty_list(index: ModelIndex):
    """Return an empty list when no chunk is stored."""
    assert index.search(query=one_hot(1), limit=5) == []


def test_search__chunk_deleted__is_not_returned(index: ModelIndex):
    """Stop returning a chunk after it is deleted, so its vector is removed too."""
    save(index, [chunk("chunk-1", position=1), chunk("chunk-2", position=2)])

    save(index, [], keep={"chunk-2"})

    assert [match.digest for match in index.search(query=one_hot(1), limit=5)] == ["chunk-2"]


def test_save_document__digest_saved_again__replaces_text_and_vector(index: ModelIndex):
    """Replace the text and the vector of a stored chunk when the same digest is saved again."""
    save(index, [chunk("chunk-1", "old text", 1)])

    save(index, [chunk("chunk-1", "new text", 2)])
    matches = index.search(query=one_hot(2), limit=5)

    assert [(m.text, m.distance) for m in matches] == [("new text", pytest.approx(0.0, abs=1e-6))]


def test_save_document__title_changed__search_returns_new_title(index: ModelIndex):
    """Return the latest title of a document."""
    save(index, [chunk("chunk-1", position=1)], document(title="Old"))

    save(index, [], document(title="New"), keep={"chunk-1"})

    assert index.search(query=one_hot(1), limit=1)[0].title == "New"


def test_model_index__two_models__keep_vectors_and_digests_apart(store: ChunkStore):
    """Give each model its own chunks, so that vectors of one model never appear in the other model."""
    first = store.model_index(TEST_SPEC, create=True)
    second = store.model_index(OTHER_SPEC, create=True)

    save(first, [chunk("chunk-1", "from first", 1)])
    second.save_document(
        document(),
        [EmbeddedChunk("chunk-2", "from second", one_hot(1, size=3, model_id=OTHER_SPEC.model_id))],
        current_digests={"chunk-2"},
    )

    assert first.stored_digests("/notes/a.txt") == {"chunk-1"}
    assert second.stored_digests("/notes/a.txt") == {"chunk-2"}
    assert [m.text for m in first.search(one_hot(1), limit=5)] == ["from first"]
    assert [m.text for m in second.search(one_hot(1, size=3, model_id=OTHER_SPEC.model_id), limit=5)] == ["from second"]


def test_model_index__same_digest_in_two_models__does_not_overwrite_the_other_model(store: ChunkStore):
    """Keep a chunk of one model when another model saves or deletes a chunk with the same digest."""
    first = store.model_index(TEST_SPEC, create=True)
    second = store.model_index(OTHER_SPEC, create=True)
    save(first, [chunk("same", "first text", 1)])

    second.save_document(
        document(), [EmbeddedChunk("same", "second text", one_hot(1, size=3, model_id=OTHER_SPEC.model_id))], {"same"}
    )
    second.save_document(document(), [], current_digests=set())

    assert [m.text for m in first.search(one_hot(1), limit=5)] == ["first text"]


def test_model_index__model_never_indexed__raises_error_that_lists_indexed_models(store: ChunkStore):
    """Raise ModelNotIndexedError for a read of a model without an index, and name the models that have one."""
    save(store.model_index(TEST_SPEC, create=True), [chunk("chunk-1")])

    with pytest.raises(ModelNotIndexedError, match="other-model.*test-model"):
        store.model_index(OTHER_SPEC, create=False)


def test_model_index__read_without_create__does_not_create_the_index(store: ChunkStore):
    """Create no index on a read, so that a search never changes the schema."""
    with pytest.raises(ModelNotIndexedError):
        store.model_index(OTHER_SPEC, create=False)

    with pytest.raises(ModelNotIndexedError):
        store.model_index(OTHER_SPEC, create=False)


def test_model_index__dimensions_changed__raises_mismatch_error(store: ChunkStore):
    """Refuse an index that was built with other dimensions than the model has now."""
    store.model_index(TEST_SPEC, create=True)
    changed = replace(TEST_SPEC, dimensions=8)

    with pytest.raises(ModelIndexMismatchError, match="Stored dimensions: 4. Expected dimensions: 8"):
        store.model_index(changed, create=True)


def test_model_index__same_dimensions_but_other_model_settings__raises_mismatch_error(store: ChunkStore):
    """Refuse an index when the Ollama model changed but the dimensions stayed the same."""
    store.model_index(TEST_SPEC, create=True)
    changed = replace(TEST_SPEC, ollama_model="newer:latest")

    with pytest.raises(ModelIndexMismatchError):
        store.model_index(changed, create=True)


def test_model_index__document_prefix_changed__raises_mismatch_error(store: ChunkStore):
    """Refuse an index when the document prefix changed, because the stored vectors were made without it."""
    store.model_index(TEST_SPEC, create=True)
    changed = replace(TEST_SPEC, document_prefix="doc: ")

    with pytest.raises(ModelIndexMismatchError):
        store.model_index(changed, create=True)


def test_save_then_reopen__data_persisted(backend: DatabaseBackend, database_path: Path, store: ChunkStore):
    """Read back the saved data from a new store that opens the same database."""
    save(store.model_index(TEST_SPEC, create=True), [chunk("chunk-1", "chunk text", 1)])
    store.close()
    # Chroma keeps one shared system per path inside the process. Clear it so the reopened client reads the disk.
    SharedSystemClient.clear_system_cache()

    with open_chunk_store(backend, database_path) as reopened:
        index = reopened.model_index(TEST_SPEC, create=False)
        assert index.stored_digests("/notes/a.txt") == {"chunk-1"}
        assert [m.text for m in index.search(one_hot(1), limit=1)] == ["chunk text"]
    SharedSystemClient.clear_system_cache()


def test_save_document__vector_of_other_model__raises_value_error(index: ModelIndex):
    """Refuse a chunk whose vector comes from another model, even when the size fits."""
    foreign = EmbeddedChunk("chunk-1", "text", one_hot(1, model_id="other-model"))

    with pytest.raises(ValueError, match="does not fit the index"):
        index.save_document(document(), [foreign], current_digests={"chunk-1"})

    assert index.stored_digests("/notes/a.txt") == set()


def test_save_document__vector_with_wrong_size__raises_value_error(index: ModelIndex):
    """Refuse a chunk whose vector has another size than the model."""
    wrong = EmbeddedChunk("chunk-1", "text", one_hot(1, size=3))

    with pytest.raises(ValueError, match="does not fit the index"):
        index.save_document(document(), [wrong], current_digests={"chunk-1"})


def test_search__query_vector_of_other_model__raises_value_error(index: ModelIndex):
    """Refuse a query vector from another model, so that vectors of two models are never compared."""
    with pytest.raises(ValueError, match="does not fit the index"):
        index.search(one_hot(1, model_id="other-model"), limit=5)


def test_model_index__query_prefix_changed__does_not_raise(store: ChunkStore):
    """Accept a changed query prefix, because query vectors are made fresh for each search."""
    store.model_index(TEST_SPEC, create=True)

    store.model_index(replace(TEST_SPEC, query_prefix="query: "), create=True)
