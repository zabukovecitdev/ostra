from ostra.models.chunk_match import ChunkMatch
from ostra.models.document import Document
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE as MXBAI
from ostra.models.vector import Vector
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.repositories.model_index import ChunkSearchIndex, ChunkWriteIndex, ModelIndex
from ostra.repositories.sqlite.sqlite_model_index import SqliteModelIndex
from ostra.services.chunk_search_service import ChunkSearchService
from ostra.services.ingestion.ingestion_pipeline import IngestionPipeline
from ostra.services.ingestion.text_chunker import TextChunker


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return the same vector for every text."""

    def _embed(self, text: str) -> list[float]:
        """Return a fixed vector."""
        return [0.5] * 1024


def count_words(text: str) -> int:
    """Count each word as one token."""
    return len(text.split())


class RecordingIndex(ChunkWriteIndex):
    """Keep the saved documents in memory instead of a database."""

    def __init__(self):
        self.saved: dict[str, list[str]] = {}

    def stored_digests(self, document_id: str) -> set[str]:
        """Return no digests."""
        return set()

    def save_document(self, document, new_chunks, current_digests) -> None:
        """Remember the text of each chunk."""
        self.saved[document.document_id] = [chunk.text for chunk in new_chunks]


class FixedSearchIndex(ChunkSearchIndex):
    """Return one fixed match."""

    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        """Return the same match for every query."""
        return [ChunkMatch(document_id="a.txt", source="s", title="a", digest="d", text="hit", distance=0.1)]


def test_sqlite_model_index__class__implements_the_full_interface():
    """Offer both the search and the write interface."""
    assert issubclass(SqliteModelIndex, ModelIndex)


def test_ingestion_pipeline__in_memory_index__stores_chunks_without_sqlite():
    """Run the pipeline against any ChunkWriteIndex."""
    index = RecordingIndex()
    chunker = TextChunker(chunk_size_tokens=10, chunk_overlap_tokens=0, count_tokens=count_words)
    pipeline = IngestionPipeline(
        model_index=index, chunker=chunker, embedding_provider=FakeEmbeddingProvider(MXBAI), reindex=False
    )

    pipeline.process_documents([Document(source="s", document_id="a.txt", title="a", text="alpha beta")])

    assert index.saved == {"a.txt": ["alpha beta"]}


def test_chunk_search_service__in_memory_index__returns_its_matches():
    """Search through any ChunkSearchIndex."""
    service = ChunkSearchService(embedding_provider=FakeEmbeddingProvider(MXBAI), model_index=FixedSearchIndex())

    assert [match.text for match in service.search("q", limit=1)] == ["hit"]
