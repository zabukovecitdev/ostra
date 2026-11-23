from conftest import ConstantEmbeddingProvider, make_spec

from ostra.models.chunk_match import ChunkMatch
from ostra.models.file_match import FileMatch
from ostra.models.vector import Vector
from ostra.repositories.model_index import ChunkSearchIndex
from ostra.services.chunk_search_service import (
    MAX_FILE_SEARCH_CHUNKS,
    MIN_FILE_SEARCH_CHUNKS,
    ChunkSearchService,
    group_by_file,
)

SPEC = make_spec("test-model", 4)


def chunk(document_id: str, distance: float, text: str = "text") -> ChunkMatch:
    """Build a chunk match of a file."""
    return ChunkMatch(
        document_id=document_id, source="filesystem", title=document_id, digest=text, text=text, distance=distance
    )


class FixedIndex(ChunkSearchIndex):
    """Return the given chunks for every query and remember the requested limits."""

    def __init__(self, matches: list[ChunkMatch]):
        self._matches = matches
        self.limits: list[int] = []

    def search(self, query: Vector, limit: int) -> list[ChunkMatch]:
        self.limits.append(limit)
        return self._matches[:limit]


def make_service(matches: list[ChunkMatch]) -> tuple[ChunkSearchService, FixedIndex]:
    """Build a search service over fixed chunks."""
    index = FixedIndex(matches)
    return ChunkSearchService(embedding_provider=ConstantEmbeddingProvider(SPEC), model_index=index), index


def test_group_by_file__several_chunks_of_one_file__keeps_nearest_and_counts_them():
    """Combine the chunks of a file into one match with its nearest chunk."""
    matches = [chunk("a.md", 0.3, "far"), chunk("b.md", 0.2), chunk("a.md", 0.1, "near")]

    files = group_by_file(matches)

    assert files == [
        FileMatch(document_id="a.md", title="a.md", distance=0.1, match_count=2, best_text="near"),
        FileMatch(document_id="b.md", title="b.md", distance=0.2, match_count=1, best_text="text"),
    ]


def test_group_by_file__no_chunks__returns_no_files():
    """Return an empty list for a search without chunks."""
    assert group_by_file([]) == []


def test_search_files__more_files_than_limit__returns_nearest_files():
    """Return at most limit files, nearest first."""
    service, _ = make_service([chunk("a.md", 0.1), chunk("a.md", 0.2), chunk("b.md", 0.3), chunk("c.md", 0.4)])

    files = service.search_files("query", limit=2)

    assert [file.document_id for file in files] == ["a.md", "b.md"]


def test_search_files__small_limit__reads_more_chunks_than_files():
    """Read enough chunks to find several files even when the nearest chunks share a file."""
    service, index = make_service([])

    service.search_files("query", limit=1)

    assert index.limits == [MIN_FILE_SEARCH_CHUNKS]


def test_search_files__huge_limit__reads_at_most_the_sqlite_vec_limit():
    """Stay within the largest number of neighbours that sqlite-vec accepts."""
    service, index = make_service([])

    service.search_files("query", limit=10_000)

    assert index.limits == [MAX_FILE_SEARCH_CHUNKS]


def test_search_files__max_distance__drops_far_chunks_before_grouping():
    """Count only the chunks within the distance, and drop files without such chunks."""
    service, _ = make_service([chunk("a.md", 0.1), chunk("a.md", 0.6), chunk("b.md", 0.7)])

    files = service.search_files("query", limit=5, max_distance=0.5)

    assert [(file.document_id, file.match_count) for file in files] == [("a.md", 1)]
