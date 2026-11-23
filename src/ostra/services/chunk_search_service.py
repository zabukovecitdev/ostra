from ostra.models.chunk_match import ChunkMatch
from ostra.models.file_match import FileMatch
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.repositories.model_index import ChunkSearchIndex

# A file search reads more chunks than files, because one file can have several close chunks.
CHUNKS_PER_FILE = 5
MIN_FILE_SEARCH_CHUNKS = 50
# sqlite-vec accepts at most 4096 neighbours in one query.
MAX_FILE_SEARCH_CHUNKS = 4096


class ChunkSearchService:
    """Find the stored chunks that are closest in meaning to a query text."""

    def __init__(self, embedding_provider: EmbeddingProvider, model_index: ChunkSearchIndex):
        self._embedding_provider = embedding_provider
        self._model_index = model_index

    def search(self, query: str, limit: int, max_distance: float | None = None) -> list[ChunkMatch]:
        """Embed the query and return the closest stored chunks.

        Args:
            query: The text to search for.
            limit: The number of chunks to return.
            max_distance: Drop the chunks with a larger cosine distance. None keeps every chunk.

        Returns:
            The closest chunks, nearest first.
        """
        matches = self._model_index.search(query=self._embedding_provider.embed_query(query), limit=limit)
        if max_distance is None:
            return matches

        return [match for match in matches if match.distance <= max_distance]

    def search_files(self, query: str, limit: int, max_distance: float | None = None) -> list[FileMatch]:
        """Embed the query and return the files of the closest stored chunks.

        Args:
            query: The text to search for.
            limit: The number of files to return.
            max_distance: Drop the chunks with a larger cosine distance. None keeps every chunk.

        Returns:
            At most limit files, nearest first. A file is as near as its nearest chunk.
        """
        chunk_limit = min(max(limit * CHUNKS_PER_FILE, MIN_FILE_SEARCH_CHUNKS), MAX_FILE_SEARCH_CHUNKS)
        return group_by_file(self.search(query, limit=chunk_limit, max_distance=max_distance))[:limit]


def group_by_file(matches: list[ChunkMatch]) -> list[FileMatch]:
    """Combine the chunks of each file into one file match.

    Args:
        matches: The chunks of a search, nearest first.

    Returns:
        One match for each file, nearest first, with the nearest chunk of the file and its number of chunks.
    """
    nearest: dict[str, ChunkMatch] = {}
    counts: dict[str, int] = {}
    for match in sorted(matches, key=lambda match: match.distance):
        nearest.setdefault(match.document_id, match)
        counts[match.document_id] = counts.get(match.document_id, 0) + 1

    return [
        FileMatch(
            document_id=document_id,
            title=match.title,
            distance=match.distance,
            match_count=counts[document_id],
            best_text=match.text,
        )
        for document_id, match in nearest.items()
    ]
