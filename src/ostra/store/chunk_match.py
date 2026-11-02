from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMatch:
    """Hold one stored chunk returned by a vector search."""

    document_id: str
    source: str
    title: str
    digest: str
    text: str
    distance: float
