from dataclasses import dataclass


@dataclass(frozen=True)
class FileMatch:
    """Hold one file of a search, with its nearest chunk."""

    document_id: str
    title: str
    distance: float
    match_count: int
    best_text: str
