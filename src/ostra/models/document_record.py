from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentRecord:
    """Hold what the store keeps about a document, apart from its chunks."""

    document_id: str
    source: str
    title: str
