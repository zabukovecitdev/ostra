from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Hold one document that a source offers to the ingestion pipeline.

    The pipeline stores the chunks of a document under its `document_id`. It does not know where the
    document comes from.
    """

    source: str
    document_id: str
    title: str
    text: str
