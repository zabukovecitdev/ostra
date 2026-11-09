from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Hold one document that a source offers to the ingestion pipeline.

    The pipeline stores the chunks of a document under its `document_id`. It does not know where the
    document comes from.
    """

    source: str
    document_id: str
    version: str
    title: str
    text: str


class DocumentSource(ABC):
    """Define the interface that every source of documents implements."""

    def count_documents(self) -> int | None:
        """Return how many documents the source offers, so that a progress bar can show the total.

        Returns:
            The number of documents, or None when the source cannot know it before it reads them.
        """
        return None

    @abstractmethod
    def stream_documents(self) -> Iterator[Document]:
        """Yield each document of the source, one at a time.

        A source must not use the database. It runs in its own thread.

        Yields:
            One document with its full text.
        """
