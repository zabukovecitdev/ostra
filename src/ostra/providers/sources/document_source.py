from abc import ABC, abstractmethod
from collections.abc import Iterator

from ostra.models.document import Document


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
