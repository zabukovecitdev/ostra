from types import TracebackType
from typing import Self

from tqdm import tqdm

from ostra.providers.sources.document_source import DocumentSource


class ProgressReporter:
    """Show the indexing progress on the terminal, with the count of documents and the current document.

    tqdm hides the bar when the output is not a terminal. The bar writes to stderr.
    """

    def __init__(self, sources: list[DocumentSource], disable: bool | None = None):
        """Start the bar with the total count of documents, when every source knows it.

        Args:
            sources: The sources that will be indexed.
            disable: Hide the bar when True, show it when False. None hides it when the output is not a terminal.
        """
        self._bar = tqdm(total=_total_documents(sources), unit="doc", desc="Indexing", disable=disable)

    def document_started(self, document_id: str) -> None:
        """Show the ID of the document that is being processed after the bar.

        Args:
            document_id: The ID of the document.
        """
        self._bar.set_postfix_str(document_id, refresh=True)

    def document_processed(self, document_id: str) -> None:
        """Count one processed document on the bar.

        Args:
            document_id: The ID of the processed document. Not shown, because the next document replaces it.
        """
        self._bar.update(1)

    def close(self) -> None:
        """Close the bar."""
        self._bar.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _total_documents(sources: list[DocumentSource]) -> int | None:
    """Add up the document counts of the sources, when every source knows its count.

    Args:
        sources: The sources that will be indexed.

    Returns:
        The total, or None when a source cannot know its count before it reads the documents.
    """
    counts = [source.count_documents() for source in sources]
    known = [count for count in counts if count is not None]
    return sum(known) if len(known) == len(counts) else None
