from collections.abc import Iterator

from ostra.cli.progress import ProgressReporter
from ostra.models.document import Document
from ostra.providers.sources.document_source import DocumentSource


class CountedSource(DocumentSource):
    """Know how many documents it offers."""

    def count_documents(self) -> int | None:
        """Return three."""
        return 3

    def stream_documents(self) -> Iterator[Document]:
        """Offer no documents."""
        return iter(())


class UncountedSource(DocumentSource):
    """Not know how many documents it offers."""

    def stream_documents(self) -> Iterator[Document]:
        """Offer no documents."""
        return iter(())


def test_progress_reporter__all_sources_count__total_is_the_sum():
    """Show a total when every source knows its count."""
    with ProgressReporter([CountedSource(), CountedSource()], disable=False) as progress:
        assert progress._bar.total == 6


def test_progress_reporter__a_source_cannot_count__total_is_unknown():
    """Show no total when one source cannot count."""
    with ProgressReporter([CountedSource(), UncountedSource()], disable=False) as progress:
        assert progress._bar.total is None


def test_progress_reporter__documents_processed__advances_the_bar():
    """Advance by one for each processed document."""
    with ProgressReporter([CountedSource()], disable=False) as progress:
        progress.document_started("a.txt")
        progress.document_processed("a.txt")

        assert progress._bar.n == 1
