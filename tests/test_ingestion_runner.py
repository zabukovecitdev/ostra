import threading
from collections.abc import Iterable, Iterator

import pytest

from ostra.pipeline.ingestion_runner import run_ingestion
from ostra.sources.base import Document, DocumentSource


def make_document(number: int) -> Document:
    return Document("test", f"doc://{number}", "v", f"Document {number}", "text")


class ListSource(DocumentSource):
    """Offer a fixed list of documents, then optionally raise an error."""

    def __init__(self, documents: Iterable[Document], error: Exception | None = None):
        self._documents = list(documents)
        self._error = error
        self.yielded = 0

    def stream_documents(self) -> Iterator[Document]:
        """Yield the documents, count them, then raise the error when one is set."""
        for document in self._documents:
            self.yielded += 1
            yield document

        if self._error is not None:
            raise self._error


class RecordingPipeline:
    """Record the documents and the thread that processes them."""

    def __init__(self, fail_after: int | None = None):
        self.processed: list[str] = []
        self.threads: set[int] = set()
        self._fail_after = fail_after

    def process_documents(self, documents: Iterable[Document]) -> None:
        """Process each document, and raise an error after the given number of documents."""
        for document in documents:
            self.threads.add(threading.get_ident())
            self.processed.append(document.document_id)
            if self._fail_after is not None and len(self.processed) >= self._fail_after:
                raise RuntimeError("pipeline failed")


def test_run_ingestion__several_documents__processes_all_in_order():
    """Pass every document from the source to the pipeline in the order of the source."""
    pipeline = RecordingPipeline()

    run_ingestion(ListSource(make_document(number) for number in range(20)), pipeline, queue_size=2)

    assert pipeline.processed == [f"doc://{number}" for number in range(20)]


def test_run_ingestion__called__processes_in_the_calling_thread():
    """Process the documents in the calling thread, so that only this thread uses the database."""
    pipeline = RecordingPipeline()

    run_ingestion(ListSource([make_document(1)]), pipeline)

    assert pipeline.threads == {threading.get_ident()}


def test_run_ingestion__source_without_documents__processes_nothing():
    """Finish without an error when the source has no documents."""
    pipeline = RecordingPipeline()

    run_ingestion(ListSource([]), pipeline)

    assert pipeline.processed == []


def test_run_ingestion__source_fails__processes_earlier_documents_then_raises():
    """Process the documents that came before the failure, then raise the error of the source."""
    pipeline = RecordingPipeline()
    source = ListSource([make_document(1), make_document(2)], error=OSError("disk gone"))

    with pytest.raises(OSError, match="disk gone"):
        run_ingestion(source, pipeline)

    assert pipeline.processed == ["doc://1", "doc://2"]


def test_run_ingestion__pipeline_fails_with_full_queue__raises_and_stops_the_source():
    """Raise the pipeline error and stop the source, even when the source waits on a full queue."""
    pipeline = RecordingPipeline(fail_after=1)
    source = ListSource(make_document(number) for number in range(1000))

    with pytest.raises(RuntimeError, match="pipeline failed"):
        run_ingestion(source, pipeline, queue_size=1)

    assert source.yielded < 1000
