import queue
import threading
from collections.abc import Iterator

from ostra.pipeline.ingestion_pipeline import IngestionPipeline
from ostra.sources.base import Document, DocumentSource

DEFAULT_QUEUE_SIZE = 8
PUT_TIMEOUT_SECONDS = 0.1


def run_ingestion(source: DocumentSource, pipeline: IngestionPipeline, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
    """Read the documents of a source in a thread and process them in the calling thread through a queue.

    The queue is bounded, so the source waits when the pipeline is slow and does not fill the memory.
    Only the calling thread uses the database.

    Args:
        source: The source that offers the documents.
        pipeline: The pipeline that processes them.
        queue_size: The largest number of documents that wait in the queue.

    Raises:
        Exception: The error of the source, after the pipeline processed the documents that came before it.
            The error of the pipeline is raised at once, and the source stops.
    """
    documents: queue.Queue[Document | None] = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    failures: list[Exception] = []
    producer = threading.Thread(target=_produce, args=(source, documents, stop, failures), daemon=True)
    producer.start()
    try:
        pipeline.process_documents(_drain(documents))
    finally:
        stop.set()
        producer.join()

    if failures:
        raise failures[0]


def _produce(
    source: DocumentSource, documents: queue.Queue[Document | None], stop: threading.Event, failures: list[Exception]
) -> None:
    """Put each document of the source in the queue, then put the end marker.

    Args:
        source: The source that offers the documents.
        documents: The queue to fill.
        stop: Set when the consumer has stopped, so that the producer must give up.
        failures: List that receives the error of the source, when there is one.
    """
    try:
        for document in source.stream_documents():
            if not _put(documents, document, stop):
                return
    except Exception as error:
        failures.append(error)
    finally:
        _put(documents, None, stop)


def _put(documents: queue.Queue[Document | None], item: Document | None, stop: threading.Event) -> bool:
    """Put an item in the queue and wait while the queue is full, unless the consumer has stopped.

    Args:
        documents: The queue to fill.
        item: A document, or None for the end marker.
        stop: Set when the consumer has stopped.

    Returns:
        True when the item is in the queue, and False when the consumer stopped first.
    """
    while not stop.is_set():
        try:
            documents.put(item, timeout=PUT_TIMEOUT_SECONDS)
        except queue.Full:
            continue

        return True

    return False


def _drain(documents: queue.Queue[Document | None]) -> Iterator[Document]:
    """Yield the documents from the queue until the end marker arrives.

    Args:
        documents: The queue to read.

    Yields:
        One document at a time, in the order of the queue.
    """
    while True:
        document = documents.get()
        if document is None:
            return

        yield document
