from collections.abc import Callable, Iterable, Iterator

from ostra.embeddings.base import EmbeddingProvider
from ostra.hashing import hash_text
from ostra.services.text_chunker import TextChunker
from ostra.sources.base import Document
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk
from ostra.store.model_index import ModelIndex


class IngestionPipeline:
    """Chunk, embed and store documents, and skip the chunks that a previous run already processed.

    The pipeline knows nothing about the origin of a document.
    """

    def __init__(
        self,
        model_index: ModelIndex,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        reindex: bool,
        on_document_processed: Callable[[str], None] | None = None,
        on_document_started: Callable[[str], None] | None = None,
    ):
        self._model_index = model_index
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._reindex = reindex
        self._on_document_processed = on_document_processed
        self._on_document_started = on_document_started

    def process_documents(self, documents: Iterable[Document]) -> None:
        """Process each document, one at a time, in the order that the iterable gives them.

        Call the on_document_started callback with the document ID before each document, when one is set.
        Call the on_document_processed callback with the document ID after each document, when one is set.

        Args:
            documents: The documents to process.
        """
        for document in documents:
            if self._on_document_started is not None:
                self._on_document_started(document.document_id)
            self._process_document(document)
            if self._on_document_processed is not None:
                self._on_document_processed(document.document_id)

    def _process_document(self, document: Document) -> None:
        """Embed and store each chunk whose digest is not stored, then delete the chunks that are gone.

        A chunk is skipped when its digest is stored for the document, whatever its position.
        When reindex is set, no stored digest is skipped, so every chunk is embedded again.
        A repeated chunk is embedded once. Print each embedded chunk.
        The index saves each chunk when it is embedded, so an interrupted run continues at the next chunk.

        Args:
            document: The document to process.
        """
        stored_digests = set() if self._reindex else self._model_index.stored_digests(document.document_id)
        chunks_by_digest: dict[str, str] = {}
        for chunk in self._chunker.stream_chunks(iter(document.text.split())):
            chunks_by_digest.setdefault(hash_text(chunk), chunk)

        new_digests = [digest for digest in chunks_by_digest if digest not in stored_digests]
        self._model_index.save_document(
            DocumentRecord(document_id=document.document_id, source=document.source, title=document.title),
            self._embed(chunks_by_digest, new_digests),
            current_digests=chunks_by_digest.keys(),
        )

    def _embed(self, chunks_by_digest: dict[str, str], digests: list[str]) -> Iterator[EmbeddedChunk]:
        """Embed the chunks one at a time, in order, and print each one.

        Args:
            chunks_by_digest: The text of each chunk by its digest.
            digests: The digests of the chunks to embed.

        Yields:
            One embedded chunk at a time.
        """
        for digest in digests:
            text = chunks_by_digest[digest]
            yield EmbeddedChunk(digest=digest, text=text, embedding=self._embedding_provider.embed_document(text))
            print(text)
