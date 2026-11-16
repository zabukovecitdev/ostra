from pathlib import Path

from conftest import make_fake_model

from ostra.models.document import Document
from ostra.providers.sources.document_source import DocumentSource
from ostra.repositories.model_index import ChunkWriteIndex


class ListSource(DocumentSource):
    """Offer fixed documents, with no file system."""

    def __init__(self, *texts: str):
        self._texts = texts

    def stream_documents(self):
        """Yield one document for each text."""
        for number, text in enumerate(self._texts):
            yield Document(source="list", document_id=f"doc{number}", title="t", text=text)


class RecordingIndex(ChunkWriteIndex):
    """Remember the IDs of the saved documents."""

    def __init__(self):
        self.saved: list[str] = []

    def stored_digests(self, document_id: str) -> set[str]:
        """Return no digests."""
        return set()

    def save_document(self, document, new_chunks, current_digests) -> None:
        """Remember the document ID and drain the chunks."""
        list(new_chunks)
        self.saved.append(document.document_id)


def test_run_indexing__two_custom_sources__indexes_the_documents_of_both(tmp_path: Path):
    """Process every document of every injected source, in the order of the sources."""
    index = RecordingIndex()

    processed = run_indexing(
        config=AppConfig(database_path=tmp_path / "db"),
        model_index=index,
        embedding_model=make_fake_model(),
        sources=[ListSource("one two"), ListSource("three")],
    )

    assert index.saved == ["doc0", "doc0"]
    assert processed == 2
