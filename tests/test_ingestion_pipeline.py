import shutil
from pathlib import Path

import pytest

from ostra.models.document import Document
from ostra.models.document_record import DocumentRecord
from ostra.models.embedded_chunk import EmbeddedChunk
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.models.vector import Vector
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.providers.sources.file_system_source import FileSystemSource
from ostra.repositories.chunk_store import ChunkStore
from ostra.repositories.model_index import ModelIndex
from ostra.services.ingestion_pipeline import IngestionPipeline
from ostra.services.ingestion_runner import run_ingestion
from ostra.services.text_chunker import TextChunker
from ostra.utils.hashing import hash_text

DATA_DIRECTORY = Path(__file__).parent / "data"


@pytest.fixture
def index(store: ChunkStore) -> ModelIndex:
    """Provide the index of the mxbai model, because the fake provider returns 1024 values."""
    return store.model_index(MXBAI_EMBED_LARGE, create=True)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return a 1024-value vector that depends on the text length and count the calls."""

    def __init__(self):
        super().__init__(MXBAI_EMBED_LARGE)
        self.calls: list[str] = []

    def _embed(self, text: str) -> list[float]:
        """Record the text without its document prefix and return a vector for it."""
        self.calls.append(text.removeprefix(self._spec.document_prefix))
        return [float(len(text))] + [0.5] * 1023


def one_token_per_word(word: str) -> int:
    """Count each word as one token so that the budgets are easy to follow in the tests."""
    return 1


class FolderIngestion:
    """Run a pipeline on the files of one folder through the queue, as the application does."""

    def __init__(self, directory: Path, pipeline: IngestionPipeline, provider: FakeEmbeddingProvider):
        self._directory = directory
        self._pipeline = pipeline
        self.provider = provider

    def process_files(self) -> None:
        """Read the files of the folder and process them."""
        run_ingestion(FileSystemSource(directories=[self._directory]), self._pipeline)


def make_service(
    index: ModelIndex,
    directory: Path,
    reindex: bool = False,
    chunk_size_tokens: int = 510,
    chunk_overlap_tokens: int = 50,
    embedding_provider: FakeEmbeddingProvider | None = None,
) -> FolderIngestion:
    """Build an ingestion that reads one directory and stores results through the index."""
    provider = embedding_provider or FakeEmbeddingProvider()
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            count_tokens=one_token_per_word,
        ),
        embedding_provider=provider,
        reindex=reindex,
    )
    return FolderIngestion(directory, pipeline, provider)


def stored_chunks(index: ModelIndex, file_path: Path) -> set[str]:
    """Return the stored chunk digests of one file."""
    return index.stored_digests(str(file_path))


def test_process_files__new_file__embeds_one_chunk(index, tmp_path: Path):
    """Embed the words of a new file as one chunk when the file is shorter than the chunk size."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path)

    service.process_files()

    assert service.provider.calls == ["alpha beta"]


def test_process_files__new_file__saves_digest_of_each_chunk(index, tmp_path: Path):
    """Save the digest of each chunk of the file."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(index, tmp_path).process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha beta")}


def test_process_files__identical_chunks__embeds_the_chunk_once(index, tmp_path: Path):
    """Embed a repeated chunk only once within one file."""
    (tmp_path / "a.txt").write_text("same\nsame\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)

    service.process_files()

    assert service.provider.calls == ["same"]


def test_process_files__overlapping_chunks__embeds_each_chunk_with_overlap(index, tmp_path: Path):
    """Embed each chunk, and start each chunk with the words that overlap the previous chunk."""
    (tmp_path / "a.txt").write_text("a b c d e\n")
    service = make_service(index, tmp_path, chunk_size_tokens=3, chunk_overlap_tokens=1)

    service.process_files()

    assert service.provider.calls == ["a b c", "c d e"]


def test_process_files__second_run_file_unchanged__embeds_nothing(index, tmp_path: Path):
    """Embed no chunks on a second run when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path)
    service.process_files()
    service.provider.calls.clear()

    service.process_files()

    assert service.provider.calls == []


def test_process_files__run_stopped_after_first_chunk__resumes_at_next_chunk(index, tmp_path: Path):
    """Skip the chunks that a stopped run already saved and process the remaining chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    index.save_document(
        DocumentRecord(str(file_path), "filesystem", "a.txt"),
        [EmbeddedChunk(hash_text("alpha"), "alpha", Vector("mxbai-embed-large", (0.1,) * 1024))],
        current_digests={hash_text("alpha")},
    )

    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)

    service.process_files()

    assert service.provider.calls == ["beta", "gamma"]


def test_process_files__chunk_changed__embeds_only_the_changed_chunk(index, tmp_path: Path):
    """Embed only the chunk whose content changed since the last run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    service.provider.calls.clear()
    file_path.write_text("alpha\nbeta changed\n")

    service.process_files()

    assert service.provider.calls == ["changed"]


def test_process_files__word_inserted_above__embeds_only_the_new_chunk(index, tmp_path: Path):
    """Embed only the new chunk when a word is inserted above existing chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    service.provider.calls.clear()
    file_path.write_text("new\nalpha\nbeta\n")

    service.process_files()

    assert service.provider.calls == ["new"]


def test_process_files__file_got_shorter__deletes_digests_of_removed_chunks(index, tmp_path: Path):
    """Delete the stored digests of chunks that the file no longer has."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    file_path.write_text("alpha\n")

    service.process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha")}


def test_process_files__reindex_and_unchanged_file__embeds_every_chunk_again(index, tmp_path: Path):
    """Embed every chunk again on a reindex run even when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(index, tmp_path).process_files()
    service = make_service(index, tmp_path, reindex=True)

    service.process_files()

    assert service.provider.calls == ["alpha beta"]


def test_process_files__reindex_and_identical_chunks__embeds_the_chunk_once(index, tmp_path: Path):
    """Embed a repeated chunk only once on a reindex run."""
    (tmp_path / "a.txt").write_text("same\nsame\n")
    service = make_service(index, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0)

    service.process_files()

    assert service.provider.calls == ["same"]


def test_process_files__reindex_and_chunk_removed__deletes_its_digest(index, tmp_path: Path):
    """Delete the stored digest of a removed chunk on a reindex run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()
    file_path.write_text("alpha\n")

    make_service(index, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha")}


def test_process_files__long_text_file__chunks_are_sized_and_overlap(index, tmp_path: Path):
    """Split a long file into chunks of at most the chunk size, where each chunk overlaps the next."""
    file_path = tmp_path / "printing_press_history.txt"
    shutil.copy(DATA_DIRECTORY / "printing_press_history.txt", file_path)
    words = file_path.read_text().split()

    service = make_service(index, tmp_path, chunk_size_tokens=200, chunk_overlap_tokens=20)
    service.process_files()
    chunks = [text.split() for text in service.provider.calls]

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(previous[-20:] == current[:20] for previous, current in zip(chunks, chunks[1:], strict=False))
    assert chunks[-1][-10:] == words[-10:]
    assert len(stored_chunks(index, file_path)) == len(chunks)


def test_process_files__new_chunk__embeds_chunk_and_stores_vector(index, tmp_path: Path):
    """Embed each new chunk and store its vector with the chunk digest."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    provider = FakeEmbeddingProvider()

    make_service(index, tmp_path, embedding_provider=provider).process_files()

    assert provider.calls == ["alpha beta"]
    matches = index.search(Vector("mxbai-embed-large", tuple([10.0] + [0.5] * 1023)), limit=1)
    assert [(match.document_id, match.digest, match.text) for match in matches] == [
        (str(file_path), hash_text("alpha beta"), "alpha beta")
    ]
    assert matches[0].distance == pytest.approx(0.0, abs=1e-5)


def test_process_files__second_run_unchanged__does_not_embed_again(index, tmp_path: Path):
    """Skip the embedding of chunks that a previous run already stored."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(index, tmp_path).process_files()
    provider = FakeEmbeddingProvider()

    make_service(index, tmp_path, embedding_provider=provider).process_files()

    assert provider.calls == []


def test_process_files__reindex__embeds_every_chunk_again(index, tmp_path: Path):
    """Embed every chunk again on a reindex run."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(index, tmp_path).process_files()
    provider = FakeEmbeddingProvider()

    make_service(index, tmp_path, reindex=True, embedding_provider=provider).process_files()

    assert provider.calls == ["alpha beta"]


def test_process_documents__document_from_any_source__stores_chunks_under_its_id_with_source_and_title(index):
    """Store the chunks of a document that did not come from a file, with its source and title."""
    document = Document(source="notes", document_id="note://1", version="v1", title="One", text="alpha beta")
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(chunk_size_tokens=510, chunk_overlap_tokens=50, count_tokens=one_token_per_word),
        embedding_provider=FakeEmbeddingProvider(),
        reindex=False,
    )

    pipeline.process_documents([document])

    matches = index.search(Vector("mxbai-embed-large", tuple([10.0] + [0.5] * 1023)), limit=5)
    assert [(m.document_id, m.source, m.title, m.digest) for m in matches] == [
        ("note://1", "notes", "One", hash_text("alpha beta"))
    ]


def test_process_documents__callback_set__receives_each_document_id(index):
    """Call the callback with the ID of each document after it is processed."""
    seen: list[str] = []
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(chunk_size_tokens=510, chunk_overlap_tokens=50, count_tokens=one_token_per_word),
        embedding_provider=FakeEmbeddingProvider(),
        reindex=False,
        on_document_processed=seen.append,
    )
    documents = [Document("notes", f"note://{number}", "v", "t", "text") for number in (1, 2)]

    pipeline.process_documents(documents)

    assert seen == ["note://1", "note://2"]


def test_process_documents__embedding_fails_midway__keeps_finished_chunks_and_resumes(index):
    """Keep the chunks that were embedded before an error, and embed only the rest on the next run."""

    class FailingProvider(FakeEmbeddingProvider):
        def _embed(self, text: str) -> list[float]:
            if text == "gamma":
                raise RuntimeError("Ollama is down")
            return super()._embed(text)

    chunker = TextChunker(chunk_size_tokens=1, chunk_overlap_tokens=0, count_tokens=one_token_per_word)
    document = Document("notes", "note://1", "v", "t", "alpha beta gamma")
    failing = IngestionPipeline(index, chunker, FailingProvider(), reindex=False)

    with pytest.raises(RuntimeError, match="Ollama is down"):
        failing.process_documents([document])
    provider = FakeEmbeddingProvider()
    IngestionPipeline(index, chunker, provider, reindex=False).process_documents([document])

    assert provider.calls == ["gamma"]


def test_process_documents__callbacks__started_before_each_document_and_processed_after(index, tmp_path: Path):
    """Call the started callback before each document is embedded and the processed callback after it."""
    events: list[str] = []
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(chunk_size_tokens=510, chunk_overlap_tokens=0, count_tokens=one_token_per_word),
        embedding_provider=FakeEmbeddingProvider(),
        reindex=False,
        on_document_processed=lambda document_id: events.append(f"processed {document_id}"),
        on_document_started=lambda document_id: events.append(f"started {document_id}"),
    )
    documents = [
        Document(source="filesystem", document_id=f"doc{number}", version="v", title="t", text="words here")
        for number in (1, 2)
    ]

    pipeline.process_documents(documents)

    assert events == ["started doc1", "processed doc1", "started doc2", "processed doc2"]
