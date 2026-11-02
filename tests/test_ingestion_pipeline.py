import shutil
from pathlib import Path

import pytest

from ostra.embeddings.base import EmbeddingProvider
from ostra.embeddings.vector import Vector
from ostra.hashing import hash_text
from ostra.models.registry import MXBAI_EMBED_LARGE
from ostra.pipeline.ingestion_pipeline import IngestionPipeline
from ostra.pipeline.ingestion_runner import run_ingestion
from ostra.services.text_chunker import TextChunker
from ostra.sources.base import Document
from ostra.sources.file_system_source import FileSystemSource
from ostra.store.chunk_store import ChunkStore
from ostra.store.document_record import DocumentRecord
from ostra.store.embedded_chunk import EmbeddedChunk
from ostra.store.model_index import ModelIndex

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
        """Record the text and return a three-value vector for it."""
        self.calls.append(text)
        return [float(len(text))] + [0.5] * 1023


def one_token_per_word(word: str) -> int:
    """Count each word as one token so that the budgets are easy to follow in the tests."""
    return 1


class FolderIngestion:
    """Run a pipeline on the files of one folder through the queue, as the application does."""

    def __init__(self, directory: Path, pipeline: IngestionPipeline):
        self._directory = directory
        self._pipeline = pipeline

    def process_files(self) -> None:
        """Read the files of the folder and process them."""
        run_ingestion(FileSystemSource(directories=[self._directory]), self._pipeline)


def make_service(
    index: ModelIndex,
    directory: Path,
    reindex: bool = False,
    chunk_size_tokens: int = 510,
    chunk_overlap_tokens: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> FolderIngestion:
    """Build an ingestion that reads one directory and stores results through the index."""
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            count_tokens=one_token_per_word,
        ),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        reindex=reindex,
    )
    return FolderIngestion(directory, pipeline)


def stored_chunks(index: ModelIndex, file_path: Path) -> set[str]:
    """Return the stored chunk digests of one file."""
    return index.stored_digests(str(file_path))


def test_process_files__new_file__prints_each_chunk(index, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Print the words of a new file as one chunk when the file is shorter than the chunk size."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")

    make_service(index, tmp_path).process_files()

    assert capsys.readouterr().out == "alpha beta\n"


def test_process_files__new_file__saves_digest_of_each_chunk(index, tmp_path: Path):
    """Save the digest of each chunk of the file."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(index, tmp_path).process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha beta")}


def test_process_files__identical_chunks__prints_the_chunk_once(index, tmp_path: Path, capsys):
    """Print a repeated chunk only once within one file."""
    (tmp_path / "a.txt").write_text("same\nsame\n")

    make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "same\n"


def test_process_files__overlapping_chunks__prints_each_chunk_with_overlap(index, tmp_path: Path, capsys):
    """Print each chunk, and start each chunk with the words that overlap the previous chunk."""
    (tmp_path / "a.txt").write_text("a b c d e\n")

    make_service(index, tmp_path, chunk_size_tokens=3, chunk_overlap_tokens=1).process_files()

    assert capsys.readouterr().out == "a b c\nc d e\n"


def test_process_files__second_run_file_unchanged__prints_nothing(index, tmp_path: Path, capsys):
    """Print no chunks on a second run when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path)
    service.process_files()
    capsys.readouterr()

    service.process_files()

    assert capsys.readouterr().out == ""


def test_process_files__run_stopped_after_first_chunk__resumes_at_next_chunk(index, tmp_path: Path, capsys):
    """Skip the chunks that a stopped run already saved and process the remaining chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    index.save_document(
        DocumentRecord(str(file_path), "filesystem", "a.txt"),
        [EmbeddedChunk(hash_text("alpha"), "alpha", Vector("mxbai-embed-large", (0.1,) * 1024))],
        current_digests={hash_text("alpha")},
    )

    make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "beta\ngamma\n"


def test_process_files__chunk_changed__prints_only_the_changed_chunk(index, tmp_path: Path, capsys):
    """Print only the chunk whose content changed since the last run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("alpha\nbeta changed\n")

    service.process_files()

    assert capsys.readouterr().out == "changed\n"


def test_process_files__word_inserted_above__prints_only_the_new_chunk(index, tmp_path: Path, capsys):
    """Print only the new chunk when a word is inserted above existing chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("new\nalpha\nbeta\n")

    service.process_files()

    assert capsys.readouterr().out == "new\n"


def test_process_files__file_got_shorter__deletes_digests_of_removed_chunks(index, tmp_path: Path):
    """Delete the stored digests of chunks that the file no longer has."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    service = make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    file_path.write_text("alpha\n")

    service.process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha")}


def test_process_files__reindex_and_unchanged_file__prints_every_chunk_again(index, tmp_path: Path, capsys):
    """Print every chunk again on a reindex run even when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(index, tmp_path).process_files()
    capsys.readouterr()

    make_service(index, tmp_path, reindex=True).process_files()

    assert capsys.readouterr().out == "alpha beta\n"


def test_process_files__reindex_and_identical_chunks__prints_the_chunk_once(index, tmp_path: Path, capsys):
    """Print a repeated chunk only once on a reindex run."""
    (tmp_path / "a.txt").write_text("same\nsame\n")

    make_service(index, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "same\n"


def test_process_files__reindex_and_chunk_removed__deletes_its_digest(index, tmp_path: Path):
    """Delete the stored digest of a removed chunk on a reindex run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    make_service(index, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()
    file_path.write_text("alpha\n")

    make_service(index, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert stored_chunks(index, file_path) == {hash_text("alpha")}


def test_process_files__long_text_file__chunks_are_sized_and_overlap(index, tmp_path: Path, capsys):
    """Split a long file into chunks of at most the chunk size, where each chunk overlaps the next."""
    file_path = tmp_path / "printing_press_history.txt"
    shutil.copy(DATA_DIRECTORY / "printing_press_history.txt", file_path)
    words = file_path.read_text().split()

    make_service(index, tmp_path, chunk_size_tokens=200, chunk_overlap_tokens=20).process_files()
    chunks = [line.split() for line in capsys.readouterr().out.splitlines()]

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


def test_process_documents__started_callback_set__runs_before_processed_callback(index):
    """Call the started callback before the processed callback of each document."""
    events: list[str] = []
    pipeline = IngestionPipeline(
        model_index=index,
        chunker=TextChunker(chunk_size_tokens=510, chunk_overlap_tokens=50, count_tokens=one_token_per_word),
        embedding_provider=FakeEmbeddingProvider(),
        reindex=False,
        on_document_processed=lambda document_id: events.append(f"done {document_id}"),
        on_document_started=lambda document_id: events.append(f"start {document_id}"),
    )
    documents = [Document("notes", f"note://{number}", "v", "t", "text") for number in (1, 2)]

    pipeline.process_documents(documents)

    assert events == ["start note://1", "done note://1", "start note://2", "done note://2"]


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
