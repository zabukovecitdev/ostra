import shutil
from pathlib import Path

import pytest

from ostra.connectors.base import VectorDatabaseConnector
from ostra.embeddings.base import EmbeddingProvider
from ostra.hashing import hash_file, hash_text
from ostra.repositories.file_chunk_repository import FileChunkRepository
from ostra.repositories.file_repository import FileRepository
from ostra.services.file_ingestion_service import FileIngestionService
from ostra.services.file_service import FileService
from ostra.services.text_chunker import TextChunker

DATA_DIRECTORY = Path(__file__).parent / "data"


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return a 1024-value vector that depends on the text length and count the calls."""

    def __init__(self):
        self.calls: list[str] = []

    def get_embedding(self, text: str) -> list[float]:
        """Record the text and return a three-value vector for it."""
        self.calls.append(text)
        return [float(len(text))] + [0.5] * 1023


def one_token_per_word(word: str) -> int:
    """Count each word as one token so that the budgets are easy to follow in the tests."""
    return 1


def make_service(
    connector: VectorDatabaseConnector,
    directory: Path,
    reindex: bool = False,
    chunk_size_tokens: int = 510,
    chunk_overlap_tokens: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> FileIngestionService:
    """Build an ingestion service that reads one directory and stores results through the connector."""
    return FileIngestionService(
        file_service=FileService(directories=[directory]),
        file_repository=FileRepository(connector=connector),
        file_chunk_repository=FileChunkRepository(connector=connector),
        chunker=TextChunker(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            count_tokens=one_token_per_word,
        ),
        embedding_provider=embedding_provider or FakeEmbeddingProvider(),
        reindex=reindex,
    )


def stored_chunks(connector: VectorDatabaseConnector, file_path: Path) -> set[str]:
    """Return the stored chunk digests of one file."""
    return connector.get_chunk_digests(str(file_path))


def test_process_files__new_file__prints_each_chunk(connector, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Print the words of a new file as one chunk when the file is shorter than the chunk size."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()

    assert capsys.readouterr().out == "alpha beta\n"


def test_process_files__new_file__saves_file_digest(connector, tmp_path: Path):
    """Save the digest of the whole file."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()

    assert connector.get_file_digest(str(file_path)) == hash_file(file_path)


def test_process_files__new_file__saves_digest_of_each_chunk(connector, tmp_path: Path):
    """Save the digest of each chunk of the file."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")

    make_service(connector, tmp_path).process_files()

    assert stored_chunks(connector, file_path) == {hash_text("alpha beta")}


def test_process_files__identical_chunks__prints_the_chunk_once(connector, tmp_path: Path, capsys):
    """Print a repeated chunk only once within one file."""
    (tmp_path / "a.txt").write_text("same\nsame\n")

    make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "same\n"


def test_process_files__overlapping_chunks__prints_each_chunk_with_overlap(connector, tmp_path: Path, capsys):
    """Print each chunk, and start each chunk with the words that overlap the previous chunk."""
    (tmp_path / "a.txt").write_text("a b c d e\n")

    make_service(connector, tmp_path, chunk_size_tokens=3, chunk_overlap_tokens=1).process_files()

    assert capsys.readouterr().out == "a b c\nc d e\n"


def test_process_files__second_run_file_unchanged__prints_nothing(connector, tmp_path: Path, capsys):
    """Print no chunks on a second run when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    capsys.readouterr()

    service.process_files()

    assert capsys.readouterr().out == ""


def test_process_files__run_stopped_after_first_chunk__resumes_at_next_chunk(connector, tmp_path: Path, capsys):
    """Skip the chunks that a stopped run already saved and process the remaining chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    FileChunkRepository(connector=connector).save_chunk(str(file_path), hash_text("alpha"), "alpha", [0.1] * 1024)

    make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "beta\ngamma\n"


def test_process_files__chunk_changed__prints_only_the_changed_chunk(connector, tmp_path: Path, capsys):
    """Print only the chunk whose content changed since the last run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("alpha\nbeta changed\n")

    service.process_files()

    assert capsys.readouterr().out == "changed\n"


def test_process_files__word_inserted_above__prints_only_the_new_chunk(connector, tmp_path: Path, capsys):
    """Print only the new chunk when a word is inserted above existing chunks."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    service = make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    capsys.readouterr()
    file_path.write_text("new\nalpha\nbeta\n")

    service.process_files()

    assert capsys.readouterr().out == "new\n"


def test_process_files__file_got_shorter__deletes_digests_of_removed_chunks(connector, tmp_path: Path):
    """Delete the stored digests of chunks that the file no longer has."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    service = make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0)
    service.process_files()
    file_path.write_text("alpha\n")

    service.process_files()

    assert stored_chunks(connector, file_path) == {hash_text("alpha")}


def test_process_files__file_changed__updates_stored_file_digest(connector, tmp_path: Path):
    """Replace the stored file digest when the file content changed."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\n")
    service = make_service(connector, tmp_path)
    service.process_files()
    file_path.write_text("alpha\nbeta\n")

    service.process_files()

    assert connector.get_file_digest(str(file_path)) == hash_file(file_path)


def test_process_files__reindex_and_unchanged_file__prints_every_chunk_again(connector, tmp_path: Path, capsys):
    """Print every chunk again on a reindex run even when the file did not change."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(connector, tmp_path).process_files()
    capsys.readouterr()

    make_service(connector, tmp_path, reindex=True).process_files()

    assert capsys.readouterr().out == "alpha beta\n"


def test_process_files__reindex_and_identical_chunks__prints_the_chunk_once(connector, tmp_path: Path, capsys):
    """Print a repeated chunk only once on a reindex run."""
    (tmp_path / "a.txt").write_text("same\nsame\n")

    make_service(connector, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert capsys.readouterr().out == "same\n"


def test_process_files__reindex_and_chunk_removed__deletes_its_digest(connector, tmp_path: Path):
    """Delete the stored digest of a removed chunk on a reindex run."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    make_service(connector, tmp_path, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()
    file_path.write_text("alpha\n")

    make_service(connector, tmp_path, reindex=True, chunk_size_tokens=1, chunk_overlap_tokens=0).process_files()

    assert stored_chunks(connector, file_path) == {hash_text("alpha")}


def test_process_files__long_text_file__chunks_are_sized_and_overlap(connector, tmp_path: Path, capsys):
    """Split a long file into chunks of at most the chunk size, where each chunk overlaps the next."""
    file_path = tmp_path / "printing_press_history.txt"
    shutil.copy(DATA_DIRECTORY / "printing_press_history.txt", file_path)
    words = file_path.read_text().split()

    make_service(connector, tmp_path, chunk_size_tokens=200, chunk_overlap_tokens=20).process_files()
    chunks = [line.split() for line in capsys.readouterr().out.splitlines()]

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert all(previous[-20:] == current[:20] for previous, current in zip(chunks, chunks[1:], strict=False))
    assert chunks[-1][-10:] == words[-10:]
    assert len(stored_chunks(connector, file_path)) == len(chunks)


def test_process_files__new_chunk__embeds_chunk_and_stores_vector(connector, tmp_path: Path):
    """Embed each new chunk and store its vector with the chunk digest."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("alpha\nbeta\n")
    provider = FakeEmbeddingProvider()

    make_service(connector, tmp_path, embedding_provider=provider).process_files()

    assert provider.calls == ["alpha beta"]
    assert connector.get_chunk_embedding(str(file_path), hash_text("alpha beta")) == [10.0] + [0.5] * 1023


def test_process_files__second_run_unchanged__does_not_embed_again(connector, tmp_path: Path):
    """Skip the embedding of chunks that a previous run already stored."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(connector, tmp_path).process_files()
    provider = FakeEmbeddingProvider()

    make_service(connector, tmp_path, embedding_provider=provider).process_files()

    assert provider.calls == []


def test_process_files__reindex__embeds_every_chunk_again(connector, tmp_path: Path):
    """Embed every chunk again on a reindex run."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    make_service(connector, tmp_path).process_files()
    provider = FakeEmbeddingProvider()

    make_service(connector, tmp_path, reindex=True, embedding_provider=provider).process_files()

    assert provider.calls == ["alpha beta"]
