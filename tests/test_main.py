import logging
from pathlib import Path

import pytest

from personal_rag import main
from personal_rag.config import AppConfig
from personal_rag.embeddings.base import EmbeddingProvider
from personal_rag.models.base import EmbeddingModel
from personal_rag.tokenization.base import Tokenizer


class FakeTokenizer(Tokenizer):
    """Count each word as one token."""

    def count_tokens(self, text: str) -> int:
        """Return the number of words in the text."""
        return len(text.split())


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return the same 1024-value vector for every text."""

    def get_embedding(self, text: str) -> list[float]:
        """Return a fixed vector."""
        return [0.5] * 1024


class FakeEmbeddingModel(EmbeddingModel):
    """Stand in for the Ollama model so that the tests do not call a server or download a tokenizer."""

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Return the fake provider."""
        return FakeEmbeddingProvider()

    def get_tokenizer(self) -> Tokenizer:
        """Return the fake tokenizer."""
        return FakeTokenizer()


@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch: pytest.MonkeyPatch):
    """Replace the embedding model that main creates with the fake model for every test."""
    monkeypatch.setattr(main, "MxbaiEmbeddingModel", lambda settings: FakeEmbeddingModel())


def test_main__database_ready__prints_database_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the database path from the config after the database is ready."""
    database_path = tmp_path / "personal_rag.db"
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=database_path, directories=[]))

    main.main([])

    assert capsys.readouterr().out == f"Database ready at {database_path}\n"
    assert database_path.is_file()


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[])
    )

    main.main([])

    assert basic_config_calls == [{"level": logging.INFO}]


def test_main__directories_have_files__prints_each_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print each line of the files in the configured directories without the line ending."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\nbeta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )

    main.main([])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha beta"]


def test_main__reindex_flag__prints_lines_of_unchanged_file_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the lines of an unchanged file again when the reindex flag is given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )
    main.main([])
    capsys.readouterr()

    main.main(["--reindex"])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha"]


def test_main__no_reindex_flag__skips_lines_of_unchanged_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print no lines of an unchanged file when the reindex flag is not given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )
    main.main([])
    capsys.readouterr()

    main.main([])

    assert capsys.readouterr().out.splitlines()[1:] == []


def test_main__search_flag__prints_stored_chunk_with_its_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the stored chunk and its file path when the search flag is given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha beta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "personal_rag.db", directories=[notes])
    )
    main.main([])
    capsys.readouterr()

    main.main(["--search", "alpha", "--limit", "1"])

    output = capsys.readouterr().out
    assert f"file={notes / 'a.txt'}" in output
    assert "alpha beta" in output


def test_main__search_flag_without_limit__uses_default_limit():
    """Use five as the default number of chunks for a search."""
    arguments = main._parse_arguments(["--search", "alpha"])

    assert (arguments.search, arguments.limit) == ("alpha", 5)
