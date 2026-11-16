import json
import logging
from pathlib import Path

import pytest

from ostra.cli import main
from ostra.cli.parser import build_parser
from ostra.cli.search_output import MAX_SHORT_TEXT_CHARACTERS
from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.models.embedding_model_registry import MXBAI_EMBED_LARGE
from ostra.providers.embeddings.embedding_provider import EmbeddingProvider
from ostra.providers.tokenizers.tokenizer import Tokenizer
from ostra.services import index_session
from ostra.services.embedding.embedding_model import EmbeddingModel


class FakeTokenizer(Tokenizer):
    """Count each word as one token."""

    def count_tokens(self, text: str) -> int:
        """Return the number of words in the text."""
        return len(text.split())


class FakeEmbeddingProvider(EmbeddingProvider):
    """Return the same 1024-value vector for every text."""

    def _embed(self, text: str) -> list[float]:
        """Return a fixed vector."""
        return [0.5] * 1024


def make_fake_model() -> EmbeddingModel:
    """Stand in for the Ollama model so that the tests do not call a server or download a tokenizer."""
    return EmbeddingModel(
        spec=MXBAI_EMBED_LARGE, provider=FakeEmbeddingProvider(MXBAI_EMBED_LARGE), tokenizer=FakeTokenizer()
    )


@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch: pytest.MonkeyPatch):
    """Replace the embedding model that main creates with the fake model for every test."""
    monkeypatch.setattr(index_session, "build_embedding_model", lambda config, model_id=None: make_fake_model())


def test_main__database_ready__prints_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print that the database is ready and create the database file."""
    database_path = tmp_path / "ostra.db"
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=database_path, directories=[]))

    main.main(["index"])

    assert capsys.readouterr().out == f"Database ready.\n{NO_DIRECTORIES_MESSAGE}\n"
    assert database_path.is_file()


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[]))

    main.main(["index"])

    assert basic_config_calls == [{"level": logging.INFO}]


def test_main__directories_have_files__prints_no_chunk_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print no text of the indexed files, because the progress bar shows the progress instead."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\nbeta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )

    main.main(["index"])

    assert "alpha beta" not in capsys.readouterr().out


def test_main__search_flag__prints_stored_chunk_with_its_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the stored chunk and its file path when the search flag is given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha beta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )
    main.main(["index"])
    capsys.readouterr()

    main.main(["search", "alpha", "--limit", "1"])

    output = capsys.readouterr().out
    assert f"file={notes / 'a.txt'}" in output
    assert "alpha beta" in output


def test_main__search_flag_without_limit__uses_default_limit():
    """Use five as the default number of chunks for a search."""
    arguments = build_parser().parse_args(["search", "alpha"])

    assert (arguments.query, arguments.limit) == ("alpha", 5)


def test_main__directory_flag__processes_given_folder_instead_of_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Index the folder from the directory flag and skip the configured folders."""
    configured = tmp_path / "configured"
    given = tmp_path / "given"
    indexed_directories = []
    monkeypatch.setattr(
        main,
        "load_config",
        lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[configured]),
    )
    monkeypatch.setattr(
        commands, "run_indexing", lambda **kwargs: indexed_directories.append(kwargs["sources"][0].directories)
    )

    main.main(["index", "--directory", str(given)])

    assert indexed_directories == [[given]]


def test_main__directory_flag_repeated__processes_every_folder(tmp_path: Path):
    """Keep every folder when the directory flag is repeated."""
    arguments = build_parser().parse_args(["index", "--directory", "/data/one", "--directory", "~/two"])

    assert arguments.directory == [Path("/data/one"), Path("~/two")]


def test_main__no_command__prints_help_and_does_not_load_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Print the help and skip the database when no command is given."""

    def fail_if_called():
        raise AssertionError("load_config must not run without a command")

    monkeypatch.setattr(main, "load_config", fail_if_called)

    main.main([])

    assert "COMMAND" in capsys.readouterr().out


def _index_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> Path:
    """Index one text file into a SQLite database and return the path of the file."""
    notes = tmp_path / "notes"
    notes.mkdir()
    file_path = notes / "a.txt"
    file_path.write_text(text)
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )
    main.main(["index"])
    return file_path


def test_main__search__prints_only_results_without_database_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the similarity, the file and the text of each result, without the database status line."""
    file_path = _index_text(monkeypatch, tmp_path, "alpha beta\n")
    capsys.readouterr()

    main.main(["search", "alpha"])

    assert capsys.readouterr().out == f"match 100%  file={file_path}\nalpha beta\n\n"


def test_main__search_no_stored_chunks__prints_no_matching_chunks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print a message when the search finds no stored chunks."""
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[]))

    main.main(["search", "alpha"])

    assert capsys.readouterr().out == "No matching chunks.\n"


def test_main__search_long_text__prints_shortened_text_with_ellipsis(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Shorten a chunk text longer than 300 characters and end it with an ellipsis."""
    _index_text(monkeypatch, tmp_path, " ".join(["word"] * 600))
    capsys.readouterr()

    main.main(["search", "word", "--limit", "1"])

    text_line = capsys.readouterr().out.splitlines()[1]
    assert text_line.endswith("...")
    assert len(text_line) <= MAX_SHORT_TEXT_CHARACTERS + len("...")


def test_main__search_full_flag__prints_whole_chunk_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the whole chunk text, without shortening, when the full flag is given."""
    _index_text(monkeypatch, tmp_path, " ".join(["word"] * 600))
    capsys.readouterr()

    main.main(["search", "word", "--limit", "1", "--full"])

    text_line = capsys.readouterr().out.splitlines()[1]
    assert len(text_line) > MAX_SHORT_TEXT_CHARACTERS
    assert not text_line.endswith("...")


def test_main__search_max_distance_below_distance__hides_all_chunks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Hide every chunk when the largest accepted distance is below the distance of the match."""
    _index_text(monkeypatch, tmp_path, "alpha beta\n")
    capsys.readouterr()

    main.main(["search", "alpha", "--max-distance", "-0.1"])

    assert capsys.readouterr().out == "No matching chunks.\n"


def test_main__search_max_distance_above_distance__keeps_chunk(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Keep the chunk when its distance is within the accepted maximum."""
    file_path = _index_text(monkeypatch, tmp_path, "alpha beta\n")
    capsys.readouterr()

    main.main(["search", "alpha", "--max-distance", "0.5"])

    assert f"file={file_path}" in capsys.readouterr().out


def test_main__search_json_flag__prints_list_with_full_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the results as a JSON list that holds the file path, the scores and the full text."""
    file_path = _index_text(monkeypatch, tmp_path, " ".join(["word"] * 600))
    capsys.readouterr()

    main.main(["search", "word", "--limit", "2", "--json"])

    chunks = json.loads(capsys.readouterr().out)
    assert len(chunks) == 2
    assert chunks[0]["file_path"] == str(file_path)
    assert chunks[0]["similarity"] == pytest.approx(1.0)
    assert len(chunks[0]["text"]) > MAX_SHORT_TEXT_CHARACTERS


def test_main__search_max_distance_option__parses_float():
    """Read the maximum distance as a number."""
    arguments = build_parser().parse_args(["search", "alpha", "--max-distance", "0.4"])

    assert arguments.max_distance == 0.4


def test_main__index_without_directories__prints_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Tell the user to add folders when neither the config nor the command line names any."""
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db"))

    main.main(["index"])

    assert NO_DIRECTORIES_MESSAGE in capsys.readouterr().out


def test_main__positional_folders__are_combined_with_directory_flag():
    """Keep the positional folders first, then the folders from the directory flag."""
    arguments = build_parser().parse_args(["index", "/data/one", "~/two", "--directory", "/data/three"])

    assert arguments.folders == [Path("/data/one"), Path("~/two")]
    assert arguments.directory == [Path("/data/three")]


def test_main__positional_folder_with_recursive__parses_both():
    """Accept the recursive flag after a positional folder."""
    arguments = build_parser().parse_args(["index", "/data/notes", "--recursive"])

    assert arguments.folders == [Path("/data/notes")]
    assert arguments.recursive is True
