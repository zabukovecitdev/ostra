import json
import logging
from pathlib import Path

import pytest

from ostra import main
from ostra.config import NO_DIRECTORIES_MESSAGE, AppConfig
from ostra.embeddings.base import EmbeddingProvider
from ostra.models.embedding_model import EmbeddingModel
from ostra.models.registry import MXBAI_EMBED_LARGE
from ostra.store.database_backend import DatabaseBackend
from ostra.tokenization.base import Tokenizer


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
    monkeypatch.setattr(main, "build_embedding_model", lambda config, model_id=None: make_fake_model())


def test_main__database_ready__prints_backend_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the database path from the config after the database is ready."""
    database_path = tmp_path / "ostra.db"
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=database_path, directories=[]))

    main.main(["index"])

    assert capsys.readouterr().out == f"Database ready. Backend=sqlite\n{NO_DIRECTORIES_MESSAGE}\n"
    assert database_path.is_file()


def test_main__started__configures_info_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Configure INFO logging when the application starts."""
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(main.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs))
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[]))

    main.main(["index"])

    assert basic_config_calls == [{"level": logging.INFO}]


def test_main__directories_have_files__prints_each_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print each line of the files in the configured directories without the line ending."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\nbeta\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )

    main.main(["index"])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha beta"]


def test_main__reindex_flag__prints_lines_of_unchanged_file_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the lines of an unchanged file again when the reindex flag is given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )
    main.main(["index"])
    capsys.readouterr()

    main.main(["index", "--reindex"])

    assert capsys.readouterr().out.splitlines()[1:] == ["alpha"]


def test_main__no_reindex_flag__skips_lines_of_unchanged_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print no lines of an unchanged file when the reindex flag is not given."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("alpha\n")
    monkeypatch.setattr(
        main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[notes])
    )
    main.main(["index"])
    capsys.readouterr()

    main.main(["index"])

    assert capsys.readouterr().out.splitlines()[1:] == []


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
    arguments = main._build_parser().parse_args(["search", "alpha"])

    assert (arguments.query, arguments.limit) == ("alpha", 5)


def test_main__database_flag_chroma__creates_chroma_store_beside_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Use the Chroma backend and create its store when the database flag names chroma."""
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[]))

    main.main(["index", "--database", "chroma"])

    assert capsys.readouterr().out == f"Database ready. Backend=chroma\n{NO_DIRECTORIES_MESSAGE}\n"
    assert (tmp_path / "chroma").is_dir()
    assert not (tmp_path / "ostra.db").exists()


def test_main__config_selects_chroma__uses_chroma_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Use the backend from the config file when no database flag is given."""
    monkeypatch.setattr(
        main,
        "load_config",
        lambda: AppConfig(
            database_path=tmp_path / "ostra.db",
            directories=[],
            database_backend=DatabaseBackend.CHROMA,
        ),
    )

    main.main(["index"])

    assert capsys.readouterr().out == f"Database ready. Backend=chroma\n{NO_DIRECTORIES_MESSAGE}\n"


def test_main__database_flag_unknown__exits_with_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Exit with an error when the database flag names an unsupported backend."""
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[]))

    with pytest.raises(SystemExit):
        main.main(["index", "--database", "postgres"])


def test_main__directory_flag__processes_given_folder_instead_of_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Process the folder from the directory flag and skip the configured folders."""
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "a.txt").write_text("configured text\n")
    given = tmp_path / "given"
    given.mkdir()
    (given / "b.txt").write_text("given text\n")
    monkeypatch.setattr(
        main,
        "load_config",
        lambda: AppConfig(database_path=tmp_path / "ostra.db", directories=[configured]),
    )

    main.main(["index", "--directory", str(given)])

    output = capsys.readouterr().out
    assert "given text" in output
    assert "configured text" not in output


def test_main__directory_flag_repeated__processes_every_folder(tmp_path: Path):
    """Keep every folder when the directory flag is repeated."""
    arguments = main._build_parser().parse_args(["index", "--directory", "/data/one", "--directory", "~/two"])

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


def test_main__database_option_after_command__selects_backend():
    """Accept the database option after the command name, as well as before it."""
    arguments = main._build_parser().parse_args(["search", "alpha", "--database", "chroma"])

    assert arguments.database is DatabaseBackend.CHROMA


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
    assert len(text_line) <= main.MAX_SHORT_TEXT_CHARACTERS + len("...")


def test_main__search_full_flag__prints_whole_chunk_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the whole chunk text, without shortening, when the full flag is given."""
    _index_text(monkeypatch, tmp_path, " ".join(["word"] * 600))
    capsys.readouterr()

    main.main(["search", "word", "--limit", "1", "--full"])

    text_line = capsys.readouterr().out.splitlines()[1]
    assert len(text_line) > main.MAX_SHORT_TEXT_CHARACTERS
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
    assert len(chunks[0]["text"]) > main.MAX_SHORT_TEXT_CHARACTERS


def test_main__search_max_distance_option__parses_float():
    """Read the maximum distance as a number."""
    arguments = main._build_parser().parse_args(["search", "alpha", "--max-distance", "0.4"])

    assert arguments.max_distance == 0.4


def test_main__index_without_directories__prints_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Tell the user to add folders when neither the config nor the command line names any."""
    monkeypatch.setattr(main, "load_config", lambda: AppConfig(database_path=tmp_path / "ostra.db"))

    main.main(["index"])

    assert NO_DIRECTORIES_MESSAGE in capsys.readouterr().out
