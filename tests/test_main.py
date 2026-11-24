import json
import logging
from pathlib import Path

import pytest
from conftest import make_fake_model
from test_config_wizard import CANCEL, KEEP, ScriptedPrompter
from test_search_picker import PickingPrompter

from ostra.cli import commands, main
from ostra.cli.config.wizard import SetupResult
from ostra.cli.parser import build_parser
from ostra.cli.search_output import MAX_SHORT_TEXT_CHARACTERS
from ostra.cli.search_picker import QUIT
from ostra.constants import NO_DIRECTORIES_MESSAGE
from ostra.models.app_config import AppConfig
from ostra.services import index_session


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


def _pick_quit(monkeypatch: pytest.MonkeyPatch) -> PickingPrompter:
    """Answer the file list of the search with Quit, and return the prompter to inspect its questions."""
    prompter = PickingPrompter([QUIT])
    monkeypatch.setattr(commands, "QuestionaryPrompter", lambda: prompter)
    return prompter


def test_main__search_interactive_flag__lists_files_to_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """List each matching file once, instead of printing the chunks, when --interactive is given."""
    file_path = _index_text(monkeypatch, tmp_path, " ".join(["word"] * 600))
    capsys.readouterr()
    prompter = _pick_quit(monkeypatch)

    main.main(["search", "word", "--interactive"])

    assert [choice.value for choice in prompter.choices[0]] == [str(file_path), QUIT]
    assert capsys.readouterr().out == ""


def test_main__search_in_terminal__lists_files_to_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """List the files without a flag when a person runs the search in a terminal."""
    _index_text(monkeypatch, tmp_path, "alpha beta\n")
    prompter = _pick_quit(monkeypatch)
    monkeypatch.setattr(commands, "_is_terminal", lambda: True)

    main.main(["search", "alpha"])

    assert len(prompter.choices) == 1


@pytest.mark.parametrize("flags", [["--full"], ["--json"], ["--no-interactive"], ["--interactive", "--json"]])
def test_main__search_in_terminal_with_print_flag__prints_chunks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, flags: list[str]
):
    """Print the chunks in a terminal when --full, --json or --no-interactive asks for them."""
    _index_text(monkeypatch, tmp_path, "alpha beta\n")
    capsys.readouterr()
    prompter = _pick_quit(monkeypatch)
    monkeypatch.setattr(commands, "_is_terminal", lambda: True)

    main.main(["search", "alpha", *flags])

    assert prompter.choices == []
    assert "alpha beta" in capsys.readouterr().out


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


def test_main__no_recursive_flag__overrides_config(monkeypatch: pytest.MonkeyPatch):
    """Parse --no-recursive as False and leave the option unset when it is not given."""
    assert build_parser().parse_args(["index", "--no-recursive"]).recursive is False
    assert build_parser().parse_args(["index"]).recursive is None


def test_main__config__runs_setup_without_opening_the_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run the setup and skip load_config and the index, which the setup does not need."""
    setup_calls = []

    def fail_if_called():
        raise AssertionError("load_config must not run for init")

    monkeypatch.setattr(main, "load_config", fail_if_called)
    monkeypatch.setattr(commands, "default_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(
        commands,
        "run_setup",
        lambda prompter, path, **kwargs: (
            setup_calls.append(path)
            or SetupResult(AppConfig(), (), "potion-multilingual-128m", index_now=False, had_config_file=False)
        ),
    )

    main.main(["config", "--defaults"])

    assert setup_calls == [tmp_path / "config.json"]


def test_main__config_without_terminal__exits_with_hint(monkeypatch: pytest.MonkeyPatch):
    """Stop with a hint when the input is not a terminal and --defaults is not given."""
    monkeypatch.setattr(commands.sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit, match="ostra config --defaults"):
        main.main(["config"])


def test_main__config_defaults__writes_config_and_does_not_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Write the config file with the given folder, index nothing and remind the user to run ostra index."""
    config_path = tmp_path / "config.json"
    notes = tmp_path / "notes"
    notes.mkdir()
    monkeypatch.setattr(commands, "default_config_path", lambda: config_path)
    monkeypatch.setattr(commands, "open_index", lambda *args, **kwargs: pytest.fail("must not index"))

    main.main(["config", "--defaults", "--directory", str(notes)])

    assert json.loads(config_path.read_text())["directories"] == [str(notes)]
    assert "Run ostra index to apply the changes (folders)." in capsys.readouterr().out


def test_main__init_alias__runs_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Run the same setup for the old init name as for config, with the same output."""
    config_path = tmp_path / "config.json"
    notes = tmp_path / "notes"
    notes.mkdir()
    monkeypatch.setattr(commands, "default_config_path", lambda: config_path)
    monkeypatch.setattr(commands, "open_index", lambda *args, **kwargs: pytest.fail("must not index"))

    main.main(["init", "--defaults", "--directory", str(notes)])

    assert json.loads(config_path.read_text())["directories"] == [str(notes)]
    assert "Run ostra index to apply the changes (folders)." in capsys.readouterr().out


def _run_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answers: list[object]) -> ScriptedPrompter:
    """Run ostra config on a config file with one folder and the bge-m3 model, and answer its questions."""
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "a.txt").write_text("alpha beta\n")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "directories": [str(notes)],
                "embedding_model": "bge-m3",
                "database_path": str(tmp_path / "file.db"),
            }
        )
    )
    monkeypatch.setattr(commands.sys.stdin, "isatty", lambda: True)
    prompter = ScriptedPrompter(answers)
    monkeypatch.setattr(commands, "QuestionaryPrompter", lambda: prompter)
    monkeypatch.setattr(commands, "default_config_path", lambda: tmp_path / "config.json")
    main.main(["config"])
    return prompter


def _add_folder_answers(tmp_path: Path, save_choice: str) -> list[object]:
    """Answer the menu by adding one folder, and the save step with the given choice."""
    more = tmp_path / "more"
    more.mkdir(exist_ok=True)
    return ["folders", KEEP, str(more), "", KEEP, "save", save_choice]


KEPT_CHUNKS = "Chunks embedded so far are kept; run ostra config again to save the settings and continue."


def _saved_model(tmp_path: Path) -> str:
    """Return the embedding model that the config file holds."""
    return json.loads((tmp_path / "config.json").read_text())["embedding_model"]


def test_main__config_ctrl_c_in_menu__saves_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return without a traceback and save nothing when the user presses Ctrl-C in the menu."""
    _run_config(monkeypatch, tmp_path, [CANCEL])

    assert _saved_model(tmp_path) == "bge-m3"
    assert not (tmp_path / "file.db").exists()


def test_main__config_index_now_uses_environment_like_ostra_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Index into the database of OSTRA_DATABASE_PATH, as ostra index and ostra search do."""
    monkeypatch.setenv("OSTRA_DATABASE_PATH", str(tmp_path / "env.db"))

    _run_config(monkeypatch, tmp_path, _add_folder_answers(tmp_path, "index_now"))

    assert (tmp_path / "env.db").is_file()
    assert not (tmp_path / "file.db").exists()


def test_main__config_index_now_server_down__prints_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print a short message instead of a traceback when the embedding server cannot be reached."""

    def refuse(**kwargs):
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(commands, "run_indexing", refuse)

    _run_config(monkeypatch, tmp_path, _add_folder_answers(tmp_path, "index_now"))

    assert "Indexing stopped: Failed to connect to Ollama." in capsys.readouterr().out


def test_main__config_model_change_index_now__indexes_before_saving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Index with the new model while the config file still holds the old one, then save the new model."""
    models_seen_while_indexing: list[str] = []
    monkeypatch.setattr(
        commands,
        "run_indexing",
        lambda **kwargs: models_seen_while_indexing.append(_saved_model(tmp_path)),
    )

    _run_config(monkeypatch, tmp_path, ["model", "potion-multilingual-128m", "save", "index_now"])

    assert models_seen_while_indexing == ["bge-m3"]
    assert _saved_model(tmp_path) == "potion-multilingual-128m"
    assert (tmp_path / "file.db").is_file()


def test_main__config_model_change_index_fails__keeps_old_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Keep the old model in the config file when indexing with the new model fails."""

    def refuse(**kwargs):
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(commands, "run_indexing", refuse)

    _run_config(monkeypatch, tmp_path, ["model", "potion-multilingual-128m", "save", "index_now"])

    output = capsys.readouterr().out
    assert _saved_model(tmp_path) == "bge-m3"
    assert "Indexing stopped: Failed to connect to Ollama." in output
    assert f"The settings were not saved, so ostra still uses bge-m3. {KEPT_CHUNKS}" in output


def test_main__config_model_change_ctrl_c_while_indexing__keeps_old_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Keep the old model in the config file when the user presses Ctrl-C while indexing with the new model."""

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands, "run_indexing", interrupt)

    _run_config(monkeypatch, tmp_path, ["model", "potion-multilingual-128m", "save", "index_now"])

    output = capsys.readouterr().out
    assert _saved_model(tmp_path) == "bge-m3"
    assert "Indexing stopped.\n" in output
    assert "The settings were not saved, so ostra still uses bge-m3." in output


def test_main__config_model_change_index_now_uses_environment_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Index a model change into the database of OSTRA_DATABASE_PATH, and save the file's database path."""
    monkeypatch.setenv("OSTRA_DATABASE_PATH", str(tmp_path / "env.db"))

    _run_config(monkeypatch, tmp_path, ["model", "potion-multilingual-128m", "save", "index_now"])

    assert (tmp_path / "env.db").is_file()
    assert not (tmp_path / "file.db").exists()
    assert json.loads((tmp_path / "config.json").read_text())["database_path"] == str(tmp_path / "file.db")


def test_main__config_without_model_change_ctrl_c_while_indexing__says_settings_are_saved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Keep the saved settings and print how to continue when Ctrl-C stops indexing a change that is not a model."""

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands, "run_indexing", interrupt)

    _run_config(monkeypatch, tmp_path, _add_folder_answers(tmp_path, "index_now"))

    assert "Indexing stopped. The settings are saved; run ostra index to continue.\n" in capsys.readouterr().out
    assert json.loads((tmp_path / "config.json").read_text())["directories"][-1] == str(tmp_path / "more")


def _first_run_result(tmp_path: Path, index_now: bool = True) -> SetupResult:
    """Return the answers of a first setup that changes the model and chooses to index now."""
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    config = AppConfig.from_file_values({"directories": [notes], "embedding_model": "bge-m3"})
    return SetupResult(
        config=config,
        index_changes=("model",),
        previous_model="potion-multilingual-128m",
        index_now=index_now,
        had_config_file=False,
    )


def test_index_then_save__first_run_index_fails__says_nothing_was_saved_and_no_model_used(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Do not name a previous model on a first run, because no config file existed before."""

    def refuse(**kwargs):
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(commands, "run_indexing", refuse)
    config_path = tmp_path / "config.json"

    commands._index_then_save(_first_run_result(tmp_path), config_path)

    output = capsys.readouterr().out
    assert "Indexing stopped: Failed to connect to Ollama." in output
    assert f"The settings were not saved. {KEPT_CHUNKS}" in output
    assert output.count("Indexing stopped") == 1
    assert "still uses" not in output
    assert not config_path.exists()


def test_index_then_save__first_run_ctrl_c_while_indexing__prints_one_stop_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print the first-run message alone when the user presses Ctrl-C, without a second stop line."""

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands, "run_indexing", interrupt)
    config_path = tmp_path / "config.json"

    commands._index_then_save(_first_run_result(tmp_path), config_path)

    assert capsys.readouterr().out == f"Indexing stopped.\nThe settings were not saved. {KEPT_CHUNKS}\n"
    assert not config_path.exists()


def test_main__config_defaults_invalid_config__exits_with_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Exit with an error status when --defaults cannot read the existing config file."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not json")
    monkeypatch.setattr(commands, "default_config_path", lambda: config_path)

    with pytest.raises(SystemExit) as raised:
        main.main(["config", "--defaults"])

    assert raised.value.code == 1
    assert config_path.read_text() == "not json"


def test_main__config_nothing_changed__asks_no_index_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Ask nothing about indexing and index nothing when no setting changed."""
    prompter = _run_config(monkeypatch, tmp_path, ["save", "save"])

    assert prompter.finished()
    assert "What now?" not in prompter.questions
    assert not (tmp_path / "file.db").exists()


def test_main__config_index_later__prints_reminder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Save a model change, remind the user to run ostra index, and warn that the new model finds nothing yet."""
    _run_config(monkeypatch, tmp_path, ["model", "potion-multilingual-128m", "save", "index_later"])

    output = capsys.readouterr().out
    assert _saved_model(tmp_path) == "potion-multilingual-128m"
    assert "Run ostra index to apply the changes (model)." in output
    assert "Searches with the new model find nothing until then." in output
    assert not (tmp_path / "file.db").exists()


def test_main__config_invalid_environment_value__prints_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Print a short message instead of a traceback when an OSTRA_ variable is not valid."""
    monkeypatch.setenv("OSTRA_CHUNK_SIZE_TOKENS", "abc")

    _run_config(monkeypatch, tmp_path, ["database", False, str(tmp_path / "new.db"), "save", "index_now"])

    output = capsys.readouterr().out
    assert "Indexing stopped:" in output
    assert "Fix the problem, then run ostra index." in output


def test_main__config_setting_argument__is_optional_and_checked():
    """Accept one of the four settings as the shortcut of ostra config, and nothing else."""
    assert build_parser().parse_args(["config", "chunks"]).setting == "chunks"
    assert build_parser().parse_args(["config"]).setting is None

    with pytest.raises(SystemExit):
        build_parser().parse_args(["config", "nonsense"])


def test_main__config_missing_directory__exits_with_status_2_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Stop before any question when a --directory path is not a folder."""
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(commands, "default_config_path", lambda: config_path)

    with pytest.raises(SystemExit) as raised:
        main.main(["config", "--defaults", "--directory", str(tmp_path / "typo")])

    assert raised.value.code == 2
    assert "typo" in capsys.readouterr().err
    assert not config_path.exists()
