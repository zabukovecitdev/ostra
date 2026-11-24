import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_config_wizard import CANCEL, KEEP, MENU_QUESTION, ScriptedPrompter, make_services, read

from ostra.cli.config import wizard
from ostra.cli.config.draft import SetupContext, SetupDraft
from ostra.cli.config.menu import leave_without_saving, menu_choices
from ostra.cli.config.prompter import Choice, SetupCancelled
from ostra.cli.config.steps import fit_chunk_size
from ostra.cli.config.wizard import run_setup
from ostra.constants import DEFAULT_CHUNK_OVERLAP_TOKENS
from ostra.services.setup.ollama_check import OllamaStatus


class MenuPrompter(ScriptedPrompter):
    """Remember the titles that each menu shows."""

    def __init__(self, answers: list[object]):
        super().__init__(answers)
        self.menus: list[list[str]] = []

    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        if message == MENU_QUESTION:
            self.menus.append([choice.title for choice in choices])
        return super().select(message, choices, default)


def make_draft(model: str, chunk_size: int | None, directories: list[Path] | None = None) -> SetupDraft:
    """Build an answer draft with the given model and chunk size."""
    return SetupDraft(
        embedding_model=model,
        ollama_base_url=None,
        directories=directories or [],
        recursive=False,
        database_path=Path("ostra.db"),
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
    )


def make_notes_config(tmp_path: Path, **settings) -> tuple[Path, Path]:
    """Write a config file with one existing folder, and return the config path and the folder."""
    notes = tmp_path / "notes"
    notes.mkdir(exist_ok=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(notes)], "embedding_model": "bge-m3", **settings}))
    return config_path, notes


@pytest.fixture
def save_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every write of the config file, so that a test can count the saves."""
    calls: list[Path] = []
    real_save = wizard.save_config

    def record(config, config_path):
        calls.append(config_path)
        real_save(config, config_path)

    monkeypatch.setattr(wizard, "save_config", record)
    return calls


def test_run_setup__config_with_folders__opens_menu_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Show the settings and the menu, instead of the walkthrough, when the config file has folders."""
    config_path, _ = make_notes_config(tmp_path)
    original = config_path.read_text()
    prompter = ScriptedPrompter(["exit"])

    assert run_setup(prompter, config_path, make_services()[0]) is None

    assert prompter.questions == [MENU_QUESTION]
    assert f"Settings in {config_path}" in capsys.readouterr().out
    assert config_path.read_text() == original


def test_menu__titles_show_current_values_and_refresh(tmp_path: Path):
    """Show the current value of each setting in the menu, and the new value after a change."""
    config_path, notes = make_notes_config(tmp_path, recursive=True)
    prompter = MenuPrompter(["chunks", False, "300", "40", "save", "index_later"])

    run_setup(prompter, config_path, make_services()[0])

    first, second = prompter.menus
    assert any(title.endswith(f"{notes} (with subfolders)") for title in first)
    assert any("model default (510 / 50)" in title for title in first)
    assert any("300 / 40" in title for title in second)


def test_menu__model_change_then_index_later__saves_once_with_new_model(tmp_path: Path, save_calls: list[Path]):
    """Change the model in the menu, then save and index later: one save with the new model."""
    config_path, _ = make_notes_config(tmp_path)
    services, _ = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["model", "potion-multilingual-128m", "save", "index_later"])

    result = run_setup(prompter, config_path, services)

    assert result is not None and result.index_changes == ("model",)
    assert not result.index_now
    assert save_calls == [config_path]
    assert read(config_path)["embedding_model"] == "potion-multilingual-128m"


def test_menu__model_and_folder_changes__one_save_lists_both(tmp_path: Path, save_calls: list[Path]):
    """Make two changes in one menu session: one save, and both changes in the result."""
    config_path, notes = make_notes_config(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    services, _ = make_services([OllamaStatus(reachable=True, model_present=True)])
    answers = ["model", "mxbai-embed-large", KEEP, "folders", KEEP, str(other), "", KEEP, "save", "index_later"]

    result = run_setup(ScriptedPrompter(answers), config_path, services)

    assert result is not None and result.index_changes == ("model", "folders")
    assert len(save_calls) == 1
    assert read(config_path)["directories"] == [str(notes), str(other)]


def test_menu__exit_without_saving__writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Leave the config file unchanged when the user chooses to exit without saving."""
    config_path, _ = make_notes_config(tmp_path)
    original = config_path.read_text()

    assert run_setup(ScriptedPrompter(["exit"]), config_path, make_services()[0]) is None
    assert config_path.read_text() == original
    assert "Nothing was saved" in capsys.readouterr().out


def test_menu__ctrl_c_in_menu__writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Leave the config file unchanged when the user presses Ctrl-C in the menu."""
    config_path, _ = make_notes_config(tmp_path)
    original = config_path.read_text()

    assert run_setup(ScriptedPrompter([CANCEL]), config_path, make_services()[0]) is None
    assert config_path.read_text() == original
    assert "Nothing was saved" in capsys.readouterr().out


def test_menu__ctrl_c_after_change__writes_nothing(tmp_path: Path, save_calls: list[Path]):
    """Save nothing when the user presses Ctrl-C after a change, before the save step."""
    config_path, _ = make_notes_config(tmp_path)
    prompter = ScriptedPrompter(["chunks", False, "300", "40", CANCEL])

    assert run_setup(prompter, config_path, make_services()[0]) is None
    assert save_calls == []


def test_menu__model_shortcut__runs_model_steps_then_save(tmp_path: Path):
    """Go straight to the model steps and then the save step, without the menu."""
    config_path, _ = make_notes_config(tmp_path)
    services, calls = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["mxbai-embed-large", KEEP, "index_later"])

    result = run_setup(prompter, config_path, services, shortcut="model")

    assert result is not None and result.index_changes == ("model",)
    assert MENU_QUESTION not in prompter.questions
    assert prompter.questions[0] == "Which embedding model should ostra use?"
    assert len(calls) == 1


def test_menu__shortcut_without_folders__walks_through_instead(tmp_path: Path):
    """Ignore the shortcut and walk through every setting when the config file has no folders."""
    config_path = tmp_path / "config.json"
    prompter = ScriptedPrompter([KEEP, "", False, True, True, "save"])

    assert run_setup(prompter, config_path, make_services()[0], shortcut="chunks") is not None

    assert prompter.questions[0] == "Which embedding model should ostra use?"
    assert MENU_QUESTION not in prompter.questions


def test_menu__back_to_menu_from_save__shows_menu_again(tmp_path: Path):
    """Show the menu again when the user goes back to it from the save step."""
    config_path, _ = make_notes_config(tmp_path)
    prompter = ScriptedPrompter(["save", "menu", "exit"])

    assert run_setup(prompter, config_path, make_services()[0]) is None
    assert prompter.questions.count(MENU_QUESTION) == 2


def test_menu__directory_option__adds_folder_without_question(tmp_path: Path):
    """Add the folder of the --directory option to the configured folders in the menu."""
    config_path, notes = make_notes_config(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    prompter = ScriptedPrompter(["save", "index_later"])

    result = run_setup(prompter, config_path, make_services()[0], directories=[other])

    assert result is not None and result.config.directories == [notes, other]
    assert result.index_changes == ("folders",)


def test_menu_choices__show_defaults_for_empty_chunk_settings():
    """Show the model default for the chunk size, and the overlap, when no chunk size is set."""
    titles = [choice.title for choice in menu_choices(make_draft("potion-multilingual-128m", None))]

    assert any("model default (510 / 50)" in title for title in titles)
    assert any(title.startswith("Folders") and title.endswith("none") for title in titles)


def test_fit_chunk_size__fits__asks_nothing(tmp_path: Path):
    """Ask nothing when the chunk size fits the chosen model."""
    prompter = ScriptedPrompter([])
    context = SetupContext(prompter, make_draft("mxbai-embed-large", 300), tmp_path / "config.json")

    fit_chunk_size(context)

    assert context.draft.chunk_size_tokens == 300
    assert prompter.questions == []


def test_fit_chunk_size__too_large_and_default_accepted__clears_chunk_size(tmp_path: Path):
    """Use the model default when the chunk size does not fit and the user accepts the default."""
    prompter = ScriptedPrompter([True])
    context = SetupContext(prompter, make_draft("mxbai-embed-large", 600), tmp_path / "config.json")

    fit_chunk_size(context)

    assert context.draft.chunk_size_tokens is None
    assert prompter.questions == ["Use the model default of 510 tokens per chunk?"]


def test_fit_chunk_size__too_large_and_default_refused__asks_chunk_settings(tmp_path: Path):
    """Ask for the chunk settings when the chunk size does not fit and the user wants their own."""
    prompter = ScriptedPrompter([False, False, "300", "40"])
    context = SetupContext(prompter, make_draft("mxbai-embed-large", 600), tmp_path / "config.json")

    fit_chunk_size(context)

    assert (context.draft.chunk_size_tokens, context.draft.chunk_overlap_tokens) == (300, 40)
    assert prompter.questions[1].startswith("Use the default chunk settings")


def test_leave_without_saving__raises_cancelled(tmp_path: Path):
    """Stop the setup when a step of the exit entry runs."""
    with pytest.raises(SetupCancelled):
        leave_without_saving(SetupContext(ScriptedPrompter([]), make_draft("bge-m3", None), tmp_path / "c.json"))
