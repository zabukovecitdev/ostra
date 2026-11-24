import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from ostra.cli.config.draft import SetupServices
from ostra.cli.config.prompter import Choice, DefaultsPrompter, Prompter, Validator
from ostra.cli.config.steps import CHANGE_ADDRESS, CONTINUE, RETRY
from ostra.cli.config.wizard import run_setup
from ostra.constants import DEFAULT_CHUNK_OVERLAP_TOKENS
from ostra.services.setup.ollama_check import OllamaStatus

KEEP = object()  # Answer with the default of the question.


class ScriptedPrompter(Prompter):
    """Answer the questions from a list, in order, and remember the questions and the rejected answers."""

    def __init__(self, answers: Sequence[object]):
        self._answers = list(answers)
        self.questions: list[str] = []
        self.errors: list[str] = []

    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        answer = self._next(message, default)
        assert answer in [choice.value for choice in choices]
        return str(answer)

    def text(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        while True:
            answer = str(self._next(message, default))
            error = validate(answer) if validate else None
            if error is None:
                return answer
            self.errors.append(error)

    def path(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        return self.text(message, default, validate)

    def confirm(self, message: str, default: bool = True) -> bool:
        return bool(self._next(message, default))

    def _next(self, message: str, default: object) -> object:
        self.questions.append(message)
        assert self._answers, f"No answer left for: {message}"
        answer = self._answers.pop(0)
        return default if answer is KEEP else answer

    def finished(self) -> bool:
        return not self._answers


def make_services(statuses: Sequence[OllamaStatus] = ()) -> tuple[SetupServices, list]:
    """Build a fake Ollama check and a list that records its calls."""
    calls: list = []
    remaining = list(statuses)

    def check(url: str, model: str) -> OllamaStatus:
        calls.append(("check", url, model))
        return remaining.pop(0)

    return SetupServices(check_ollama=check), calls


@pytest.fixture(autouse=True)
def no_ostra_environment(monkeypatch: pytest.MonkeyPatch):
    """Remove OSTRA_ variables of the developer's shell, so that they do not change the results."""
    for name in [name for name in os.environ if name.upper().startswith("OSTRA_")]:
        monkeypatch.delenv(name)


def read(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def test_run_setup__default_model_with_one_folder__saves_settings(tmp_path: Path):
    """Save the chosen folder and the subfolder answer with the default model and chunk settings."""
    notes = tmp_path / "notes"
    notes.mkdir()
    config_path = tmp_path / "config.json"
    prompter = ScriptedPrompter(["potion-multilingual-128m", str(notes), "", True, True, True, True])

    result = run_setup(prompter, config_path, make_services()[0])

    assert result is not None
    saved = read(config_path)
    assert saved["embedding_model"] == "potion-multilingual-128m"
    assert saved["directories"] == [str(notes)]
    assert saved["recursive"] is True
    assert "chunk_size_tokens" not in saved
    assert saved["chunk_overlap_tokens"] == DEFAULT_CHUNK_OVERLAP_TOKENS
    assert prompter.finished()


def test_run_setup__cancelled_at_summary__writes_no_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Write nothing when the user does not confirm the summary."""
    config_path = tmp_path / "config.json"
    prompter = ScriptedPrompter([KEEP, "", False, True, True, False])

    assert run_setup(prompter, config_path, make_services()[0]) is None
    assert not config_path.exists()
    assert "Nothing was saved" in capsys.readouterr().out


def test_run_setup__missing_folder__asks_again(tmp_path: Path):
    """Reject a folder that does not exist and accept the next answer."""
    notes = tmp_path / "notes"
    notes.mkdir()
    prompter = ScriptedPrompter([KEEP, str(tmp_path / "missing"), str(notes), "", False, True, True, True])

    result = run_setup(prompter, tmp_path / "config.json", make_services()[0])

    assert result is not None and result.config.directories == [notes]
    assert prompter.errors == ["The folder does not exist."]


def test_run_setup__existing_config__offers_current_values(tmp_path: Path):
    """Keep the folders and the model of an existing config file when the user presses Enter."""
    notes = tmp_path / "notes"
    notes.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(notes)], "recursive": True}))
    prompter = ScriptedPrompter([KEEP, KEEP, "", KEEP, KEEP, KEEP, KEEP])

    result = run_setup(prompter, config_path, make_services()[0])

    assert result is not None
    assert (result.config.directories, result.config.recursive) == ([notes], True)


def test_run_setup__custom_chunk_size_too_large__asks_again(tmp_path: Path):
    """Reject a chunk size above the model input and save the next valid answer."""
    config_path = tmp_path / "config.json"
    prompter = ScriptedPrompter([KEEP, "", False, False, "9999", "300", "40", True, True])

    run_setup(prompter, config_path, make_services()[0])

    assert prompter.errors == ["Enter a number from 1 to 512."]
    saved = read(config_path)
    assert (saved["chunk_size_tokens"], saved["chunk_overlap_tokens"]) == (300, 40)


def test_run_setup__overlap_not_below_chunk_size__asks_again(tmp_path: Path):
    """Reject an overlap that is as large as the chunk size."""
    prompter = ScriptedPrompter([KEEP, "", False, False, "100", "100", "abc", "10", True, True])

    result = run_setup(prompter, tmp_path / "config.json", make_services()[0])

    assert result is not None and result.config.chunk_overlap_tokens == 10
    assert prompter.errors == ["Enter a number from 0 to 99.", "Enter a whole number."]


def test_run_setup__ollama_model_present__saves_address(tmp_path: Path):
    """Check the Ollama server for the model and save an address other than the default."""
    services, calls = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["bge-m3", "http://gpu-box:11434", "", False, True, True, True])

    result = run_setup(prompter, tmp_path / "config.json", services)

    assert result is not None and result.config.ollama_base_url == "http://gpu-box:11434"
    assert calls == [("check", "http://gpu-box:11434", "bge-m3:latest")]


def test_run_setup__default_ollama_address_kept__saves_no_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Save no address when the user keeps the default, so that OLLAMA_BASE_URL keeps working later."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    services, calls = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["bge-m3", KEEP, "", False, True, True, True])

    run_setup(prompter, tmp_path / "config.json", services)

    assert "ollama_base_url" not in read(tmp_path / "config.json")
    assert calls == [("check", "http://localhost:11434", "bge-m3:latest")]


def test_run_setup__defaults_with_unreachable_ollama__checks_once_and_saves(tmp_path: Path):
    """Check the server once and go on when no person can choose what to do next."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embedding_model": "bge-m3"}))
    services, calls = make_services([OllamaStatus(reachable=False, error="refused")])

    result = run_setup(DefaultsPrompter(), config_path, services)

    assert result is not None
    assert len(calls) == 1


def test_run_setup__ollama_model_missing__prints_pull_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Show the pull command when the server lacks the model, and save when the user goes on."""
    services, _ = make_services([OllamaStatus(reachable=True, model_present=False)])
    prompter = ScriptedPrompter(["mxbai-embed-large", KEEP, CONTINUE, "", False, True, True, True])

    result = run_setup(prompter, tmp_path / "config.json", services)

    assert result is not None
    assert "ollama pull mxbai-embed-large" in capsys.readouterr().out


def test_run_setup__ollama_unreachable__retries_and_changes_address(tmp_path: Path):
    """Check again, then ask for a new address, until the server answers."""
    services, calls = make_services(
        [
            OllamaStatus(reachable=False, error="refused"),
            OllamaStatus(reachable=False, error="refused"),
            OllamaStatus(reachable=True, model_present=True),
        ]
    )
    answers = ["bge-m3", "http://localhost:11434", RETRY, CHANGE_ADDRESS, "http://192.168.1.20:11434"]
    prompter = ScriptedPrompter([*answers, "", False, True, True, True])

    result = run_setup(prompter, tmp_path / "config.json", services)

    assert result is not None and result.config.ollama_base_url == "http://192.168.1.20:11434"
    assert [call[1] for call in calls] == [
        "http://localhost:11434",
        "http://localhost:11434",
        "http://192.168.1.20:11434",
    ]


def test_run_setup__invalid_ollama_address__asks_again(tmp_path: Path):
    """Reject an address without a scheme."""
    services, _ = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["bge-m3", "localhost", "http://localhost:11434", "", False, True, True, True])

    run_setup(prompter, tmp_path / "config.json", services)

    assert prompter.errors == ["Enter an address such as http://localhost:11434."]


def test_run_setup__new_database_path__saves_it(tmp_path: Path):
    """Save the database path that the user enters."""
    database = tmp_path / "data" / "notes.db"
    prompter = ScriptedPrompter([KEEP, "", False, True, False, str(database), True])

    result = run_setup(prompter, tmp_path / "config.json", make_services()[0])

    assert result is not None and result.config.database_path == database


def test_run_setup__environment_variable_set__warns_and_does_not_save_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Warn about an OSTRA_ variable and save the answer, not the variable."""
    monkeypatch.setenv("OSTRA_EMBEDDING_MODEL", "bge-m3")
    config_path = tmp_path / "config.json"

    run_setup(DefaultsPrompter(), config_path, make_services()[0])

    assert "OSTRA_EMBEDDING_MODEL" in capsys.readouterr().out
    assert read(config_path)["embedding_model"] == "potion-multilingual-128m"


def test_run_setup__defaults_with_directory__saves_folder_once(tmp_path: Path):
    """Add a folder of the --directory option to the configured folders without a duplicate."""
    notes = tmp_path / "notes"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(notes)]}))

    result = run_setup(DefaultsPrompter(), config_path, make_services()[0], directories=[notes, tmp_path / "more"])

    assert result is not None and result.config.directories == [notes, tmp_path / "more"]


def test_run_setup__invalid_config_without_questions__keeps_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Leave a config file that cannot be read unchanged when no person can agree to replace it."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not json")

    assert run_setup(DefaultsPrompter(), config_path, make_services()[0]) is None
    assert config_path.read_text() == "not json"
    assert "could not be read" in capsys.readouterr().out


def test_run_setup__invalid_config_user_keeps_it__saves_nothing(tmp_path: Path):
    """Cancel when the user does not want to replace a config file that cannot be read."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"chunk_size_tokens": "abc"}')

    assert run_setup(ScriptedPrompter([False]), config_path, make_services()[0]) is None
    assert config_path.read_text() == '{"chunk_size_tokens": "abc"}'


def test_run_setup__invalid_config_user_replaces_it__saves_new_settings(tmp_path: Path):
    """Start from the defaults and save when the user agrees to replace the file."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not json")
    prompter = ScriptedPrompter([True, KEEP, "", False, True, True, True])

    assert run_setup(prompter, config_path, make_services()[0]) is not None
    assert read(config_path)["embedding_model"] == "potion-multilingual-128m"


def test_run_setup__configured_folders_replaced__keeps_given_folders(tmp_path: Path):
    """Keep the folders of the --directory option when the user drops the configured folders."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(tmp_path / "old")]}))
    prompter = ScriptedPrompter([KEEP, False, "", False, True, True, True])

    result = run_setup(prompter, config_path, make_services()[0], directories=[tmp_path / "given"])

    assert result is not None and result.config.directories == [tmp_path / "given"]


def test_run_setup__nothing_changed__needs_no_index(tmp_path: Path):
    """Report no change when the user keeps every setting of an existing config file."""
    notes = tmp_path / "notes"
    notes.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(notes)]}))

    result = run_setup(DefaultsPrompter(), config_path, make_services()[0])

    assert result is not None and result.index_changes == ()


def test_run_setup__chunk_size_changed__needs_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Name the chunk size as a change that needs indexing, in the summary and the result."""
    prompter = ScriptedPrompter([KEEP, "", False, False, "300", "40", True, True])

    result = run_setup(prompter, tmp_path / "config.json", make_services()[0])

    assert result is not None
    assert result.index_changes == ("chunk size", "chunk overlap")
    assert not result.model_changed
    assert "Needs ostra index   chunk size, chunk overlap" in capsys.readouterr().out


def test_run_setup__model_changed__reports_model_change(tmp_path: Path):
    """Report a model change, which leaves the new model with an empty index."""
    services, _ = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["bge-m3", KEEP, "", False, True, True, True])

    result = run_setup(prompter, tmp_path / "config.json", services)

    assert result is not None and result.model_changed


def test_run_setup__folders_in_other_order__is_no_change(tmp_path: Path):
    """Ignore the order of the folders when telling whether they changed."""
    one, two = tmp_path / "one", tmp_path / "two"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": [str(one), str(two)]}))
    prompter = ScriptedPrompter([KEEP, False, str(two), str(one), "", False, True, True, True])
    one.mkdir()
    two.mkdir()

    result = run_setup(prompter, config_path, make_services()[0])

    assert result is not None and result.index_changes == ()


def test_run_setup__relative_configured_folder_given_again__lists_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Count a relative configured folder and its absolute form from --directory as one folder."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"directories": ["notes"]}))

    result = run_setup(DefaultsPrompter(), config_path, make_services()[0], directories=[tmp_path / "notes"])

    assert result is not None and result.config.directories == [tmp_path / "notes"]


def test_run_setup__lowercase_environment_variable__warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Warn about a lowercase ostra_ variable too, because it overrides the setting as well."""
    monkeypatch.setenv("ostra_recursive", "true")

    run_setup(DefaultsPrompter(), tmp_path / "config.json", make_services()[0])

    # Windows stores environment variable names in upper case, so the warning names OSTRA_RECURSIVE there.
    assert "ostra_recursive" in capsys.readouterr().out.lower()


def test_run_setup__address_from_environment_kept__saves_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Save the address that OLLAMA_BASE_URL offered, so that a shell without the variable uses it too."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu:11434")
    services, _ = make_services([OllamaStatus(reachable=True, model_present=True)])
    prompter = ScriptedPrompter(["bge-m3", KEEP, "", False, True, True, True])

    run_setup(prompter, tmp_path / "config.json", services)

    assert read(tmp_path / "config.json")["ollama_base_url"] == "http://gpu:11434"
