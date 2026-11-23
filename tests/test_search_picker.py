from collections.abc import Sequence
from pathlib import Path

import pytest

from ostra.cli.init.prompter import Choice, Prompter, SetupCancelled, Validator
from ostra.cli.search_output import MAX_SHORT_TEXT_CHARACTERS
from ostra.cli.search_picker import PICK_MESSAGE, QUIT, display_path, pick_and_open
from ostra.models.file_match import FileMatch

CANCEL = object()  # Answer like Ctrl-C.


class PickingPrompter(Prompter):
    """Pick the given values in order, and remember the choices and defaults of each question."""

    def __init__(self, answers: Sequence[object]):
        self._answers = list(answers)
        self.choices: list[list[Choice]] = []
        self.defaults: list[str | None] = []

    def select(self, message: str, choices: Sequence[Choice], default: str | None = None) -> str:
        assert message == PICK_MESSAGE
        self.choices.append(list(choices))
        self.defaults.append(default)
        answer = self._answers.pop(0)
        if answer is CANCEL:
            raise SetupCancelled
        assert answer in [choice.value for choice in choices]
        return str(answer)

    def text(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        raise AssertionError("The picker asks no text.")

    def path(self, message: str, default: str = "", validate: Validator | None = None) -> str:
        raise AssertionError("The picker asks no path.")

    def confirm(self, message: str, default: bool = True) -> bool:
        raise AssertionError("The picker asks no confirmation.")


def file_match(document_id: str, distance: float = 0.13, match_count: int = 1, text: str = "alpha") -> FileMatch:
    """Build a file match."""
    return FileMatch(
        document_id=document_id, title=document_id, distance=distance, match_count=match_count, best_text=text
    )


class OpenRecorder:
    """Stand in for open_result and remember the opened paths."""

    def __init__(self):
        self.opened: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.opened.append(path)
        return f"opened {path}"


def test_pick_and_open__pick_file_then_quit__opens_that_file(capsys: pytest.CaptureFixture[str]):
    """Open the picked file, print what was opened, and stop on Quit."""
    opener = OpenRecorder()
    prompter = PickingPrompter(["/notes/a.md", QUIT])

    pick_and_open([file_match("/notes/a.md"), file_match("/notes/b.md")], prompter, opener)

    assert opener.opened == [Path("/notes/a.md")]
    assert capsys.readouterr().out == f"opened {Path('/notes/a.md')}\n"


def test_pick_and_open__two_picks__keeps_list_open_on_last_pick():
    """Ask again after each open, with the cursor on the file picked last."""
    opener = OpenRecorder()
    prompter = PickingPrompter(["/notes/a.md", "/notes/b.md", QUIT])

    pick_and_open([file_match("/notes/a.md"), file_match("/notes/b.md")], prompter, opener)

    assert opener.opened == [Path("/notes/a.md"), Path("/notes/b.md")]
    assert prompter.defaults == [None, "/notes/a.md", "/notes/b.md"]


def test_pick_and_open__ctrl_c__stops_without_opening():
    """Count Ctrl-C as Quit."""
    opener = OpenRecorder()

    pick_and_open([file_match("/notes/a.md")], PickingPrompter([CANCEL]), opener)

    assert opener.opened == []


def test_pick_and_open__no_matches__prints_message_without_asking(capsys: pytest.CaptureFixture[str]):
    """Print a message and ask nothing when the search found no files."""
    prompter = PickingPrompter([])

    pick_and_open([], prompter, OpenRecorder())

    assert capsys.readouterr().out == "No matching files.\n"
    assert prompter.choices == []


def test_pick_and_open__choices__show_similarity_path_count_and_preview():
    """Show each file with its similarity and number of chunks, its chunk as a preview, and Quit last."""
    prompter = PickingPrompter([QUIT])
    long_text = "word\n" * MAX_SHORT_TEXT_CHARACTERS

    pick_and_open(
        [file_match("/notes/a.md", distance=0.13, match_count=3, text=long_text), file_match("/notes/b.md")],
        prompter,
        OpenRecorder(),
    )

    first, second, quit_choice = prompter.choices[0]
    assert first.title == f" 87%  {Path('/notes/a.md')}  (3 matches)"
    assert first.value == "/notes/a.md"
    assert first.description.startswith("word word")
    assert first.description.endswith("...")
    assert second.title.endswith("(1 match)")
    assert (quit_choice.title, quit_choice.value) == ("Quit", QUIT)


def test_display_path__inside_home__starts_with_tilde():
    """Shorten a path in the home folder."""
    assert display_path(str(Path.home() / "notes" / "a.md")) == str(Path("~") / "notes" / "a.md")


def test_display_path__outside_home__stays_unchanged():
    """Keep a path outside the home folder as it is."""
    assert display_path("/srv/notes/a.md") == "/srv/notes/a.md"
