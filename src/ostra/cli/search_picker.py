from collections.abc import Callable
from pathlib import Path

from ostra.cli.config.prompter import Choice, Prompter, SetupCancelled
from ostra.cli.result_opener import open_result
from ostra.cli.search_output import shorten_text, similarity
from ostra.models.file_match import FileMatch

# The value of the Quit choice. No file has an empty path.
QUIT = ""
PICK_MESSAGE = "Pick a file to open (Ctrl-C to quit)"


def pick_and_open(matches: list[FileMatch], prompter: Prompter, open_file: Callable[[Path], str] = open_result) -> None:
    """List the files of a search and open each file the user picks, until the user quits.

    Args:
        matches: The files of the search, nearest first.
        prompter: Asks the user to pick a file.
        open_file: Opens a file and returns a line that tells what was opened.
    """
    if not matches:
        print("No matching files.")
        return

    choices = [*(_file_choice(match) for match in matches), Choice(title="Quit", value=QUIT)]
    picked: str | None = None
    while (picked := _pick(prompter, choices, default=picked)) != QUIT:
        print(open_file(Path(picked)))


def _pick(prompter: Prompter, choices: list[Choice], default: str | None) -> str:
    """Ask for a file, and count Ctrl-C as Quit."""
    try:
        return prompter.select(PICK_MESSAGE, choices, default=default)
    except SetupCancelled:
        return QUIT


def _file_choice(match: FileMatch) -> Choice:
    """Show a file with its similarity, its path and its number of chunks, and its nearest chunk below the list."""
    count = f"{match.match_count} match" + ("es" if match.match_count > 1 else "")
    return Choice(
        title=f"{round(similarity(match.distance) * 100):>3}%  {display_path(match.document_id)}  ({count})",
        value=match.document_id,
        description=shorten_text(" ".join(match.best_text.split()), full_text=False),
    )


def display_path(document_id: str) -> str:
    """Shorten a path inside the home folder to start with ~."""
    path = Path(document_id)
    if not path.is_relative_to(Path.home()):
        return document_id

    return str(Path("~") / path.relative_to(Path.home()))
