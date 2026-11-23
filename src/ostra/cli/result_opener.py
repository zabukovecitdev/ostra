from pathlib import Path
from typing import Protocol

import click


class Launcher(Protocol):
    """Open a file or folder in its default application, as click.launch does."""

    def __call__(self, url: str, wait: bool = False, locate: bool = False) -> int: ...


def open_result(path: Path, launch: Launcher = click.launch) -> str:
    """Open a file of a search in its default application, or its folder when the file cannot be opened.

    click.launch picks the command of the platform: open on macOS, the Windows shell, and xdg-open elsewhere.

    Args:
        path: The file to open.
        launch: The function that opens a file or folder. Tests pass a fake.

    Returns:
        A line that tells the user what was opened.
    """
    if path.is_file():
        return _open_file(path, launch)

    if path.parent.is_dir():
        launch(str(path.parent))
        return f"The file no longer exists. Opened its folder: {path.parent}"

    return f"The file and its folder no longer exist: {path}"


def _open_file(path: Path, launch: Launcher) -> str:
    """Open an existing file, and show it in its folder when no application opens it.

    The exit code of the folder view is not checked, because Windows Explorer returns 1 even when it works.
    """
    if launch(str(path)) == 0:
        return f"Opened {path}"

    launch(str(path), locate=True)
    return f"No application opened the file. Tried to show it in its folder: {path}"
