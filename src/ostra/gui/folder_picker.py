from pathlib import Path

from nicegui import ui

from ostra.gui.theme import BUTTON_PROPS, FIELD_PROPS


def list_subfolders(folder: Path) -> list[Path]:
    """Return the visible subfolders of a folder, sorted by name.

    Args:
        folder: The folder to read.

    Returns:
        The subfolders whose names do not start with a dot. An empty list when the folder cannot be read.
    """
    try:
        children = list(folder.iterdir())
    except OSError:
        return []

    return sorted(
        (child for child in children if not child.name.startswith(".") and _is_folder(child)),
        key=lambda child: child.name.lower(),
    )


def _is_folder(path: Path) -> bool:
    """Tell whether a path is a folder. A path that cannot be read counts as no folder.

    Args:
        path: The path to test.

    Returns:
        True when the path is a folder.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


class FolderPicker(ui.dialog):
    """A dialog that lets the user browse the folders of this computer and choose one.

    The web interface runs on the same computer as the browser, so the dialog lists the local folders.
    Await the dialog to get the chosen folder as a Path, or None when the user cancels.
    """

    def __init__(self, start: Path | None = None):
        super().__init__()
        self._current = start if start is not None and start.is_dir() else Path.home()
        with self, ui.card().classes("w-full").style("max-width: 560px"):
            ui.label("Choose a folder").classes("text-lg font-semibold")
            self._path_input = (
                ui.input(value=str(self._current)).classes("w-full").props(FIELD_PROPS).on("keydown.enter", self._jump)
            )
            self._list = ui.column().classes("w-full gap-0").style("max-height: 320px; overflow-y: auto")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: self.submit(None)).props(BUTTON_PROPS + " flat")
                ui.button("Select this folder", on_click=lambda: self.submit(self._current)).props(BUTTON_PROPS)
        self._refresh()

    def _jump(self) -> None:
        """Open the folder that the user typed in the path box, when it exists."""
        target = Path(self._path_input.value).expanduser()
        if target.is_dir():
            self._open(target)
        else:
            ui.notify("This folder does not exist.", type="warning")

    def _open(self, folder: Path) -> None:
        """Show the content of a folder.

        Args:
            folder: The folder to show.
        """
        self._current = folder
        self._path_input.value = str(folder)
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild the list with a parent entry and the subfolders of the current folder."""
        self._list.clear()
        with self._list:
            if self._current.parent != self._current:
                self._entry("↑  ..", self._current.parent)
            for subfolder in list_subfolders(self._current):
                self._entry(subfolder.name, subfolder)

    def _entry(self, label: str, folder: Path) -> None:
        """Add one clickable row to the list.

        Args:
            label: The text of the row.
            folder: The folder that opens when the row is clicked.
        """
        ui.item(label, on_click=lambda: self._open(folder)).classes("w-full")
