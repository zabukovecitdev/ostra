from pathlib import Path

from ostra.cli.result_opener import open_result


class FakeLauncher:
    """Record the calls of click.launch and return the given exit codes in order."""

    def __init__(self, *exit_codes: int):
        self._exit_codes = list(exit_codes)
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, url: str, wait: bool = False, locate: bool = False) -> int:
        self.calls.append((url, locate))
        return self._exit_codes.pop(0) if self._exit_codes else 0


def test_open_result__file_exists__opens_file(tmp_path: Path):
    """Open the file in its default application."""
    file = tmp_path / "a.md"
    file.write_text("alpha")
    launch = FakeLauncher(0)

    message = open_result(file, launch)

    assert launch.calls == [(str(file), False)]
    assert message == f"Opened {file}"


def test_open_result__no_application_opens_file__shows_it_in_its_folder(tmp_path: Path):
    """Show the file in the file manager when opening it fails."""
    file = tmp_path / "a.md"
    file.write_text("alpha")
    launch = FakeLauncher(1)

    message = open_result(file, launch)

    assert launch.calls == [(str(file), False), (str(file), True)]
    assert str(file) in message


def test_open_result__file_deleted__opens_its_folder(tmp_path: Path):
    """Open the folder of a file that was deleted after indexing."""
    launch = FakeLauncher()

    message = open_result(tmp_path / "gone.md", launch)

    assert launch.calls == [(str(tmp_path), False)]
    assert message == f"The file no longer exists. Opened its folder: {tmp_path}"


def test_open_result__folder_deleted__opens_nothing(tmp_path: Path):
    """Only report the path when neither the file nor its folder exists."""
    launch = FakeLauncher()
    path = tmp_path / "gone" / "a.md"

    message = open_result(path, launch)

    assert launch.calls == []
    assert message == f"The file and its folder no longer exist: {path}"
