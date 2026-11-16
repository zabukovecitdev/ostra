import os
import sys
from pathlib import Path

import pytest

from ostra.utils.owner_only import ensure_owner_only_directory, open_owner_only_file, restrict_owner_only_file

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission modes do not apply on Windows")


@POSIX_ONLY
def test_ensure_owner_only_directory__missing_parents__creates_them_with_mode_700(tmp_path: Path):
    """Create the folder and restrict it to the owner."""
    directory = tmp_path / "a" / "b"

    ensure_owner_only_directory(directory)

    assert directory.stat().st_mode & 0o777 == 0o700


@POSIX_ONLY
def test_open_owner_only_file__file_missing__creates_it_with_mode_600(tmp_path: Path):
    """Create the file that only the owner can read and write."""
    path = tmp_path / "file"

    os.close(open_owner_only_file(path))

    assert path.stat().st_mode & 0o777 == 0o600


def test_open_owner_only_file__truncate__empties_the_file(tmp_path: Path):
    """Empty an existing file when asked."""
    path = tmp_path / "file"
    path.write_text("old content")

    os.close(open_owner_only_file(path, truncate=True))

    assert path.read_text() == ""


@POSIX_ONLY
def test_restrict_owner_only_file__open_permissions__sets_mode_600(tmp_path: Path):
    """Restrict an existing file to the owner."""
    path = tmp_path / "file"
    path.write_text("x")
    path.chmod(0o644)

    restrict_owner_only_file(path)

    assert path.stat().st_mode & 0o777 == 0o600
