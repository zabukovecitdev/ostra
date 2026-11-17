import os
from pathlib import Path

from ostra.constants import OWNER_ONLY_DIRECTORY_MODE, OWNER_ONLY_FILE_MODE


def ensure_owner_only_directory(directory: Path) -> None:
    """Create a folder when it is missing, and set owner-only permissions on it.

    Create the missing parent folders too.

    Args:
        directory: The folder to create or restrict.
    """
    directory.mkdir(mode=OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)
    directory.chmod(OWNER_ONLY_DIRECTORY_MODE)


def open_owner_only_file(path: Path, truncate: bool = False) -> int:
    """Open a file for writing, and create it with owner-only permissions when it is missing.

    Args:
        path: The file to open.
        truncate: Empty the file when it exists.

    Returns:
        The file descriptor. The caller closes it.
    """
    flags = os.O_CREAT | os.O_WRONLY | (os.O_TRUNC if truncate else 0)
    return os.open(path, flags, OWNER_ONLY_FILE_MODE)


def restrict_owner_only_file(path: Path) -> None:
    """Set owner-only permissions on an existing file.

    Args:
        path: The file to restrict.
    """
    path.chmod(OWNER_ONLY_FILE_MODE)
