import hashlib
from pathlib import Path


def hash_file(file_path: Path) -> str:
    """Return the SHA-256 hex digest of the raw bytes of a file.

    Args:
        file_path: Location of the file to hash.

    Returns:
        The SHA-256 hex digest of the file content.
    """
    with file_path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def hash_line(line: str) -> str:
    """Return the SHA-256 hex digest of one line encoded as UTF-8.

    Args:
        line: The line of text to hash, including its line ending.

    Returns:
        The SHA-256 hex digest of the line.
    """
    return hashlib.sha256(line.encode("utf-8")).hexdigest()
