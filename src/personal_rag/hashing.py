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


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of text encoded as UTF-8.

    Args:
        text: The text to hash.

    Returns:
        The SHA-256 hex digest of the text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
