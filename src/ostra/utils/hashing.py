import hashlib


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of text encoded as UTF-8.

    Args:
        text: The text to hash.

    Returns:
        The SHA-256 hex digest of the text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
