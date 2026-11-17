import hashlib

from ostra.utils.hashing import hash_text


def test_hash_text__line_with_line_ending__returns_sha256_of_utf8_bytes():
    """Return the SHA-256 digest of the UTF-8 bytes of the line, including its line ending."""
    assert hash_text("café\n") == hashlib.sha256("café\n".encode()).hexdigest()


def test_hash_text__lines_differ_only_in_line_ending__returns_different_digests():
    """Return different digests for lines that differ only in their line ending."""
    assert hash_text("alpha\n") != hash_text("alpha")
