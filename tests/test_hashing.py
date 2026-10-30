import hashlib
from pathlib import Path

from ostra.hashing import hash_file, hash_text


def test_hash_file__file_content__returns_sha256_of_raw_bytes(tmp_path: Path):
    """Return the SHA-256 digest of the raw bytes of the file."""
    file_path = tmp_path / "data.txt"
    file_path.write_bytes(b"alpha\r\nbeta\n")

    assert hash_file(file_path) == hashlib.sha256(b"alpha\r\nbeta\n").hexdigest()


def test_hash_text__line_with_line_ending__returns_sha256_of_utf8_bytes():
    """Return the SHA-256 digest of the UTF-8 bytes of the line, including its line ending."""
    assert hash_text("café\n") == hashlib.sha256("café\n".encode()).hexdigest()


def test_hash_text__lines_differ_only_in_line_ending__returns_different_digests():
    """Return different digests for lines that differ only in their line ending."""
    assert hash_text("alpha\n") != hash_text("alpha")
