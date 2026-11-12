import inspect
import logging
from pathlib import Path

import pytest

from ostra.providers.sources.file_system_source import SOURCE_NAME, FileSystemSource
from ostra.utils.hashing import hash_file


def read_texts(source: FileSystemSource) -> list[str]:
    """Return the text of every document that the source offers, in order."""
    return [document.text for document in source.stream_documents()]


def test_stream_documents__directory_with_files__yields_text_files_in_sorted_order(tmp_path: Path):
    """Yield the text files of a directory in sorted path order."""
    (tmp_path / "b.txt").write_text("second\n")
    (tmp_path / "a.txt").write_text("first\n")

    documents = list(FileSystemSource(directories=[tmp_path]).stream_documents())

    assert [document.document_id for document in documents] == [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]


def test_stream_documents__file__fills_every_field_of_the_document(tmp_path: Path):
    """Use the path as document ID, the byte digest as version, the file name as title and the text of the file."""
    file_path = tmp_path / "a.txt"
    file_path.write_text("first\nsecond line\n")

    document = next(FileSystemSource(directories=[tmp_path]).stream_documents())

    assert document.source == SOURCE_NAME
    assert document.document_id == str(file_path)
    assert document.version == hash_file(file_path)
    assert document.title == "a.txt"
    assert document.text == "first\nsecond line\n"


def test_stream_documents__nested_directory__skips_files_in_subdirectories(tmp_path: Path):
    """Skip files in subdirectories and find only the top level of each directory."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "note.md").write_text("nested line\n")
    (tmp_path / "top.md").write_text("top line\n")

    assert read_texts(FileSystemSource(directories=[tmp_path])) == ["top line\n"]


def test_stream_documents__several_directories__yields_files_in_directory_order(tmp_path: Path):
    """Yield the files of the first directory before the files of the second directory."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("from first\n")
    (second / "b.txt").write_text("from second\n")

    assert read_texts(FileSystemSource(directories=[first, second])) == ["from first\n", "from second\n"]


def test_stream_documents__directory_missing__skips_it_and_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Skip a missing directory and log a warning that names it."""
    missing = tmp_path / "missing"

    with caplog.at_level(logging.WARNING, logger="ostra.providers.sources.file_system_source"):
        documents = list(FileSystemSource(directories=[missing]).stream_documents())

    assert documents == []
    assert [record.getMessage() for record in caplog.records] == [f"Directory not found. Skipping it. Path={missing}"]


def test_stream_documents__empty_directory__yields_nothing(tmp_path: Path):
    """Yield no documents when the directory is empty."""
    assert list(FileSystemSource(directories=[tmp_path]).stream_documents()) == []


def test_stream_documents__file_has_invalid_utf8_bytes__replaces_them_and_keeps_reading(tmp_path: Path):
    """Replace bytes that are not valid UTF-8 and keep reading the file."""
    (tmp_path / "data.txt").write_bytes(b"ok\n\xff bad\n")

    assert read_texts(FileSystemSource(directories=[tmp_path])) == ["ok\n� bad\n"]


def test_stream_documents__non_text_files__skips_them(tmp_path: Path):
    """Skip files whose extension is not a text file extension."""
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")

    assert list(FileSystemSource(directories=[tmp_path]).stream_documents()) == []


def test_stream_documents__csv_file__is_found(tmp_path: Path):
    """Find CSV files like other text files."""
    (tmp_path / "data.csv").write_text("name,score\nada,9\n")

    assert read_texts(FileSystemSource(directories=[tmp_path])) == ["name,score\nada,9\n"]


def test_stream_documents__uppercase_text_extension__is_found(tmp_path: Path):
    """Find files whose text file extension is in uppercase."""
    (tmp_path / "NOTES.TXT").write_text("loud line\n")

    assert read_texts(FileSystemSource(directories=[tmp_path])) == ["loud line\n"]


def test_stream_documents__called__returns_generator(tmp_path: Path):
    """Return a generator so that files are read one at a time."""
    assert inspect.isgenerator(FileSystemSource(directories=[tmp_path]).stream_documents())


def test_stream_documents__recursive__yields_files_in_subdirectories_in_sorted_order(tmp_path: Path):
    """Find the files of every subdirectory when the source is recursive."""
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "deep").mkdir()
    (tmp_path / "b" / "deep" / "c.md").write_text("deep line\n")
    (tmp_path / "b" / "b.md").write_text("nested line\n")
    (tmp_path / "a.md").write_text("top line\n")

    documents = list(FileSystemSource(directories=[tmp_path], recursive=True).stream_documents())

    assert [document.document_id for document in documents] == [
        str(tmp_path / "a.md"),
        str(tmp_path / "b" / "b.md"),
        str(tmp_path / "b" / "deep" / "c.md"),
    ]


def test_stream_documents__recursive_with_hidden_directory__skips_its_files(tmp_path: Path):
    """Skip the files inside hidden subdirectories, such as .git, when the source is recursive."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "notes.txt").write_text("hidden line\n")
    (tmp_path / "note.md").write_text("visible line\n")

    assert read_texts(FileSystemSource(directories=[tmp_path], recursive=True)) == ["visible line\n"]


def test_stream_documents__recursive_inside_hidden_parent__still_finds_files(tmp_path: Path):
    """Check only the subdirectories below the scanned directory, not the directory's own path."""
    root = tmp_path / ".notes"
    root.mkdir()
    (root / "note.md").write_text("line\n")

    assert read_texts(FileSystemSource(directories=[root], recursive=True)) == ["line\n"]


def test_count_documents__recursive_and_missing_directory__counts_text_files_without_reading_them(tmp_path: Path):
    """Count the text files of the existing folders, including subfolders when the source is recursive."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("a\n")
    (tmp_path / "sub" / "b.txt").write_text("b\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    assert FileSystemSource(directories=[tmp_path, tmp_path / "missing"]).count_documents() == 1
    assert FileSystemSource(directories=[tmp_path], recursive=True).count_documents() == 2
