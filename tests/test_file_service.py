import inspect
import logging
from pathlib import Path

import pytest

from personal_rag.services.file_service import FileService


def read_all_lines(service: FileService) -> list[str]:
    """Return the lines of every file that the service finds, in order."""
    return [line for file_path in service.stream_files() for line in service.stream_lines(file_path)]


def test_stream_files__directory_with_files__yields_text_files_in_sorted_order(tmp_path: Path):
    """Yield the text files of a directory in sorted path order."""
    (tmp_path / "b.txt").write_text("second\n")
    (tmp_path / "a.txt").write_text("first\n")
    service = FileService(directories=[tmp_path])

    assert list(service.stream_files()) == [tmp_path / "a.txt", tmp_path / "b.txt"]


def test_stream_lines__file_with_lines__yields_each_line_with_line_ending(tmp_path: Path):
    """Yield each line of a file in order, keeping its line ending."""
    (tmp_path / "a.txt").write_text("first\nsecond line\n")
    service = FileService(directories=[tmp_path])

    assert list(service.stream_lines(tmp_path / "a.txt")) == ["first\n", "second line\n"]


def test_stream_files__nested_directory__skips_files_in_subdirectories(tmp_path: Path):
    """Skip files in subdirectories and find only the top level of each directory."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "note.md").write_text("nested line\n")
    (tmp_path / "top.md").write_text("top line\n")
    service = FileService(directories=[tmp_path])

    assert read_all_lines(service) == ["top line\n"]


def test_stream_files__several_directories__yields_files_in_directory_order(tmp_path: Path):
    """Yield the files of the first directory before the files of the second directory."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("from first\n")
    (second / "b.txt").write_text("from second\n")
    service = FileService(directories=[first, second])

    assert read_all_lines(service) == ["from first\n", "from second\n"]


def test_stream_files__directory_missing__skips_it_and_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Skip a missing directory and log a warning that names it."""
    missing = tmp_path / "missing"
    service = FileService(directories=[missing])

    with caplog.at_level(logging.WARNING, logger="personal_rag.services.file_service"):
        files = list(service.stream_files())

    assert files == []
    assert [record.getMessage() for record in caplog.records] == [f"Directory not found. Skipping it. Path={missing}"]


def test_stream_files__empty_directory__yields_nothing(tmp_path: Path):
    """Yield no files when the directory is empty."""
    service = FileService(directories=[tmp_path])

    assert list(service.stream_files()) == []


def test_stream_lines__file_has_invalid_utf8_bytes__replaces_them_and_keeps_reading(tmp_path: Path):
    """Replace bytes that are not valid UTF-8 and keep reading the file."""
    (tmp_path / "data.txt").write_bytes(b"ok\n\xff bad\n")
    service = FileService(directories=[tmp_path])

    assert list(service.stream_lines(tmp_path / "data.txt")) == ["ok\n", "� bad\n"]


def test_stream_files__non_text_files__skips_them(tmp_path: Path):
    """Skip files whose extension is not a text file extension."""
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    service = FileService(directories=[tmp_path])

    assert list(service.stream_files()) == []


def test_stream_files__csv_file__is_found(tmp_path: Path):
    """Find CSV files and read them line by line like other text files."""
    (tmp_path / "data.csv").write_text("name,score\nada,9\n")
    service = FileService(directories=[tmp_path])

    assert read_all_lines(service) == ["name,score\n", "ada,9\n"]


def test_stream_files__uppercase_text_extension__is_found(tmp_path: Path):
    """Find files whose text file extension is in uppercase."""
    (tmp_path / "NOTES.TXT").write_text("loud line\n")
    service = FileService(directories=[tmp_path])

    assert read_all_lines(service) == ["loud line\n"]


def test_stream_files__called__returns_generator(tmp_path: Path):
    """Return a generator so that files are found one at a time."""
    service = FileService(directories=[tmp_path])

    assert inspect.isgenerator(service.stream_files())
