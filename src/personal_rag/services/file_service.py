import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_FILE_EXTENSIONS = frozenset({".txt", ".md", ".csv"})


class FileService:
    """Find the text files in the configured directories and read their lines."""

    def __init__(self, directories: list[Path]):
        self._directories = directories

    def stream_files(self) -> Iterator[Path]:
        """Yield each text file at the top level of the configured directories.

        Yield the files of each directory in sorted order.
        Skip a directory that does not exist and log a warning.

        Yields:
            The path of one text file.
        """
        for directory in self._directories:
            if not directory.is_dir():
                logger.warning("Directory not found. Skipping it. Path=%s", directory)
                continue

            for file_path in sorted(directory.iterdir()):
                if _is_text_file(file_path):
                    yield file_path

    def stream_words(self, file_path: Path) -> Iterator[str]:
        """Yield each whitespace-separated word of one file in reading order.

        Args:
            file_path: Location of the file to read.

        Yields:
            One word of text.
        """
        for line in self.stream_lines(file_path):
            yield from line.split()

    def stream_lines(self, file_path: Path) -> Iterator[str]:
        """Yield each line of one file without reading the whole file into memory.

        Read the file as UTF-8 and replace bytes that cannot be decoded.

        Args:
            file_path: Location of the file to read.

        Yields:
            One line of text, including its line ending.
        """
        with file_path.open(encoding="utf-8", errors="replace") as file:
            yield from file


def _is_text_file(file_path: Path) -> bool:
    """Check whether a path is a file with a text file extension.

    Args:
        file_path: Path to check.

    Returns:
        True when the path is a file and its extension is a text file extension, ignoring case.
    """
    return file_path.is_file() and file_path.suffix.lower() in TEXT_FILE_EXTENSIONS
