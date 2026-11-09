import logging
from collections.abc import Iterator
from pathlib import Path

from ostra.hashing import hash_file
from ostra.sources.base import Document, DocumentSource

logger = logging.getLogger(__name__)

SOURCE_NAME = "filesystem"
TEXT_FILE_EXTENSIONS = frozenset({".txt", ".md", ".csv"})


class FileSystemSource(DocumentSource):
    """Offer the text files of folders as documents.

    The document ID of a file is its path. The version of a file is the digest of its bytes.
    """

    def __init__(self, directories: list[Path], recursive: bool = False):
        self._directories = directories
        self._recursive = recursive

    def count_documents(self) -> int | None:
        """Count the text files that stream_documents would yield, without reading them.

        Returns:
            The number of text files in the folders that exist.
        """
        return sum(
            1
            for directory in self._directories
            if directory.is_dir()
            for file_path in self._list_paths(directory)
            if _is_text_file(file_path)
        )

    def stream_documents(self) -> Iterator[Document]:
        """Yield each text file of the folders as one document.

        Find only the top level of each folder, or every subfolder too when the source is recursive. A recursive
        scan skips hidden subfolders, such as .git. Yield the files of each folder in sorted path order. Skip a
        folder that does not exist and log a warning.
        Read each file as UTF-8 and replace bytes that cannot be decoded.

        Yields:
            One document for each text file.
        """
        for directory in self._directories:
            if not directory.is_dir():
                logger.warning("Directory not found. Skipping it. Path=%s", directory)
                continue

            for file_path in self._list_paths(directory):
                if _is_text_file(file_path):
                    yield _read_document(file_path)

    def _list_paths(self, directory: Path) -> list[Path]:
        """List the paths in a folder in sorted order.

        Args:
            directory: Folder to list.

        Returns:
            The paths at the top level of the folder, or the paths in every subfolder that is not hidden when the
            source is recursive.
        """
        if not self._recursive:
            return sorted(directory.iterdir())
        return sorted(path for path in directory.rglob("*") if not _is_in_hidden_folder(path, directory))


def _read_document(file_path: Path) -> Document:
    """Read one file into a document.

    Args:
        file_path: Location of the file to read.

    Returns:
        The document with the full text of the file.
    """
    return Document(
        source=SOURCE_NAME,
        document_id=str(file_path),
        version=hash_file(file_path),
        title=file_path.name,
        text=file_path.read_text(encoding="utf-8", errors="replace"),
    )


def _is_in_hidden_folder(path: Path, directory: Path) -> bool:
    """Check whether a path lies inside a hidden subfolder of a folder.

    Args:
        path: Path inside the folder.
        directory: The folder that was scanned.

    Returns:
        True when a subfolder between the folder and the path has a name that starts with a dot.
    """
    return any(part.startswith(".") for part in path.relative_to(directory).parts[:-1])


def _is_text_file(file_path: Path) -> bool:
    """Check whether a path is a file with a text file extension.

    Args:
        file_path: Path to check.

    Returns:
        True when the path is a file and its extension is a text file extension, ignoring case.
    """
    return file_path.is_file() and file_path.suffix.lower() in TEXT_FILE_EXTENSIONS
